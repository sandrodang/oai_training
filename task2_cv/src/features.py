"""Trich xuat patch feature, giu ti le anh goc. Streaming - khong cache feature tho."""
import io, glob, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from PIL.JpegImagePlugin import get_sampling
from torch.utils.data import Dataset, DataLoader

# ---- Chuan hoa mien nen JPEG ----------------------------------------------
# Train duoc nen quality ~75 (QT_lum_mean 29.0), test quality ~95 (5.77).
# => moi anh test (KE CA anh normal) co tan so cao sac net hon moi anh train,
#    memory bank hoc tu q75 se cham toan bo test la "la" -> nhieu diem gia.
# Khac phuc: dua MOI anh (train/holdout/test) qua DUNG MOT chuoi bien doi xac dinh
#    decode -> encode(QT_test) -> encode(QT_train) -> decode
# Uniform, tu dong, khong phu thuoc tung sample.
_REF = {}
def _ref_tables(root):
    if _REF: return _REF
    tr = sorted(glob.glob(os.path.join(root, "dataset_train", "train", "*", "*.jpg")))[0]
    te = sorted(glob.glob(os.path.join(root, "public_test", "images", "*", "*.jpg")))[0]
    for k, f in (("train", tr), ("test", te)):
        im = Image.open(f); im.load()
        _REF[k] = ([list(im.quantization[i]) for i in sorted(im.quantization)], get_sampling(im))
    return _REF

def jpeg_chain(im, root):
    r = _ref_tables(root)
    for k in ("test", "train"):
        qt, sub = r[k]
        b = io.BytesIO()
        im.save(b, "JPEG", qtables=qt, subsampling=sub)
        b.seek(0); im = Image.open(b); im.load()
    return im.convert("RGB")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def target_hw(w, h, long_side, mult):
    """Resize giu ti le: canh dai = long_side, lam tron ve boi so mult.
    KHONG center-crop -> khong cat mat vat the sat ria (quan trong voi cat 03/05)."""
    s = long_side / max(w, h)
    return (max(mult, int(round(h * s / mult)) * mult),
            max(mult, int(round(w * s / mult)) * mult))

class ImgDS(Dataset):
    def __init__(self, files, long_side, mult, flip=None, jnorm=None):
        self.files, self.long_side, self.mult, self.flip = files, long_side, mult, flip
        self.jnorm = jnorm          # root cua repo; None = tat chuan hoa
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB")
        if self.jnorm: im = jpeg_chain(im, self.jnorm)
        th, tw = target_hw(im.width, im.height, self.long_side, self.mult)
        im = im.resize((tw, th), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.
        if self.flip == "h":   x = torch.flip(x, [2])
        elif self.flip == "v": x = torch.flip(x, [1])
        elif self.flip == "hv": x = torch.flip(x, [1, 2])
        return (x - MEAN) / STD

_MODELS = {}
def get_model(name, device):
    if name in _MODELS: return _MODELS[name]
    import timm
    if name == "wrn50":
        m = timm.create_model("wide_resnet50_2.tv2_in1k", pretrained=True,
                              features_only=True, out_indices=(2, 3))
        meta = {"mult": 32, "kind": "cnn", "dim": 1536}
    elif name in ("dinov2b", "dinov2b_ml", "dinov2l_ml", "dinov2g_ml"):
        hub = {"dinov2l_ml": "vit_large_patch14_reg4_dinov2.lvd142m",
               "dinov2g_ml": "vit_giant_patch14_reg4_dinov2.lvd142m"}.get(
                   name, "vit_base_patch14_reg4_dinov2.lvd142m")
        m = timm.create_model(hub, pretrained=True, num_classes=0, dynamic_img_size=True)
        nb = len(m.blocks)
        meta = {"mult": 14, "kind": "vit", "prefix": m.num_prefix_tokens,
                # nhieu tang trung gian: tang nong giu chi tiet cuc bo, tang sau giu ngu nghia
                "layers": None if name == "dinov2b" else [nb - 9, nb - 6, nb - 3]}
    elif name == "convnext":
        # ho kien truc CNN KHAC han ResNet -> them da dang that su, khong trung lap WRN50
        m = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True,
                              features_only=True, out_indices=(1, 2))
        meta = {"mult": 32, "kind": "cnn"}
    else:
        raise ValueError(name)
    m = m.eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)
    _MODELS[name] = (m, meta)
    return _MODELS[name]

def _cnn_feats(m, x):
    """layer2 + layer3, avgpool 3x3 (PatchCore local aggregation), gop o luoi layer2."""
    f2, f3 = m(x)
    f2 = F.avg_pool2d(f2, 3, 1, 1)
    f3 = F.avg_pool2d(f3, 3, 1, 1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([f2, f3], 1).flatten(2).transpose(1, 2)

@torch.no_grad()
def iter_feats(files, model_name, long_side, device, flip=None, bs=8, nw=8, jnorm=None):
    """Yield (patch B,N,D half CPU, cls B,D float CPU) theo tung batch. Khong giu het trong RAM."""
    m, meta = get_model(model_name, device)
    dl = DataLoader(ImgDS(files, long_side, meta["mult"], flip, jnorm), batch_size=bs,
                    shuffle=False, num_workers=nw, pin_memory=True)
    for x in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if meta["kind"] == "cnn":
                f = _cnn_feats(m, x); cls = f.mean(1)
            elif meta.get("layers"):
                outs = m.get_intermediate_layers(x, n=meta["layers"], norm=True)
                f = torch.cat(outs, dim=-1); cls = f.mean(1)
            else:
                t = m.forward_features(x)
                cls, f = t[:, 0], t[:, meta["prefix"]:]
        yield f.half().cpu(), cls.float().cpu()
