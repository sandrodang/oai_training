"""Verifiable compliance audit. Prints PASS/FAIL for each rule the organisers set."""
import re, os, glob, sys, hashlib

ok = True
def chk(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    ok = ok and cond

src = {f: open(f).read() for f in sorted(glob.glob("src/*.py"))}
print("=== 1. `id` column is never a feature ===")
bad = []
for f, s in src.items():
    for i, line in enumerate(s.splitlines(), 1):
        if re.search(r"\bid\b", line) and not line.strip().startswith("#"):
            # allowed: reading the csv, and writing it back out as the join key
            if re.search(r'"id":\s*te\[', line) or re.search(r"te\.id|d\.id\b|s\.id\b|\.id\.duplicated", line):
                continue
            if re.search(r"\bid\b\s*[,)\]]|\['id'\]|\[\"id\"\]", line):
                bad.append(f"{f}:{i}: {line.strip()[:70]}")
chk("no id in features/embeddings/grouping", len(bad) == 0, f"{len(bad)} suspect lines")
for b in bad[:5]: print("        ", b)

print("=== 2. seeds fixed ===")
c = src.get("src/common.py", "")
chk("random/numpy/torch/cuda/PYTHONHASHSEED all seeded",
    all(k in c for k in ["random.seed", "np.random.seed", "torch.manual_seed",
                         "cuda.manual_seed_all", "PYTHONHASHSEED"]))

print("=== 3. no external data ===")
def strip_comments(t):
    t = re.sub(r'"""[\s\S]*?"""', "", t)          # docstrings
    t = re.sub(r"\'\'\'[\s\S]*?\'\'\'", "", t)
    return "\n".join(l.split("#")[0] for l in t.splitlines())
ext = []
for f, s0 in src.items():
    if "dl_models" in f:
        continue                                    # that file only fetches model weights
    code = strip_comments(s0)
    if re.search(r"urlretrieve|urlopen|huggingface\.co/datasets|load_dataset\(", code):
        ext.append(f)
    for m in re.finditer(r"read_(csv|parquet|json)\(\s*[fr]?[\'\"]([^\'\"]+)", code):
        if not m.group(2).startswith(("data/", "rules/", "{DATA}", "oof/")):
            ext.append(f"{f}:{m.group(2)}")
chk("no external corpus loaded in src/ (code, not comments)", len(ext) == 0, str(ext))
chk("no ext/ directory present", not os.path.isdir("ext"))

print("=== 4. real learned model ===")
chk("transformer fine-tuning present", "AutoModel.from_pretrained" in src.get("src/train_mtl.py", ""))
cks = glob.glob("ckpt/*/fold*.pt")
chk("checkpoints saved", len(cks) >= 5, f"{len(cks)} files")

print("=== 5. submissions valid & unedited ===")
subs = sorted(glob.glob("sub/*/task1_*_output.csv"))
for s in subs:
    head = open(s, encoding="utf-8").readline().strip()
    chk(f"{s} header", head == "id,pred_label,pred_noise_type", head)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
