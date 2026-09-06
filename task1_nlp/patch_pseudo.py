"""Add pseudo-labelling: append confidently-predicted TEST rows to the training
set. Uses only data supplied by the organisers -- no external corpus."""
p = 'src/train_mtl.py'; s = open(p).read()
assert '--pseudo' not in s, "already patched"

s = s.replace('p.add_argument("--aug", type=float, default=0.0)',
              'p.add_argument("--pseudo", default=None)        # npz of test probabilities\n'
              'p.add_argument("--pseudo_th", type=float, default=0.9)\n'
              'p.add_argument("--aug", type=float, default=0.0)')

s = s.replace('''    ks = list(range(5)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]''',
'''    if a.pseudo:
        d = np.load(a.pseudo)
        ph, pn = d["te_h"], d["te_n"]
        keep = (ph.max(1) >= a.pseudo_th) & (pn.max(1) >= a.pseudo_th)
        ex = pd.DataFrame({
            "text": te.text.values[keep], "view": np.asarray(te_view, dtype=object)[keep],
            "y_hate": ph.argmax(1)[keep], "y_noise": pn.argmax(1)[keep],
            "label": "", "noise_type": "", "group": ["PSEUDO%d" % i for i in range(keep.sum())],
            "fold": -1,
        })
        print(f"[pseudo] {keep.sum()}/{len(te)} test rows above {a.pseudo_th}", flush=True)
    else:
        ex = None

    ks = list(range(5)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]''')

# fold -1 never lands in validation, so pseudo rows only ever join the training side
s = s.replace('    trn, val = df[df.fold != k], df[df.fold == k]',
              '''    trn, val = df[df.fold != k], df[df.fold == k]
    if PSEUDO is not None:
        trn = pd.concat([trn, PSEUDO], ignore_index=False)''')
s = s.replace('def run_fold(df, k, tok, te_texts):', 'PSEUDO = None\n\n\ndef run_fold(df, k, tok, te_texts):')
s = s.replace('        (s, (vh, vn), (th, tn)), idx = run_fold(df, k, tok, te_view)',
              '        globals()["PSEUDO"] = ex\n        (s, (vh, vn), (th, tn)), idx = run_fold(df, k, tok, te_view)')
open(p, 'w').write(s); print("pseudo-labelling patched in")
