"""Extract the private-test archive. Handles AES-encrypted zips (PK 5.1),
which the system `unzip` cannot read even with the correct password."""
import sys, os, argparse, zipfile
import pyzipper

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--password", default=None)
ap.add_argument("--out", default="data")
a = ap.parse_args()

pw = a.password.encode() if a.password else None
os.makedirs(a.out, exist_ok=True)
for opener in (pyzipper.AESZipFile, zipfile.ZipFile):
    try:
        with opener(a.zip) as z:
            if pw:
                z.setpassword(pw)
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise RuntimeError("no CSV inside the archive")
            for n in names:
                dst = os.path.join(a.out, os.path.basename(n))
                with z.open(n) as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
                print(f"[extract] {dst}  ({os.path.getsize(dst)} bytes)  via {opener.__name__}")
            sys.exit(0)
    except RuntimeError as e:
        if "password" in str(e).lower():
            print(f"!! wrong or missing password ({opener.__name__}): {e}")
            sys.exit(2)
    except Exception as e:
        last = e
print(f"!! could not extract: {last}")
sys.exit(1)
