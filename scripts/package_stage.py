"""Package code and instructions only; never package environments or raw records."""
import argparse
import zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='migration/video_rec_stage.zip');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/a.out
    if out.exists():raise ValueError('Package exists; choose a new --out.')
    out.parent.mkdir(parents=True,exist_ok=True)
    files=[root/'requirements-stage-lock.txt',root/'README.md']
    for folder in ['rec_lab','scripts','configs','docs','tests']:
        files += [f for f in (root/folder).rglob('*') if f.is_file() and f.suffix in ['.py','.md','.yaml','.json'] and '__pycache__' not in f.parts]
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f.relative_to(root)))
    print(out)
if __name__=='__main__':main()
