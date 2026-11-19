"""Download official KuaiRand-Pure, verify its published checksum, extract safely.
Read the dataset's terms and cite its authors when using it in research.
Raw data are not included in the delivered project archive.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path,PurePosixPath

URL='https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1'
MD5='0820331067a3784d9691136f772b35a7'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/raw');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    archive=root/'KuaiRand-Pure.tar.gz';dest=root/'kuairand_extracted'
    if dest.exists():
        raise ValueError('Extraction destination exists. Preserve it or choose a new --root.')
    if not archive.exists():
        partial=archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(URL,timeout=30) as response,open(partial,'wb') as f:
                count=0;last=0
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    f.write(block);count+=len(block)
                    if count-last>=8*1024*1024:
                        print(f'Downloaded {count//(1024*1024)} MiB',flush=True);last=count
            partial.rename(archive)
        except BaseException:
            partial.unlink(missing_ok=True);raise
    md5=hashlib.md5();sha=hashlib.sha256()
    with open(archive,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            md5.update(block);sha.update(block)
    if md5.hexdigest()!=MD5:
        raise ValueError('Archive checksum differs from published MD5. Inspect/download again.')
    stage=Path(tempfile.mkdtemp(prefix='extract-',dir=root))
    try:
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar.getmembers():
                relative=PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts or not (member.isdir() or member.isfile()):
                    raise ValueError('Unsafe archive member: '+member.name)
                target=stage.joinpath(*relative.parts)
                if member.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source,open(target,'wb') as output:
                        shutil.copyfileobj(source,output)
        stage.rename(dest)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    provenance={'url':URL,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'extracted_to':str(dest),
                'reference':'https://github.com/chongminggao/KuaiRand'}
    (root/'kuairand_source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
