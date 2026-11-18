"""New run ONLY: construct a pinned UK or US corpus with an 8-gram exclusion filter.

    python code/build_corpus.py biglam/hmd_newspapers corpora/hmd.txt --max-tokens 30000000 --revision <revision>
    python code/build_corpus.py PleIAs/US-PD-Newspapers corpora/uspd.txt --max-tokens 30000000 --revision <revision>

Old scores' historic corpus versions and 30M/50M cap remain UNVERIFIED.
"""
import argparse,hashlib,json,re
from pathlib import Path
import pandas as pd
from datasets import load_dataset
ROOT=Path(__file__).resolve().parent.parent
K=8

def norm(text):return re.sub(r'\W+',' ',str(text).casefold()).strip()
def shingles(words):
 if len(words)<K:return set()
 return {tuple(words[i:i+K]) for i in range(len(words)-K+1)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('dataset');ap.add_argument('output')
 ap.add_argument('--max-tokens',type=int,default=30_000_000)
 ap.add_argument('--revision',default=None)
 ap.add_argument('--seed',type=int,default=20260825)
 args=ap.parse_args()
 excluded=set()
 for f in ['benchmark.csv','external.csv']:
  for sent in pd.read_csv(ROOT/'data'/f).sentence:excluded.update(shingles(norm(sent).split()))
 ds=load_dataset(args.dataset,revision=args.revision,split='train',streaming=True)
 ds=ds.shuffle(seed=args.seed,buffer_size=1000)
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
 count=0;lines=0;dig=hashlib.sha256()
 with out.open('w',encoding='utf-8') as fh:
  for item in ds:
   for raw in str(item.get('text','')).splitlines():
    words=norm(raw).split()
    if not words or (shingles(words)&excluded):continue
    remaining=args.max_tokens-count
    if remaining<=0:break
    words=words[:remaining]
    line=' '.join(words)+'\n'
    fh.write(line);dig.update(line.encode('utf-8'));count+=len(words);lines+=1
   if count>=args.max_tokens:break
 manifest={'dataset':args.dataset,'revision':args.revision,'seed':args.seed,
           'max_tokens_requested':args.max_tokens,'tokens_written':count,'lines_written':lines,
           'sha256':dig.hexdigest(),'exclusion':'8-gram overlapping original benchmark/external OCR sentences',
           'experiment_role':'NEW_REBUILD_NOT_HISTORIC_INPUT'}
 (Path(str(out)+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 print(manifest)

if __name__=='__main__':main()
