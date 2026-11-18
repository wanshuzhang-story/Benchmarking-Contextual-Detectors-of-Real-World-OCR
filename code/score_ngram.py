"""Corpus-specific stupid-backoff in-domain trigram score, retained basic algorithm.

    python code/score_ngram.py data/benchmark.csv corpora/hmd.txt scores/rebuilt --corpus UK --output-name ngram-uk.csv
    python code/score_ngram.py data/external.csv corpora/uspd.txt scores/rebuilt --corpus US --output-name ngram-us.csv
"""
import argparse,math,re
from collections import Counter
from pathlib import Path
import pandas as pd
from common import load
TOKEN=re.compile(r'\w+');ALPHA=.4;MAX_VOCAB=2_000_000

def windows(sentence,spans):
 for s,e in spans:
  l=TOKEN.findall(sentence[:s].lower())[-2:];r=TOKEN.findall(sentence[e:].lower())[:2]
  yield ['<s>']*(2-len(l))+l+[sentence[s:e].lower()]+r+['</s>']*(2-len(r))

def logp(a,b,c,C,V,N):
 if C[(a,b,c)] and C[(a,b)]:return math.log(C[(a,b,c)]/C[(a,b)])
 if C[(b,c)] and C[(b,)]:return math.log(ALPHA)+math.log(C[(b,c)]/C[(b,)])
 return 2*math.log(ALPHA)+math.log((C[(c,)]+1)/(N+V))

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('data');ap.add_argument('corpus_text');ap.add_argument('output_dir')
 ap.add_argument('--corpus',choices=['UK','US'],help='For external.csv, score only a single corpus subset')
 ap.add_argument('--output-name',default='ngram-hmd.ngram.csv')
 a=ap.parse_args()
 df=load(a.data)
 if a.corpus and 'corpus' in df.columns:df=df[df.corpus==a.corpus]
 hits=[(id,list(windows(s,sp))) for id,s,sp in zip(df.id,df.sentence,df.spans)]
 needed=set()
 for _,sets in hits:
  for w in sets:
   for i in range(2,5):
    t=tuple(w[i-2:i+1]);needed.update((t,t[:2],t[1:],t[1:2],t[2:]))
 C=Counter();N=0;vocab=set()
 with open(a.corpus_text,encoding='utf-8') as fh:
  for line in fh:
   tokens=['<s>','<s>']+TOKEN.findall(line)+['</s>','</s>']
   N+=len(tokens)-4
   for i in range(len(tokens)):
    for n in [1,2,3]:
     gram=tuple(tokens[i:i+n])
     if len(gram)==n and gram in needed:C[gram]+=1
    if len(vocab)<MAX_VOCAB:vocab.add(tokens[i])
 V=len(vocab)
 rows=[]
 for id,sets in hits:
  v=[-sum(logp(*w[i-2:i+1],C,V,N) for i in range(2,5)) for w in sets]
  rows.append({'id':id,'susp':max(v) if v else None})
 path=Path(a.output_dir);path.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows,columns=['id','susp']).to_csv(path/a.output_name,index=False)
 print('Scored',len(rows),'for',a.corpus or 'all','using NEW corpus',a.corpus_text)

if __name__=='__main__':main()
