"""Merge two corpus-specific external trigram scores without altering historic files."""
import argparse
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser()
 p.add_argument('uk_scores');p.add_argument('us_scores');p.add_argument('output_file')
 a=p.parse_args()
 src=Path(__file__).resolve().parent.parent/'data/external.csv'
 ids=pd.read_csv(src,dtype={'id':str}).id
 frames=[pd.read_csv(fn,dtype={'id':str}) for fn in (a.uk_scores,a.us_scores)]
 combined=pd.concat(frames,ignore_index=True)
 assert len(combined)==len(ids) and combined.id.is_unique and set(combined.id)==set(ids)
 combined=combined.set_index('id').reindex(ids).reset_index()
 Path(a.output_file).parent.mkdir(parents=True,exist_ok=True)
 combined.to_csv(a.output_file,index=False)
 print('Merged UK+US',len(combined),'rows into',a.output_file)
if __name__=='__main__':main()
