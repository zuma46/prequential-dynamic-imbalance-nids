from pathlib import Path
import argparse,sys
import pandas as pd
from inference import primary,sensitivity
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reps',type=int,default=5000);p.add_argument('--seed',type=int,default=20260805);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
b=pd.read_csv(a.input/'nonoverlap_block_metrics_derived.csv');b=b[b.is_full_50k_block].copy()
c=pd.read_parquet(a.input/'nonoverlap_block_class_metrics_derived.parquet');c=c[c.is_full_50k_block].copy()
p,ix,u=primary(b,a.out,a.reps,a.seed);sensitivity(c,u,ix,a.out)
