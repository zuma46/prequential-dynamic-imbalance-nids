#!/usr/bin/env python3
"""Regenerate pairwise inference on the complete chronological block grid.
No learner fitting or raw network dataset access. Run from any directory.
Use run_analysis.py with explicit input and output directories.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, math
import numpy as np
import pandas as pd

MODELS = ['HT','OB','MOOB','MUOB']
METRICS = ['macro_f1','balanced_accuracy','gmean','dynamic_minority_attack_recall','worst_attack_recall','benign_fpr']

def holm(p):
    x=np.asarray(p,float)
    if not np.isfinite(x).all(): raise ValueError('Holm requires every prespecified test p-value')
    ix=np.argsort(x,kind='stable'); out=np.empty(len(x))
    out[ix]=np.minimum(1,np.maximum.accumulate(x[ix]*(len(x)-np.arange(len(x)))))
    return out

def stable_seed(base,*parts):
    return base+int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)%1_000_000

def block_indices(n,L,reps,seed):
    if not 1<=L<=n: raise ValueError('Invalid span')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-L+1,size=(reps,math.ceil(n/L)))
    return (starts[:,:,None]+np.arange(L)).reshape(reps,-1)[:,:n]

def inference(x,indices):
    """Undefined positions remain NaN. Bootstrap full-grid index blocks first.
    Mean is conditional on eligible positions in each resampled grid.
    The centered-null distribution uses the same index draws and mask.
    """
    x=np.asarray(x,float); valid=np.isfinite(x); point=x[valid].mean()
    sampled=x[indices]; counts=np.isfinite(sampled).sum(axis=1); keep=counts>0
    boot=np.nansum(sampled[keep],axis=1)/counts[keep]
    null=boot-point
    return dict(mean_advantage_a_over_b=float(point),
        bootstrap_ci_low_advantage=float(np.quantile(boot,.025)),
        bootstrap_ci_high_advantage=float(np.quantile(boot,.975)),
        bootstrap_p_centered=float((1+(np.abs(null)>=abs(point)).sum())/(len(null)+1)),
        fraction_blocks_a_better=float(np.mean(x[valid]>0)),fraction_blocks_tied=float(np.mean(x[valid]==0)),
        n_nonoverlap_blocks=int(valid.sum()),n_grid_blocks=len(x),
        bootstrap_reps=len(indices),bootstrap_reps_valid=int(keep.sum()),
        bootstrap_reps_no_eligible=int((~keep).sum()),
        bootstrap_eligible_count_min=int(counts[keep].min()),bootstrap_eligible_count_max=int(counts[keep].max()))

def primary(blocks,out,reps,base):
    assert not blocks.duplicated(['canonical_dataset','algorithm','seed','block_index']).any()
    rows=[]; indices={}; universes={}
    for ds,gd in blocks.groupby('canonical_dataset',sort=True):
        grid=np.sort(gd.block_index.unique()); assert np.array_equal(grid,np.arange(1,len(grid)+1))
        assert set(gd.algorithm)==set(MODELS)
        universes[ds]=grid
        avg=gd.groupby(['algorithm','block_index'])[METRICS].mean()
        for model in MODELS:
            assert np.array_equal(np.sort(gd[gd.algorithm==model].block_index.unique()),grid)
        for L in [10,20,40]:
            seed=stable_seed(base,ds,L); ix=block_indices(len(grid),L,reps,seed);indices[(ds,L)]=ix
            for metric in METRICS:
                for a,b in itertools.combinations(MODELS,2):
                    xa=avg.loc[a,metric].reindex(grid).to_numpy();xb=avg.loc[b,metric].reindex(grid).to_numpy()
                    raw=xa-xb; direction=-1 if metric=='benign_fpr' else 1
                    v=inference(raw*direction,ix)
                    rows.append(dict(canonical_dataset=ds,metric=metric,metric_direction='lower_better'if direction<0 else'higher_better',model_a=a,model_b=b,block_size=50000,bootstrap_block_length_blocks=L,bootstrap_block_length_instances=L*50000,bootstrap_seed=seed,mean_raw_difference_a_minus_b=float(np.nanmean(raw)),median_raw_difference_a_minus_b=float(np.nanmedian(raw)),ci_excludes_zero=bool(v['bootstrap_ci_low_advantage']>0 or v['bootstrap_ci_high_advantage']<0),**v))
    p=pd.DataFrame(rows)
    for (_,L),idx in p.groupby(['canonical_dataset','bootstrap_block_length_blocks']).groups.items():
        assert len(idx)==36;p.loc[idx,'holm_adjusted_p']=holm(p.loc[idx,'bootstrap_p_centered'])
    for L,idx in p.groupby('bootstrap_block_length_blocks').groups.items():
        assert len(idx)==72;p.loc[idx,'holm_adjusted_p_global72']=holm(p.loc[idx,'bootstrap_p_centered'])
    p['significant_holm_alpha']=p.holm_adjusted_p<.05;p['significant_global72_alpha_0_05']=p.holm_adjusted_p_global72<.05
    p['status_change_vs_within_dataset_holm']=np.where(p.significant_holm_alpha & ~p.significant_global72_alpha_0_05,'loses_significance','unchanged')
    p.to_csv(out/'Table_S_Pairwise_MBB_All_Scales.csv',index=False)
    p20=p[p.bootstrap_block_length_blocks==20].copy()
    p20.to_csv(out/'Table_S_Pairwise_MBB_Full_Numeric.csv',index=False)
    p20.to_csv(out/'Table_S_Global72_Holm_Sensitivity.csv',index=False)
    effect=[]
    for r in p20.itertuples(index=False):
        if 'MOOB' not in [r.model_a,r.model_b]:continue
        pos=r.model_a=='MOOB';sgn=1 if pos else -1
        effect.append(dict(canonical_dataset=r.canonical_dataset,comparator=r.model_b if pos else r.model_a,metric=r.metric,
            moob_minus_comparator_advantage=sgn*r.mean_advantage_a_over_b,
            bootstrap_ci_low_moob_advantage=r.bootstrap_ci_low_advantage if pos else -r.bootstrap_ci_high_advantage,
            bootstrap_ci_high_moob_advantage=r.bootstrap_ci_high_advantage if pos else -r.bootstrap_ci_low_advantage,
            holm_adjusted_p_within_dataset=r.holm_adjusted_p,
            fraction_blocks_moob_better=r.fraction_blocks_a_better if pos else 1-r.fraction_blocks_a_better-r.fraction_blocks_tied,
            fraction_blocks_tied=r.fraction_blocks_tied,n_nonoverlap_blocks=r.n_nonoverlap_blocks,n_grid_blocks=r.n_grid_blocks,
            bootstrap_block_length_blocks=20,bootstrap_block_length_instances=1000000,bootstrap_reps_valid=r.bootstrap_reps_valid))
    pd.DataFrame(effect).to_csv(out/'Table_S_MOOB_Primary_Effects_With_WinFraction.csv',index=False)
    cs=[]
    for key,g in p.groupby(['canonical_dataset','model_a','model_b','metric']):
        signs=np.sign(g.mean_advantage_a_over_b); ns=int(g.significant_holm_alpha.sum())
        cs.append(dict(zip(['canonical_dataset','model_a','model_b','metric'],key))|dict(n_significant_scales=ns,same_direction=bool(signs.nunique()==1),consistent=bool(ns==3 and signs.nunique()==1),status='consistent'if ns==3 and signs.nunique()==1 else'never significant'if ns==0 else'scale-sensitive'))
    pd.DataFrame(cs).to_csv(out/'Table_S_Cross_Scale_Status.csv',index=False)
    return p,indices,universes

def sensitivity(classes,universes,indices,out):
    keys=['canonical_dataset','algorithm','block_index','class']
    avg=classes.groupby(keys,as_index=False).agg(support=('support','first'),recall=('recall','mean'),block_n=('block_n','first'))
    rows=[];values=[]
    for ds,gd in avg.groupby('canonical_dataset'):
        grid=universes[ds];K=gd['class'].nunique()
        active=gd[(gd.algorithm=='HT')&(gd.support>0)].groupby('block_index')['class'].nunique().reindex(grid)
        for definition in ['0.5/K','1/K','2/K','1/|A_t|']:
            threshold={'0.5/K':.5/K,'1/K':1/K,'2/K':2/K}.get(definition)
            cut=gd.block_index.map(1/active) if threshold is None else threshold
            q=gd[(gd.support>0)&~gd['class'].str.lower().eq('benign')&(gd.support/gd.block_n<cut)]
            means=q.groupby(['algorithm','block_index']).recall.mean()
            series={}
            for m in MODELS:
                x=means.loc[m].reindex(grid).to_numpy() if m in means.index.get_level_values(0)else np.full(len(grid),np.nan)
                series[m]=x
                values.extend(dict(canonical_dataset=ds,algorithm=m,block_index=int(b),dmar=float(v),definition=definition)for b,v in zip(grid,x))
            for comp in ['HT','OB']:
                v=inference(series['MOOB']-series[comp],indices[(ds,20)])
                rows.append(dict(canonical_dataset=ds,definition=definition,contrast='MOOB-'+comp,n_matched_blocks=v['n_nonoverlap_blocks'],n_grid_blocks=len(grid),mean_paired_dmar_difference=v['mean_advantage_a_over_b'],mbb_ci_low=v['bootstrap_ci_low_advantage'],mbb_ci_high=v['bootstrap_ci_high_advantage'],bootstrap_p_centered=v['bootstrap_p_centered'],fraction_blocks_moob_better=v['fraction_blocks_a_better'],fraction_blocks_tied=v['fraction_blocks_tied'],bootstrap_block_length_blocks=20,bootstrap_reps=v['bootstrap_reps'],bootstrap_reps_valid=v['bootstrap_reps_valid'],bootstrap_reps_no_eligible=v['bootstrap_reps_no_eligible']))
    s=pd.DataFrame(rows);assert len(s)==16;s['holm_adjusted_p_sensitivity_family']=holm(s.bootstrap_p_centered)
    s['ci_excludes_zero']=(s.mbb_ci_low>0)|(s.mbb_ci_high<0);s['positive_direction']=s.mean_paired_dmar_difference>0
    points=s[s.definition=='1/K'].set_index(['canonical_dataset','contrast']).mean_paired_dmar_difference
    s['primary_effect']=[points.loc[(r.canonical_dataset,r.contrast)]for r in s.itertuples()]
    s['same_direction_as_primary']=np.sign(s.mean_paired_dmar_difference)==np.sign(s.primary_effect)
    s.to_csv(out/'Table_S_DMAR_Threshold_Sensitivity.csv',index=False)
    pd.DataFrame(values).to_csv(out/'Table_S_DMAR_Threshold_BlockValues.csv',index=False)
    return s
