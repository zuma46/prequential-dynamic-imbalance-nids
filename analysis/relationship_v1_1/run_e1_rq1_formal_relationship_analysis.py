#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

REQ={"canonical_dataset","algorithm","seed","block_index","class","support","recall","prior","is_dynamic_minority"}
MODELS=["HT","OB","MOOB","MUOB"]
INFER_MODELS=["HT","OB","MOOB"]
ESTIMANDS=["support_slope","minority_contrast"]
IMPLEMENTATION_VERSION="v1_1_FULL_BLOCK_TOPOLOGY_FIX"


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_csv(df,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False,float_format='%.12g')


def holm_adjust(pvals):
    a=np.asarray(pvals,float); out=np.full(len(a),np.nan); valid=np.where(np.isfinite(a))[0]
    if not len(valid): return out
    order=valid[np.argsort(a[valid])]; m=len(order); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,min(1.0,(m-rank)*a[idx])); out[idx]=running
    return out


def as_bool_series(s):
    if s.dtype==bool:return s
    return s.astype(str).str.strip().str.lower().isin({'1','true','t','yes','y'})


def is_benign(s):
    return s.astype(str).str.strip().str.lower().isin({'benign','normal','normal.'})


def read_table(p):
    p=Path(p)
    if p.suffix.lower()=='.parquet': return pd.read_parquet(p)
    if p.suffix.lower()=='.csv': return pd.read_csv(p)
    raise ValueError('Input must be .parquet or .csv')


def prepare_seed_averaged(raw):
    missing=REQ-set(raw.columns)
    if missing: raise RuntimeError(f'Missing required columns: {sorted(missing)}')
    d=raw.copy()
    if 'is_full_50k_block' in d.columns: d=d[as_bool_series(d['is_full_50k_block'])].copy()
    d['class']=d['class'].astype(str); d['block_index']=pd.to_numeric(d['block_index'],errors='raise').astype(int); d['support']=pd.to_numeric(d['support'],errors='coerce')
    d['recall']=pd.to_numeric(d['recall'],errors='coerce'); d['prior']=pd.to_numeric(d['prior'],errors='coerce')
    d['is_dynamic_minority']=as_bool_series(d['is_dynamic_minority']); d=d[d['algorithm'].isin(MODELS)].copy()
    keys=['canonical_dataset','algorithm','block_index','class']
    for col in ['support','is_dynamic_minority']:
        bad=d.groupby(keys,dropna=False)[col].nunique(dropna=False); bad=bad[bad>1]
        if len(bad): raise RuntimeError(f'{col} is not invariant across seeds for {len(bad)} groups')
    agg={'support':('support','max'),'prior':('prior','mean'),'recall':('recall','mean'),
         'is_dynamic_minority':('is_dynamic_minority','first'),'n_seed_records':('seed','nunique')}
    if 'instance_start' in d.columns: agg['instance_start']=('instance_start','min')
    if 'instance_end' in d.columns: agg['instance_end']=('instance_end','max')
    g=d.groupby(keys,as_index=False).agg(**agg); g['is_benign']=is_benign(g['class'])
    g=g[(g.support>0)&(~g.is_benign)&g.recall.notna()].copy(); g['log10_support']=np.log10(g.support.astype(float))
    return g.sort_values(['canonical_dataset','algorithm','block_index','class']).reset_index(drop=True)



def derive_full_block_universe(raw):
    """Derive the chronological block universe BEFORE removing benign/attack-absent rows.

    This is essential for moving-block bootstrap: attack-absent full blocks are part of
    processed-flow time and must remain as empty temporal positions rather than being
    compressed out of the sequence.
    """
    d=raw.copy()
    if 'is_full_50k_block' in d.columns:
        d=d[as_bool_series(d['is_full_50k_block'])].copy()
    d=d[d['algorithm'].isin(MODELS)].copy()
    d['block_index']=pd.to_numeric(d['block_index'],errors='raise').astype(int)
    out={}
    for ds,g in d.groupby('canonical_dataset'):
        blocks=np.sort(g['block_index'].unique())
        expected=402 if 'CSE' in str(ds) else (550 if 'ToN' in str(ds) else None)
        if expected is not None and len(blocks)!=expected:
            raise RuntimeError(f'Expected {expected} full blocks for {ds}, got {len(blocks)} from unfiltered full-block topology')
        bset=set(map(int,blocks))
        for model in MODELS:
            gm=g[g.algorithm==model]
            if gm.empty:
                raise RuntimeError(f'Missing learner {model} in full-block topology for {ds}')
            mset=set(map(int,gm.block_index.unique()))
            if mset!=bset:
                miss=sorted(bset-mset)[:10]; extra=sorted(mset-bset)[:10]
                raise RuntimeError(f'Full-block topology mismatch for {ds}/{model}: missing={miss}, extra={extra}')
        out[ds]=blocks
    if not out:
        raise RuntimeError('No full-block topology could be derived from input')
    return out

def point_meta(g,estimand):
    if estimand=='support_slope':
        used=0
        for _,h in g.groupby('class'):
            if len(h)>=2 and np.std(h.log10_support.to_numpy(float))>1e-12: used+=1
        return {'n_class_blocks':len(g),'n_classes':used}
    switch=0
    for _,h in g.groupby('class'):
        if set(h.is_dynamic_minority.astype(bool))=={False,True}: switch+=1
    return {'n_class_blocks':len(g),'n_switch_classes':switch,
            'n_minority_blocks':int(g.is_dynamic_minority.sum()),
            'n_nonminority_blocks':int((~g.is_dynamic_minority).sum())}


def build_matrices(g,nblocks):
    """Block-by-class sufficient-statistic matrices for weighted class-FE regression."""
    classes=sorted(g['class'].unique()); cmap={c:i for i,c in enumerate(classes)}; C=len(classes)
    shape=(nblocks,C)
    mats={k:np.zeros(shape,float) for k in ['v0','vx','vy','vxx','vxy','vz','vzz','vzy']}
    if not len(g): return mats,classes
    bi=g['_block_pos'].to_numpy(int); cj=g['class'].map(cmap).to_numpy(int)
    x=g['log10_support'].to_numpy(float); y=g['recall'].to_numpy(float); z=g['is_dynamic_minority'].astype(float).to_numpy()
    np.add.at(mats['v0'],(bi,cj),1.0); np.add.at(mats['vx'],(bi,cj),x); np.add.at(mats['vy'],(bi,cj),y)
    np.add.at(mats['vxx'],(bi,cj),x*x); np.add.at(mats['vxy'],(bi,cj),x*y); np.add.at(mats['vz'],(bi,cj),z)
    np.add.at(mats['vzz'],(bi,cj),z*z); np.add.at(mats['vzy'],(bi,cj),z*y)
    return mats,classes


def fe_from_weight_matrix(W,mats,estimand):
    # W: B x nblocks. Returns B coefficients.
    S0=W@mats['v0']; Sy=W@mats['vy']
    good=S0>0
    if estimand=='support_slope':
        Sx=W@mats['vx']; Sxx=W@mats['vxx']; Sxy=W@mats['vxy']
        with np.errstate(divide='ignore',invalid='ignore'):
            num=np.where(good,Sxy-Sx*Sy/S0,0.).sum(axis=1)
            den=np.where(good,Sxx-Sx*Sx/S0,0.).sum(axis=1)
    else:
        Sz=W@mats['vz']; Szz=W@mats['vzz']; Szy=W@mats['vzy']
        with np.errstate(divide='ignore',invalid='ignore'):
            num=np.where(good,Szy-Sz*Sy/S0,0.).sum(axis=1)
            den=np.where(good,Szz-Sz*Sz/S0,0.).sum(axis=1)
    out=np.full(W.shape[0],np.nan); ok=den>1e-15; out[ok]=num[ok]/den[ok]
    return out


def moving_block_weights(nblocks,L,reps,seed):
    L=max(1,min(int(L),nblocks)); starts=np.arange(0,nblocks-L+1); k=int(math.ceil(nblocks/L)); rng=np.random.default_rng(seed)
    W=np.zeros((reps,nblocks),dtype=np.int16)
    for b in range(reps):
        ss=rng.choice(starts,size=k,replace=True); pos=np.concatenate([np.arange(s,s+L) for s in ss])[:nblocks]
        W[b]=np.bincount(pos,minlength=nblocks)
    return W


def centered_stats(theta,vals):
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    if not np.isfinite(theta) or not len(a): return (np.nan,np.nan,np.nan,0)
    lo,hi=np.quantile(a,[.025,.975]); centered=a-theta
    p=(1+np.sum(np.abs(centered)>=abs(theta)))/(len(a)+1)
    return float(lo),float(hi),float(p),len(a)


def stable_seed(base,*parts):
    off=int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)%1_000_000
    return int(base+off)


def analyze(seedavg,block_universe,floors,lengths,reps,base_seed):
    relrows=[]; cache={}; meta_cache={}
    for ds,gd0 in seedavg.groupby('canonical_dataset'):
        if ds not in block_universe: raise RuntimeError(f'No full-block universe for {ds}')
        blocks=np.asarray(block_universe[ds],dtype=int); bmap={int(b):i for i,b in enumerate(blocks)}; nblocks=len(blocks)
        observed=set(map(int,gd0.block_index.unique())); unknown=observed-set(bmap)
        if unknown: raise RuntimeError(f'Attack-class rows contain block indices outside full-block universe for {ds}: {sorted(unknown)[:10]}')
        gd0=gd0.copy(); gd0['_block_pos']=gd0.block_index.map(bmap).astype(int)
        W_by_L={L:moving_block_weights(nblocks,L,reps,stable_seed(base_seed,ds,L)) for L in lengths}
        for model in MODELS:
            gm0=gd0[gd0.algorithm==model].copy()
            for floor in floors:
                gm=gm0[gm0.support>=floor].copy()
                constant=(len(gm)>0 and np.nanstd(gm.recall.to_numpy(float))<=1e-15)
                mats,classes=build_matrices(gm,nblocks) if len(gm) else ({k:np.zeros((nblocks,0)) for k in ['v0','vx','vy','vxx','vxy','vz','vzz','vzy']},[])
                W0=np.ones((1,nblocks),float)
                for est in ESTIMANDS:
                    point=fe_from_weight_matrix(W0,mats,est)[0] if len(classes) else np.nan
                    meta=point_meta(gm,est); meta_cache[(ds,model,floor,est)]=meta
                    for L in lengths:
                        vals=fe_from_weight_matrix(W_by_L[L],mats,est) if len(classes) else np.array([])
                        cache[(ds,model,floor,L,est)]=vals
                        lo,hi,p,nvalid=centered_stats(point,vals)
                        status='OK'
                        if constant:
                            status='DEGENERATE_CONSTANT_RECALL'; p=np.nan
                        relrows.append({'canonical_dataset':ds,'algorithm':model,'estimand':est,'support_floor':floor,
                            'mbb_length_blocks':L,'bootstrap_reps_requested':reps,'bootstrap_reps_valid':nvalid,
                            'estimate':point,'ci_low':lo,'ci_high':hi,'p_centered':p,'status':status,**meta})
    rel=pd.DataFrame(relrows); rel['holm_p_primary_relationship_family']=np.nan
    mask=(rel.support_floor==50)&(rel.mbb_length_blocks==20)&rel.algorithm.isin(INFER_MODELS)&rel.p_centered.notna()
    for ds,idx in rel[mask].groupby('canonical_dataset').groups.items(): rel.loc[idx,'holm_p_primary_relationship_family']=holm_adjust(rel.loc[idx,'p_centered'])

    crows=[]
    for ds in rel.canonical_dataset.unique():
        for floor in floors:
            for L in lengths:
                for est in ESTIMANDS:
                    ht=rel[(rel.canonical_dataset==ds)&(rel.algorithm=='HT')&(rel.support_floor==floor)&(rel.mbb_length_blocks==L)&(rel.estimand==est)].iloc[0]
                    for model in ['OB','MOOB']:
                        rr=rel[(rel.canonical_dataset==ds)&(rel.algorithm==model)&(rel.support_floor==floor)&(rel.mbb_length_blocks==L)&(rel.estimand==est)].iloc[0]
                        theta=rr.estimate-ht.estimate if np.isfinite(rr.estimate) and np.isfinite(ht.estimate) else np.nan
                        vals=cache[(ds,model,floor,L,est)]-cache[(ds,'HT',floor,L,est)]
                        lo,hi,p,nvalid=centered_stats(theta,vals)
                        crows.append({'canonical_dataset':ds,'contrast':f'{model}-HT','estimand':est,'support_floor':floor,
                                      'mbb_length_blocks':L,'bootstrap_reps_requested':reps,'bootstrap_reps_valid':nvalid,
                                      'estimate_difference':theta,'ci_low':lo,'ci_high':hi,'p_centered':p})
    contr=pd.DataFrame(crows); contr['holm_p_primary_treatment_contrast_family']=np.nan
    mask=(contr.support_floor==50)&(contr.mbb_length_blocks==20)&contr.p_centered.notna()
    for ds,idx in contr[mask].groupby('canonical_dataset').groups.items(): contr.loc[idx,'holm_p_primary_treatment_contrast_family']=holm_adjust(contr.loc[idx,'p_centered'])
    return rel,contr


def descriptives(seedavg):
    bins=[0,10,50,100,1000,10000,np.inf]; labels=['1-9','10-49','50-99','100-999','1k-9,999','>=10k']
    d=seedavg.copy(); d['support_bin']=pd.cut(d.support,bins=bins,labels=labels,right=False)
    sb=d.groupby(['canonical_dataset','algorithm','support_bin'],observed=True).agg(n_class_blocks=('recall','size'),mean_recall=('recall','mean'),median_recall=('recall','median')).reset_index()
    q=d[d.support>=50]; ms=q.groupby(['canonical_dataset','algorithm','is_dynamic_minority']).agg(n_class_blocks=('recall','size'),mean_recall=('recall','mean'),median_recall=('recall','median')).reset_index()
    return sb,ms


def make_figures(seedavg,outdir):
    import matplotlib.pyplot as plt
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    bins=[1,10,50,100,1000,10000,np.inf]; labels=['1-9','10-49','50-99','100-999','1k-9,999','>=10k']
    d=seedavg.copy();d['support_bin']=pd.cut(d.support,bins=bins,labels=labels,right=False)
    for ds,g in d.groupby('canonical_dataset'):
        fig,ax=plt.subplots(figsize=(8.2,4.8))
        for m in MODELS:
            h=g[g.algorithm==m].groupby('support_bin',observed=True).recall.mean().reindex(labels)
            ax.plot(range(len(labels)),h.to_numpy(float),marker='o',label=m)
        ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=25,ha='right')
        ax.set_xlabel('Attack-class support within exact 50,000-flow block');ax.set_ylabel('Mean seed-averaged class recall');ax.set_ylim(-.03,1.03);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        stem='Figure_RQ1_Support_Recall_'+('CSE' if 'CSE' in ds else 'ToN');fig.savefig(outdir/(stem+'.png'),dpi=240,bbox_inches='tight');fig.savefig(outdir/(stem+'.pdf'),bbox_inches='tight');plt.close(fig)


def fmt(x,d=3): return 'NA' if not np.isfinite(x) else f'{x:.{d}f}'

def make_summary(rel,contr,seedavg,outdir,src):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);p=rel[(rel.support_floor==50)&(rel.mbb_length_blocks==20)];c=contr[(contr.support_floor==50)&(contr.mbb_length_blocks==20)]
    lines=['# E1 RQ1 Formal Relationship Analysis v1.0','',f'- Authoritative input: `{src}`',f'- SHA256: `{sha256(src)}`',f'- Seed-averaged active attack class-block rows: {len(seedavg):,}','- Learner execution: NO','- Raw dataset read: NO','- Retuning: NO','','## Primary relationship estimates (support floor 50, MBB L=20)','']
    for r in p.itertuples(index=False): lines.append(f'- {r.canonical_dataset} | {r.algorithm} | {r.estimand}: {fmt(r.estimate,4)} [{fmt(r.ci_low,4)}, {fmt(r.ci_high,4)}], Holm={fmt(r.holm_p_primary_relationship_family,5)}, status={r.status}')
    lines += ['','## Secondary treatment-coefficient contrasts vs HT','']
    for r in c.itertuples(index=False): lines.append(f'- {r.canonical_dataset} | {r.contrast} | {r.estimand}: {fmt(r.estimate_difference,4)} [{fmt(r.ci_low,4)}, {fmt(r.ci_high,4)}], Holm={fmt(r.holm_p_primary_treatment_contrast_family,5)}')
    lines += ['','## Interpretation boundary','','These are within-class temporal associations derived from frozen class-block evidence. They are not causal effects and do not establish concept drift.'];(outdir/'RQ1_FORMAL_RELATIONSHIP_SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description='Read-only E1 RQ1 formal relationship analysis from frozen exact class-block outputs.')
    ap.add_argument('--block-class-source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bootstrap-reps',type=int,default=5000);ap.add_argument('--bootstrap-seed',type=int,default=20260831);a=ap.parse_args()
    src=a.block_class_source.expanduser().resolve();out=a.out.expanduser().resolve()
    if not src.is_file():raise FileNotFoundError(src)
    out.mkdir(parents=True,exist_ok=True);raw=read_table(src)
    block_universe=derive_full_block_universe(raw)
    seedavg=prepare_seed_averaged(raw)
    if seedavg.empty:raise RuntimeError('No active non-benign class-block rows after preparation')
    nblocks={ds:int(len(blocks)) for ds,blocks in block_universe.items()}
    attack_observed_blocks=seedavg.groupby('canonical_dataset').block_index.nunique().to_dict()
    
    try:
        seedavg.to_parquet(out/'01_seed_averaged_attack_class_blocks.parquet',index=False)
    except Exception:
        seedavg.to_csv(out/'01_seed_averaged_attack_class_blocks.csv',index=False)

    rel,contr=analyze(seedavg,block_universe,[10,50,100],[10,20,40],a.bootstrap_reps,a.bootstrap_seed)
    write_csv(rel,out/'02_relationship/Table_RQ1_Formal_Relationship_AllSensitivity.csv');write_csv(rel[(rel.support_floor==50)&(rel.mbb_length_blocks==20)],out/'02_relationship/Table_RQ1_Formal_Relationship_Primary.csv')
    write_csv(contr,out/'03_treatment_contrasts/Table_RQ1_Treatment_Coefficient_Contrasts_AllSensitivity.csv');write_csv(contr[(contr.support_floor==50)&(contr.mbb_length_blocks==20)],out/'03_treatment_contrasts/Table_RQ1_Treatment_Coefficient_Contrasts_Primary.csv')
    sb,ms=descriptives(seedavg);write_csv(sb,out/'04_descriptive/Table_RQ1_Support_Bin_Recall.csv');write_csv(ms,out/'04_descriptive/Table_RQ1_Minority_State_Recall.csv');make_figures(seedavg,out/'05_figures');make_summary(rel,contr,seedavg,out/'07_manifest',src)
    
    manifest={'analysis_id':'E1_RQ1_FORMAL_RELATIONSHIP_v1_0','status':'PASS','operation':'DERIVED_READ_ONLY_ANALYSIS_OF_FROZEN_CLASS_BLOCK_OUTPUTS','learner_execution':False,'raw_dataset_read':False,'retuning':False,'model_state_change':False,'implementation_version':IMPLEMENTATION_VERSION,'input':{'path':str(src),'sha256':sha256(src),'rows':int(len(raw))},'prepared_rows':int(len(seedavg)),'dataset_full_blocks':{k:int(v) for k,v in nblocks.items()},'dataset_attack_observed_blocks':{k:int(v) for k,v in attack_observed_blocks.items()},'bootstrap_reps':a.bootstrap_reps,'bootstrap_seed':a.bootstrap_seed,'support_floors':[10,50,100],'mbb_lengths':[10,20,40],'primary_support_floor':50,'primary_mbb_length':20,'outputs':[]}
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='RQ1_FORMAL_RELATIONSHIP_MANIFEST.json':manifest['outputs'].append({'path':str(p.relative_to(out)),'sha256':sha256(p)})
    mp=out/'07_manifest/RQ1_FORMAL_RELATIONSHIP_MANIFEST.json';mp.parent.mkdir(parents=True,exist_ok=True);mp.write_text(json.dumps(manifest,indent=2))
    print('='*88);print('E1 RQ1 FORMAL RELATIONSHIP ANALYSIS v1.1');print('='*88);print('Input:',src);print('SHA256:',sha256(src));print('Full block topology:',nblocks);print('Attack-observed block indices after attack-row preparation:',attack_observed_blocks);print('Prepared attack class-block rows:',len(seedavg));print('Output:',out);print('STATUS: PASS');print('No learner execution. No raw dataset read. No retuning. No model-state change.')
if __name__=='__main__':main()
