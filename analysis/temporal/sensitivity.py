from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
d=pd.read_parquet(a.input/'nonoverlap_block_class_metrics_derived.parquet');d=d[d.is_full_50k_block].copy()
k=['canonical_dataset','algorithm','seed','block_index','class'];assert not d.duplicated(k).any()
# Reconstruct sufficient per-class counts; reject rounded recalls that do not recover counts.
tp=d.support*d.recall.fillna(0);assert np.max(np.abs(tp-tp.round()))<1e-7
d['tp']=tp.round().astype('int64');assert (d.tp<=d.support).all() and (d.tp<=d.predicted).all()
# Verify sample counts and prevalence do not depend on learner/seed.
assert d.groupby(['canonical_dataset','block_index','class']).support.nunique().max()==1
q=d[(d['class'].str.lower()!='benign')&d.is_dynamic_minority&(d.support>=50)].copy()
q['failure']=q.recall.eq(0);q['very_low']=q.recall.gt(0)&q.recall.lt(.2);q['low']=q.recall.ge(.2)&q.recall.lt(.5);q['higher']=q.recall.ge(.5)
cols=['failure','very_low','low','higher']
seeds=q.groupby(['canonical_dataset','algorithm','seed']).agg(eligible=('recall','size'),**{c:(c,'sum') for c in cols}).reset_index()
for c in cols:seeds[c+'_pct']=100*seeds[c]/seeds.eligible
seeds.to_csv(a.out/'failure_per_seed.csv',index=False)
summary=seeds.groupby(['canonical_dataset','algorithm']).agg(n_seeds=('seed','size'),failure_mean_pct=('failure_pct','mean'),failure_min_pct=('failure_pct','min'),failure_max_pct=('failure_pct','max')).reset_index()
mean=q.groupby(['canonical_dataset','algorithm','block_index','class']).recall.mean().reset_index();pooled=mean.assign(failure=mean.recall.eq(0)).groupby(['canonical_dataset','algorithm']).agg(eligible=('recall','size'),averaged_failure=('failure','sum')).reset_index();summary=summary.merge(pooled);summary['averaged_failure_pct']=100*summary.averaged_failure/summary.eligible;summary.to_csv(a.out/'failure_estimands_comparison.csv',index=False)
# Within-class evidence counts at the primary sample threshold.
r=d[(d['class'].str.lower()!='benign')&(d.support>=50)].groupby(['canonical_dataset','algorithm','block_index','class'],as_index=False).agg(support=('support','first'),recall=('recall','mean'),minority=('is_dynamic_minority','first'))
rows=[]
for (ds,model),g in r.groupby(['canonical_dataset','algorithm']):
 varying=g.groupby('class').support.nunique();switch=g.groupby('class').minority.nunique();sc=switch[switch==2].index
 rows.append(dict(dataset=ds,model=model,class_blocks=len(g),classes=g['class'].nunique(),support_varying_classes=int((varying>1).sum()),status_varying_classes=len(sc),status_class_blocks=int(g['class'].isin(sc).sum())))
pd.DataFrame(rows).to_csv(a.out/'relationship_evidence_counts.csv',index=False)
# Reaggregate counts, not metric averages, at each coarser temporal scale.
metrics=[];states=[]
for width in [1,2,4]:
 for ds,g0 in d.groupby('canonical_dataset'):
  nblocks=g0.block_index.max();g=g0[g0.block_index<=nblocks//width*width].copy();g['unit']=(g.block_index-1)//width+1
  v=g.groupby(['canonical_dataset','algorithm','seed','unit','class'],as_index=False)[['support','predicted','tp']].sum();v['recall']=v.tp/v.support.replace(0,np.nan)
  total=v.groupby(['algorithm','seed','unit']).support.transform('sum');K=v['class'].nunique();v['minority']=(v.support>0)&(v.support/total<1/K)&v['class'].str.lower().ne('benign')
  for (model,seed,unit),z in v.groupby(['algorithm','seed','unit']):
   active=z[z.support>0];att=active[active['class'].str.lower()!='benign'];ben=z[z['class'].str.lower()=='benign'].iloc[0]
   metrics.append(dict(dataset=ds,width_flows=width*50000,model=model,seed=seed,unit=unit,DMAR=z.loc[z.minority,'recall'].mean(),WAR=att.recall.min(),Macro_F1=(2*active.tp/(active.support+active.predicted)).mean(),BA=active.recall.mean(),GMean=np.prod(active.recall)**(1/len(active)),FPR=(ben.support-ben.tp)/ben.support if ben.support else np.nan))
  z=v[v.minority&(v.support>=50)].groupby(['algorithm','unit','class']).recall.mean().reset_index()
  for model,g in z.groupby('algorithm'):states.append(dict(dataset=ds,width_flows=width*50000,model=model,eligible=len(g),failure=int(g.recall.eq(0).sum()),very_low=int((g.recall.gt(0)&g.recall.lt(.2)).sum())))
m=pd.DataFrame(metrics);pd.DataFrame(states).to_csv(a.out/'coarser_failure_counts.csv',index=False)
av=m.groupby(['dataset','width_flows','model','unit'])[['DMAR','WAR','Macro_F1','BA','GMean','FPR']].mean().reset_index()
av.to_csv(a.out/'reaggregated_seed_mean_metrics.csv',index=False)
original=pd.read_csv(a.input/'nonoverlap_block_metrics_derived.csv');original=original[original.is_full_50k_block]
mapping={'DMAR':'dynamic_minority_attack_recall','WAR':'worst_attack_recall','Macro_F1':'macro_f1','BA':'balanced_accuracy','GMean':'gmean','FPR':'benign_fpr'}
check=m[m.width_flows==50000].merge(original,left_on=['dataset','model','seed','unit'],right_on=['canonical_dataset','algorithm','seed','block_index'],validate='one_to_one')
assert len(check)==len(original)
errs={}
for short,long in mapping.items():
 assert np.array_equal(check[short].isna(),check[long].isna())
 errs[short]=float((check[short]-check[long]).abs().max())
 assert errs[short]<1e-10,(short,errs[short])
(a.out/'reaggregation_validation.json').write_text(json.dumps(errs,indent=2))
results=[];influence=[]
for (ds,w),g in av.groupby(['dataset','width_flows']):
 for base in ['HT','OB']:
  z=g[g.model=='MOOB'].merge(g[g.model==base],on='unit',suffixes=('_moob','_base'))
  for metric in ['DMAR','WAR','Macro_F1','BA','GMean','FPR']:
   delta=z[metric+'_moob']-z[metric+'_base'];valid=delta.notna();episode=(valid.ne(valid.shift(fill_value=False))).cumsum()
   results.append(dict(dataset=ds,width_flows=w,comparator=base,metric=metric,n_grid=len(z),n_valid=int(valid.sum()),mean_moob_minus_base=delta.mean()))
   if w==50000:
    for e,h in z[valid].groupby(episode[valid]):
     keep=valid&~z.index.isin(h.index);influence.append(dict(dataset=ds,comparator=base,metric=metric,omitted_run=int(e),first_block=int(h.unit.min()),last_block=int(h.unit.max()),remaining_blocks=int(keep.sum()),mean_after_omission=delta[keep].mean()))
pd.DataFrame(results).to_csv(a.out/'coarser_paired_effects_descriptive.csv',index=False)
pd.DataFrame(influence).to_csv(a.out/'eligible_run_influence_descriptive.csv',index=False)
# Leave one attack class out of block DMAR, retaining original minority definition (K fixed).
a0=d[(d['class'].str.lower()!='benign')&d.is_dynamic_minority].groupby(['canonical_dataset','algorithm','block_index','class']).recall.mean().reset_index();out=[]
for ds,g in a0.groupby('canonical_dataset'):
 for cls in g['class'].unique():
  v=g[g['class']!=cls].groupby(['algorithm','block_index']).recall.mean().unstack('algorithm')
  for base in ['HT','OB']:
   x=v.MOOB-v[base];out.append(dict(dataset=ds,omitted_class=cls,comparator=base,n_valid=int(x.notna().sum()),mean_DMAR_moob_minus_base=x.mean()))
pd.DataFrame(out).to_csv(a.out/'leave_class_out_DMAR_descriptive.csv',index=False)
(a.out/'validation.json').write_text(json.dumps({'source_rows_full_blocks':len(d),'tp_integer_reconstruction_max_error':float(np.max(np.abs(tp-tp.round()))),'training_repeated':False,'coarser_analysis':'descriptive sensitivity; no new significance claims','eligible_runs':'contiguous metric-valid runs, not annotated attack episodes','input_sha256':hashlib.sha256((a.input/'nonoverlap_block_class_metrics_derived.parquet').read_bytes()).hexdigest()},indent=2))
print(summary.to_string(index=False));print('DONE')
