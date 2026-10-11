#!/usr/bin/env python3
from pathlib import Path
import argparse,json,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result-root',type=Path,required=True); a=ap.parse_args(); r=a.result_root
    checks=[]
    def ck(name,cond,detail=''):
        checks.append((name,bool(cond),detail))
    m=r/'07_manifest/RQ1_FORMAL_RELATIONSHIP_MANIFEST.json'; ck('manifest_exists',m.is_file(),str(m))
    if m.is_file():
        o=json.loads(m.read_text()); ck('status_PASS',o.get('status')=='PASS'); ck('no_learner_execution',o.get('learner_execution') is False); ck('no_raw_dataset_read',o.get('raw_dataset_read') is False); ck('no_retuning',o.get('retuning') is False)
        ck('implementation_v1_1',o.get('implementation_version')=='v1_1_FULL_BLOCK_TOPOLOGY_FIX',str(o.get('implementation_version'))); nb=o.get('dataset_full_blocks',{}); ck('CSE_402',any('CSE' in k and v==402 for k,v in nb.items()),str(nb)); ck('ToN_550',any('ToN' in k and v==550 for k,v in nb.items()),str(nb))
    p=r/'02_relationship/Table_RQ1_Formal_Relationship_Primary.csv'; ck('primary_table',p.is_file())
    if p.is_file():
        d=pd.read_csv(p); ck('primary_rows_16',len(d)==16,f'rows={len(d)}') # 2 datasets x 4 models x 2 estimands
        inf=d[d.algorithm.isin(['HT','OB','MOOB'])]; ck('inferential_rows_12',len(inf)==12,f'rows={len(inf)}'); testable=inf[inf.p_centered.notna()]; ck('testable_relationships_exist',len(testable)>0,f'rows={len(testable)}'); ck('holm_present_for_testable',testable.holm_p_primary_relationship_family.notna().all())
    c=r/'03_treatment_contrasts/Table_RQ1_Treatment_Coefficient_Contrasts_Primary.csv'; ck('contrast_table',c.is_file())
    if c.is_file():
        x=pd.read_csv(c); ck('contrast_rows_8',len(x)==8,f'rows={len(x)}'); xt=x[x.p_centered.notna()]; ck('testable_contrasts_exist',len(xt)>0,f'rows={len(xt)}'); ck('contrast_holm_present',xt.holm_p_primary_treatment_contrast_family.notna().all())
    ok=all(x[1] for x in checks)
    for n,p,d in checks: print(('PASS' if p else 'FAIL'),n,d)
    print('FINAL', 'PASS' if ok else 'FAIL', f'{sum(x[1] for x in checks)}/{len(checks)}')
    raise SystemExit(0 if ok else 2)
if __name__=='__main__': main()
