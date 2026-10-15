from pathlib import Path
import importlib.util,sys,numpy as np,json,argparse,hashlib
p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('checked_models',a.models);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for K in [3,10,15]:
 names=['Benign']+[f'attack{i}' for i in range(K-1)];t=m.DynamicClassSizeTracker(names,theta=.9,floor=1e-12)
 for i in range(1,1001):
  rates=t.update_and_get_rates('Benign')
  if i in [1,10,100,300,1000]:rows.append(dict(K=K,benign_updates=i,lambda_muob=rates.muob,lambda_moob=rates.moob,min_size=float(t.sizes.min()),sum_size=float(t.sizes.sum())))
 assert rates.muob<1e-10
 first=t.update_and_get_rates('attack0');assert first.muob<1e-9 # other known attack remains inactive
# Direct recurrence comparison to the actual tracker, including numerical floor/normalization.
t=m.DynamicClassSizeTracker(['a','b','c']);w=np.full(3,1/3)
for y in ['a']*350+['b']*20+['c']*10:
 w=.9*w;w[['a','b','c'].index(y)]+=.1;w=np.maximum(w,1e-12);w/=w.sum();r=t.update_and_get_rates(y)
 assert np.allclose(t.sizes,w,rtol=1e-13,atol=1e-15)
 assert np.isclose(r.muob,w.min()/w[['a','b','c'].index(y)])
# Verify Poisson repeat calls using a recorder, not a learned classifier.
class Recorder:
 def __init__(self):self.n=0
 def learn_one(self,x,y):self.n+=1
 def predict_one(self,x):return None
model=m.OnlineBaggingClassifier('MUOB',Recorder,5,123,['Benign','attack'])
assert model.predict_one({},'Benign')=='Benign'
expected=np.random.default_rng(123).poisson(.7,size=5)
model.learn_one({},'attack',.7)
assert [x.n for x in model.learners]==expected.tolist()
a.out.write_text(json.dumps({'source_sha256':hashlib.sha256(a.models.read_bytes()).hexdigest(),'checks':'PASS','scope':'actual tracker and repeat-update implementation; synthetic mechanism test, not detector performance or public release validation','trajectory':rows},indent=2))
print('PASS',rows[-1])
