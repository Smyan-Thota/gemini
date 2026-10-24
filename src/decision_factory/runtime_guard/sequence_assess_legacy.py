"""Legacy one-turn assessment-only CLI. Use ``sequence-guard`` for host execution."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
PACKAGE_ROOT=Path(__file__).resolve().parents[3]
DEFAULT_ADAPTER=PACKAGE_ROOT/'adapters'/'h17'
DEFAULT_CHECKPOINT=PACKAGE_ROOT/'checkpoints'/'sequence-bev-v7-experimental.pt'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def read_rows(path):
 for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
  if not line.strip(): continue
  row=json.loads(line)
  if not isinstance(row,dict): raise ValueError(f'row {n} must be an object')
  yield n,row
def qwen_history(observations,prior_actions):
 result=json.loads(json.dumps(observations,ensure_ascii=False))
 for i,action in enumerate(prior_actions,1):
  result.append({'source':'host-action-history','trust':'trusted','text':'Actual prior action '+str(i)+': '+json.dumps({k:action[k] for k in ('tool_name','arguments','host_disposition') if k in action},ensure_ascii=False,sort_keys=True,separators=(',',':'))})
  obs=action.get('observation')
  if isinstance(obs,dict) and obs.get('trust') in ('trusted','untrusted'): result.append(json.loads(json.dumps(obs,ensure_ascii=False)))
 return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,default=PACKAGE_ROOT/'examples'/'sequence-demo.jsonl')
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--qwen-base',type=Path,required=True,help='existing local snapshot at the adapter-pinned revision')
 p.add_argument('--qwen-device',choices=('cuda','cpu'),default='cuda')
 p.add_argument('--bev-device',choices=('cuda',),default='cuda')
 p.add_argument('--bev-checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
 p.add_argument('--limit',type=int)
 a=p.parse_args(argv)
 source=a.input.expanduser().resolve(strict=True); out=a.out.expanduser().resolve()
 checkpoint=a.bev_checkpoint.expanduser().resolve(strict=True); base=a.qwen_base.expanduser().resolve(strict=True)
 if out.exists(): raise FileExistsError(f'output already exists: {out}')
 if a.limit is not None and a.limit<1: raise ValueError('--limit must be positive')
 if base.name!='15852e8c16360a2fea060d615a32b45270f8a8fc': raise ValueError('Qwen snapshot directory must match the pinned revision')
 allowed={'schema','case_id','user_request','host_policy','observations','prior_actions','initial_files','action_menu'}
 cases=[]
 for n,row in read_rows(source):
  if set(row)!=allowed or row.get('schema')!='sequence_demo_case/1': raise ValueError(f'row {n} has unsupported schema or fields')
  if (not isinstance(row.get('case_id'),str) or not isinstance(row.get('user_request'),dict) or not isinstance(row.get('host_policy'),dict) or not isinstance(row.get('observations'),list) or not isinstance(row.get('prior_actions'),list) or not isinstance(row.get('initial_files'),dict) or not isinstance(row.get('action_menu'),list) or not row['action_menu']): raise ValueError(f'row {n} has invalid task/policy/history/menu')
  cases.append(row)
  if a.limit is not None and len(cases)>=a.limit: break
 if not cases: raise ValueError('input contains no cases')
 sys.path.insert(0,str(PACKAGE_ROOT/'src'))
 from decision_factory.runtime_guard.qwen import QwenProposer
 from decision_factory.runtime_guard.host import build_host_assessment_request
 from decision_factory.runtime_guard.sequence import evaluate_routed_proposal
 from decision_factory.runtime_guard import bev
 proposer=QwenProposer(DEFAULT_ADAPTER,device=a.qwen_device,base_model_path=base)
 records=[]
 for c in cases:
  task={'user_request':c['user_request'],'host_policy':c['host_policy'],'action_menu':c['action_menu'],'observations':qwen_history(c['observations'],c['prior_actions']),'initial_files':c['initial_files']}
  start=time.perf_counter(); error=None
  try: qwen=proposer.propose(task)
  except Exception as exc: qwen={'type':'ERROR'}; error=f'{type(exc).__name__}: {exc}'
  proposal=qwen.get('proposal') if qwen.get('type')=='PROPOSE' else None
  request=None; assessment=None
  if isinstance(proposal,dict):
   seed={'schema':'bev_guard_request/1','user_request':c['user_request'],'host_policy':c['host_policy'],'observations':c['observations'],'prior_actions':c['prior_actions'],'proposal':proposal}
   request=build_host_assessment_request(seed,c['initial_files'],proposal,prior_actions=c['prior_actions'])
   assessment=evaluate_routed_proposal(request,sequence_checkpoint=checkpoint,legacy_checkpoint=PACKAGE_ROOT/'checkpoints'/'legacy-v4-bev.pt',sequence_device=a.bev_device)
   final=assessment.get('disposition','REVIEW')
  else: final=qwen.get('type') if qwen.get('type') in ('STOP','REVIEW') else 'REVIEW'
  records.append({'schema':'sequence_demo_record/1','case_id':c['case_id'],'qwen':qwen,'qwen_error':error,'proposal':proposal,'guard_request':request,'guard_assessment':assessment,'final_disposition':final,'elapsed_ms':round((time.perf_counter()-start)*1000,2)})
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records),encoding='utf-8')
 from decision_factory.runtime_guard import sequence
 receipt={'schema':'sequence_demo_receipt/1','input':str(source),'input_sha256':sha(source),'output':str(out),'output_sha256':sha(out),'case_count':len(cases),'qwen_model_id':'Qwen/Qwen3.5-2B','qwen_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc','qwen_adapter_sha256':sha(DEFAULT_ADAPTER/'adapter_model.safetensors'),'bev_model_id':bev.MODEL_ID,'bev_revision':bev.MODEL_REVISION,'bev_base_weight_sha256':bev.MODEL_SHA256,'bev_checkpoint_sha256':sha(checkpoint),'bev_device':a.bev_device,'runtime':'sequence guard FP32','python':sys.version,'platform':platform.platform(),'external_effects':'none; proposal assessment only'}
 receipt_path=out.with_suffix(out.suffix+'.receipt.json'); receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
