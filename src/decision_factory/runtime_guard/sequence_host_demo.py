"""Run one real Qwen -> sequence Bev -> in-memory host transition.

The host helpers are the no-oracle schema/effect functions retained from the audited
closed-loop runner. Effects exist only in a synthetic in-memory file/receipt state.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ADAPTER=ROOT/'adapters'/'h17'
CHECKPOINT=ROOT/'checkpoints'/'sequence-bev-v7-experimental.pt'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,default=ROOT/'examples'/'sequence-host-demo.jsonl')
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--qwen-base',type=Path,required=True)
 p.add_argument('--qwen-device',choices=('cuda','cpu'),default='cuda')
 p.add_argument('--bev-device',choices=('cuda','cpu'),default='cuda')
 p.add_argument('--bev-checkpoint',type=Path,default=CHECKPOINT)
 a=p.parse_args(argv)
 source=a.input.expanduser().resolve(strict=True);out=a.out.expanduser().resolve();ckpt=a.bev_checkpoint.expanduser().resolve(strict=True);base=a.qwen_base.expanduser().resolve(strict=True)
 if out.exists():raise FileExistsError(f'output exists: {out}')
 if base.name!='15852e8c16360a2fea060d615a32b45270f8a8fc':raise ValueError('Qwen snapshot revision mismatch')
 cases=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
 if len(cases)!=1:raise ValueError('host demo accepts exactly one case')
 case=cases[0]
 allowed={'schema','case_id','task','host_policy','initial_observations','prior_actions','virtual_files','action_menu'}
 if set(case)!=allowed or case.get('schema')!='sequence_host_demo_case/1':raise ValueError('unsupported host demo schema/fields')
 if not isinstance(case['virtual_files'],dict) or not isinstance(case['action_menu'],list) or not case['action_menu']:raise ValueError('invalid synthetic host state/menu')
 sys.path.insert(0,str(ROOT/'src'))
 from decision_factory.runtime_guard.qwen import QwenProposer
 from decision_factory.runtime_guard.sequence import evaluate_sequence_proposal
 from decision_factory.runtime_guard.sequence_host import copy_json,build_request,qwen_observations,qwen_visible_files,hard_validate,host_execute
 qwen=QwenProposer(ADAPTER,device=a.qwen_device,base_model_path=base)
 menu=[{'action_id':x['action_id'],'description':x['description'],'operation':x['operation'],'arguments':copy_json(x['arguments'])} for x in case['action_menu']]
 qinput={'user_request':copy_json(case['task']),'host_policy':copy_json(case['host_policy']),'action_menu':menu,'observations':qwen_observations(case['initial_observations'],case['prior_actions']),'initial_files':qwen_visible_files(case['prior_actions'])}
 qerr=None
 try: qresult=qwen.propose(qinput)
 except Exception as exc:qresult={'type':'ERROR'};qerr=f'{type(exc).__name__}: {exc}'
 proposal=copy_json(qresult['proposal']) if qresult.get('type')=='PROPOSE' else None
 request=None;assessment=None;decision=qresult.get('type') if qresult.get('type') in ('STOP','REVIEW') else 'REVIEW';entry={'host_disposition':decision};effect=False
 if proposal is not None:
  request=build_request(case,case['prior_actions'],case['initial_observations'],proposal)
  assessment=evaluate_sequence_proposal(request,head_checkpoint=ckpt,device=a.bev_device)
  decision=assessment.get('disposition') if assessment.get('status')=='assessed' else 'REVIEW'
  # The host independently enforces schema/root/budget limits, then applies a permitted synthetic effect.
  entry,effect=host_execute(case,proposal,decision,len([x for x in case['prior_actions'] if x.get('host_disposition')=='ALLOW']))
 record={'schema':'sequence_host_demo_record/1','case_id':case['case_id'],'qwen':qresult,'qwen_error':qerr,'proposal':proposal,'guard_request':request,'guard_assessment':assessment,'guard_decision':decision,'host_entry':entry,'effect_applied':effect,'external_effects':'none; in-memory synthetic effect only'}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
 receipt={'schema':'sequence_host_demo_receipt/1','input_sha256':sha(source),'output_sha256':sha(out),'qwen_adapter_sha256':sha(ADAPTER/'adapter_model.safetensors'),'qwen_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc','bev_revision':'cf5138b5b465343d09c3b4fbd71eaca36d989d34','bev_checkpoint_sha256':sha(ckpt),'host_helpers_sha256':sha(ROOT/'src/decision_factory/runtime_guard/sequence_host.py'),'source_episode_runner_sha256':'7218482a79c37ff8c30581bda27b942f7178b6f586ffc06cd0b7a96ca9618a90','device':{'qwen':a.qwen_device,'bev':'fp32-'+a.bev_device},'oracle_veto':'none','external_effects':'none'}
 out.with_suffix(out.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'case_id':record['case_id'],'qwen':qresult.get('type'),'proposal':proposal,'guard':decision,'host_disposition':entry.get('host_disposition'),'effect_applied':effect,'receipt':receipt},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
