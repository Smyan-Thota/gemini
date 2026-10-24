"""Historical experiment orchestration snapshot. Set GEMINI_WORKSPACE to the operator-owned run directory and QWEN_MODEL_SNAPSHOT to the pinned cached base when needed. GEMINI_REPO_ROOT defaults to this checkout. Weights, generated cases and scoring keys are not distributed. These scripts preserve run-specific layouts; use the public training/inference modules for new experiments. Publication is disabled unless GEMINI_ENABLE_PUBLICATION=1; no GPU jobs were run to publish these sources."""
import os, sys
import argparse, json, hashlib, subprocess, sys, time, traceback
from pathlib import Path
import torch
from decision_factory.training import frozen_readout as train
from decision_factory.runtime_guard import frozen_readout as runtime
ROOT = Path(os.environ.get('GEMINI_REPO_ROOT', str(Path(__file__).resolve().parents[2])))
DATA = Path(os.path.join(os.environ['GEMINI_WORKSPACE'], 'multi-tool-v2'))
OUT = Path(os.path.join(os.environ['GEMINI_WORKSPACE'], 'frozen-readout-v2'))

def cache(path, name, kind='states', profile='multi_tool_scene_v2'):
    train.cache(argparse.Namespace(input=str(path), output=str(OUT / name), kind=kind, profile=profile, device='cuda', limit=None))

def score(checkpoint, split):
    p = torch.load(checkpoint, map_location='cpu', weights_only=True)
    x, y, ids, labels, base = train.read_cache(OUT / split, DATA / (split + '.labels.jsonl'))
    obj = runtime.FrozenBevReadout.__new__(runtime.FrozenBevReadout)
    obj.payload = p
    obj.columns = p['columns']
    obj.mean = p['mean']
    obj.scale = p['scale']
    obj.layers = p['layers']
    obj.temperature = p['temperature']
    obj.permit_threshold = p['permit_threshold']
    obj.block_threshold = p['block_threshold']
    answers = [obj.predict_features(row) for row in x]
    pred = torch.tensor([runtime.CHOICES.index(a[0]) for a in answers])
    m = train.metrics(y, pred)
    m.update(train.pair_metrics(ids, labels, pred))
    domains = {}
    for domain in sorted({row.get('domain', 'unknown') for row in labels.values()}):
        idx = [i for i, s in enumerate(ids) if labels[s].get('domain', 'unknown') == domain]
        domains[domain] = train.metrics(y[idx], pred[idx])
    return {'candidate': m, 'original_bev': train.metrics(y, base.argmax(-1)), 'by_domain': domains, 'predictions': [{'scenario_id': s, 'expected': runtime.CHOICES[int(y[i])], 'predicted': answers[i][0], 'scores': answers[i][1]} for i, s in enumerate(ids)]}
try:
    cache(Path(os.path.join(os.environ['GEMINI_WORKSPACE'], 'native-training-run2/inputs/validation-requests.jsonl')), 'legacy-dev152', 'requests', 'operation_facts_v3')
    for split in ['train', 'dev', 'calibration', 'challenge_dev', 'tool_ood_dev']:
        cache(DATA / (split + '.states.jsonl'), split)
    train.fit(argparse.Namespace(train_cache=str(OUT / 'train'), train_labels=str(DATA / 'train.labels.jsonl'), dev_cache=str(OUT / 'dev'), dev_labels=str(DATA / 'dev.labels.jsonl'), output=str(OUT / 'fit'), epochs=200))
    subprocess.run([sys.executable, str(ROOT / 'tools/calibrate_frozen_readout.py'), '--checkpoint', str(OUT / 'fit/selected.pt'), '--calibration-cache', str(OUT / 'calibration'), '--calibration-labels', str(DATA / 'calibration.labels.jsonl'), '--training-labels', str(DATA / 'train.labels.jsonl'), '--development-labels', str(DATA / 'dev.labels.jsonl'), '--output', str(OUT / 'calibrated')], check=True)
    report = {'schema': 'bev_frozen_readout_development_comparison/1', 'scope': 'synthetic DEV, not fresh final'}
    for name, checkpoint in [('raw', OUT / 'fit/selected.pt'), ('calibrated', OUT / 'calibrated/calibrated.pt')]:
        report[name] = {}
        for split in ['dev', 'calibration', 'challenge_dev', 'tool_ood_dev']:
            report[name][split] = score(checkpoint, split)
    train.write_json(OUT / 'comparison.json', report)
    cp = OUT / 'calibrated/calibrated.pt'
    obj = runtime.load_readout(cp, train.sha(cp), device='cuda')
    rows = train.load_jsonl(DATA / 'dev.states.jsonl')
    parity = []
    for row in rows[:3]:
        live = obj.assess_scene(row['model_state'])
        expected = next((p for p in report['calibrated']['dev']['predictions'] if p['scenario_id'] == row['scenario_id']))
        parity.append({'scenario_id': row['scenario_id'], 'live': live, 'cached': expected, 'same': live['disposition'] == expected['predicted'] and max((abs(live['scores'][k] - expected['scores'][k]) for k in runtime.CHOICES)) < 1e-05})
    train.write_json(OUT / 'serialized-parity.json', parity)
    if not all((p['same'] for p in parity)):
        raise RuntimeError('serialized production parity failed')
    train.write_json(OUT / 'pipeline-complete.json', {'status': 'completed', 'checkpoint': str(cp), 'sha256': train.sha(cp), 'parity': True})
    print('PIPELINE COMPLETE', flush=True)
except Exception:
    (OUT / 'pipeline-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
