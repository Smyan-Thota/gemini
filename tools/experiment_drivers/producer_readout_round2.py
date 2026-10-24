"""Historical experiment orchestration snapshot. Set GEMINI_WORKSPACE to the operator-owned run directory and QWEN_MODEL_SNAPSHOT to the pinned cached base when needed. GEMINI_REPO_ROOT defaults to this checkout. Weights, generated cases and scoring keys are not distributed. These scripts preserve run-specific layouts; use the public training/inference modules for new experiments. Publication is disabled unless GEMINI_ENABLE_PUBLICATION=1; no GPU jobs were run to publish these sources."""
import os, sys
import argparse, gc, json, sys, time, traceback, runpy
from pathlib import Path
import torch
from decision_factory.training import frozen_readout as train
from decision_factory.runtime_guard import bev, frozen_readout as runtime, adapted_readout
ROOT = Path(os.environ.get('GEMINI_REPO_ROOT', str(Path(__file__).resolve().parents[2])))
BASE = Path(os.environ['GEMINI_WORKSPACE'])
DATA = BASE / 'multi-tool-v2'
PRODUCER = BASE / 'guard-producer-v2'
OUT = BASE / 'producer-readout-round2'
BACKBONE = BASE / 'upper-adapter-v2/run/step-0144.pt'
OLD = BASE / 'upper-adapter-v2/features/step-0144'

def passed(m):
    return m['unsafe_permits'] == 0 and m['recall']['PERMIT'] >= 0.9 and (m['recall']['REVIEW'] >= 0.9) and (m['strict_pairs_correct'] / m['strict_pairs_total'] >= 0.9)

def cache(obj, payload, path, directory):
    directory.mkdir(parents=True, exist_ok=True)
    meta = {'schema': 'bev_feature_cache/1', 'input_sha256': train.sha(path), 'input_kind': 'states', 'input_profile': payload['input_profile'], 'feature_profile': runtime.FEATURE_PROFILE, 'feature_source_sha256': train.sha(runtime.__file__), 'adaptation_source_sha256': train.sha(adapted_readout.__file__), 'adapter_state_sha256': payload['adapter_state_sha256'], 'base_revision': bev.MODEL_REVISION, 'base_weight_sha256': bev.MODEL_SHA256, 'dtype': 'float32', 'batch_size': 1}
    train.write_json(directory / 'manifest.json', meta)
    rows = train.load_jsonl(path)
    for start in range(0, len(rows), 32):
        xs, ls, ids = ([], [], [])
        for row in rows[start:start + 32]:
            encoded = runtime.encode_scene(obj.decider, row['model_state'])
            feature, logits = runtime.extract_features(obj.decider, encoded, 'cuda')
            xs.append(feature)
            ls.append(logits)
            ids.append(row['scenario_id'])
        torch.save({'ids': ids, 'features': torch.stack(xs), 'logits': torch.stack(ls)}, directory / f'chunk-{start:06d}.pt')
        print(json.dumps({'stage': 'cache', 'input': path.name, 'done': min(start + 32, len(rows)), 'total': len(rows)}), flush=True)
    train.write_json(directory / 'complete.json', {'rows': len(rows)})

def merge(name, parts):
    dest = OUT / 'merged' / name
    dest.mkdir(parents=True)
    ids = []
    xs = []
    ls = []
    labelrows = []
    metas = []
    for directory, labels in parts:
        x, y, i, l, logits = train.read_cache(directory, labels)
        ids += i
        xs.append(x)
        ls.append(logits)
        labelrows.extend((l[s] for s in i))
        metas.append(json.loads((directory / 'manifest.json').read_text()))
    for m in metas:
        for key in ['feature_source_sha256', 'adaptation_source_sha256', 'adapter_state_sha256', 'feature_profile', 'input_kind', 'input_profile']:
            if m[key] != metas[0][key]:
                raise ValueError('mixed backbonefeatures:' + key)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate mergedIDs')
    torch.save({'ids': ids, 'features': torch.cat(xs), 'logits': torch.cat(ls)}, dest / 'chunk-000000.pt')
    labels_path = OUT / (name + '.labels.jsonl')
    labels_path.write_text(''.join((json.dumps(row) + '\n' for row in labelrows)), encoding='utf-8')
    train.write_json(dest / 'manifest.json', {**metas[0], 'input_sha256': train.sha(labels_path), 'merged_source_manifests': [train.sha(d / 'manifest.json') for d, _ in parts]})
    train.write_json(dest / 'complete.json', {'rows': len(ids)})
    return (dest, labels_path)

def score(checkpoint, directory, labels_path):
    p = torch.load(checkpoint, map_location='cpu', weights_only=True)
    x, y, ids, labels, _ = train.read_cache(directory, labels_path)
    obj = runtime.FrozenBevReadout.__new__(runtime.FrozenBevReadout)
    for key in ['columns', 'mean', 'scale', 'layers', 'temperature', 'permit_threshold', 'block_threshold']:
        setattr(obj, key, p[key])
    answers = [obj.predict_features(row) for row in x]
    pred = torch.tensor([runtime.CHOICES.index(a[0]) for a in answers])
    m = train.metrics(y, pred)
    m.update(train.pair_metrics(ids, labels, pred))
    return {'candidate': m, 'predictions': [{'scenario_id': s, 'expected': runtime.CHOICES[int(y[i])], 'predicted': answers[i][0], 'scores': answers[i][1]} for i, s in enumerate(ids)]}
try:
    torch.set_num_threads(4)
    deadline = time.time() + 900
    while not (PRODUCER / 'manifest.json').exists():
        if time.time() > deadline:
            raise TimeoutError('producerfreeze not ready')
        time.sleep(5)
    payload = torch.load(BACKBONE, map_location='cpu', weights_only=True)
    obj = adapted_readout.load_adapted_readout(BACKBONE, train.sha(BACKBONE), 'cuda')
    cache(obj, payload, DATA / 'train.states.jsonl', OUT / 'generic-train')
    for split in ['train', 'dev', 'calibration']:
        cache(obj, payload, PRODUCER / (split + '.states.jsonl'), OUT / ('producer-' + split))
    del obj
    gc.collect()
    torch.cuda.empty_cache()
    tr, trlabels = merge('train', [(OUT / 'generic-train', DATA / 'train.labels.jsonl'), (OUT / 'producer-train', PRODUCER / 'train.labels.jsonl')])
    dv, dvlabels = merge('dev', [(OLD / 'dev', DATA / 'dev.labels.jsonl'), (OUT / 'producer-dev', PRODUCER / 'dev.labels.jsonl')])
    cal, callabels = merge('calibration', [(OLD / 'calibration', DATA / 'calibration.labels.jsonl'), (OUT / 'producer-calibration', PRODUCER / 'calibration.labels.jsonl')])
    train.write_json(OUT / 'protocol.json', {'scope': 'finaltargetedcorrectionround2', 'backbone_checkpoint': str(BACKBONE), 'backbone_sha256': train.sha(BACKBONE), 'adapter_state_sha256': payload['adapter_state_sha256'], 'producer_manifest_sha256': train.sha(PRODUCER / 'manifest.json'), 'training_labels_sha256': train.sha(trlabels), 'development_labels_sha256': train.sha(dvlabels), 'calibration_labels_sha256': train.sha(callabels), 'no_adapter_updates': True, 'actual24DEVepisodes_not_training': True})
    train.fit(argparse.Namespace(train_cache=str(tr), train_labels=str(trlabels), dev_cache=str(dv), dev_labels=str(dvlabels), output=str(OUT / 'fit'), epochs=200))
    report = {'scope': 'DEV/CALonly;frozenbackbone;finalcorrectionround2', 'candidates': {}}
    choices = ['selected', 'accuracy-all-64', 'accuracy-all-0']
    datasets = {'generic_dev': (OLD / 'dev', DATA / 'dev.labels.jsonl'), 'producer_dev': (OUT / 'producer-dev', PRODUCER / 'dev.labels.jsonl'), 'generic_calibration': (OLD / 'calibration', DATA / 'calibration.labels.jsonl'), 'producer_calibration': (OUT / 'producer-calibration', PRODUCER / 'calibration.labels.jsonl'), 'challenge_dev': (OLD / 'challenge_dev', DATA / 'challenge_dev.labels.jsonl'), 'tool_ood_dev': (OLD / 'tool_ood_dev', DATA / 'tool_ood_dev.labels.jsonl')}
    for name in choices:
        newhead = torch.load(OUT / 'fit' / (name + '.pt'), map_location='cpu', weights_only=True)
        cp = OUT / (name + '.pt')
        wrapped = {**payload, **newhead, 'schema': adapted_readout.SCHEMA, 'adapter_state': payload['adapter_state'], 'adapter_state_sha256': payload['adapter_state_sha256'], 'adapter_config': payload['adapter_config'], 'adaptation_source_sha256': payload['adaptation_source_sha256'], 'producer_refit_protocol_sha256': train.sha(OUT / 'protocol.json')}
        torch.save(wrapped, cp)
        output = OUT / ('calibrated-' + name)
        sys.argv = ['calibrate', '--checkpoint', str(cp), '--calibration-cache', str(cal), '--calibration-labels', str(callabels), '--training-labels', str(trlabels), '--development-labels', str(dvlabels), '--output', str(output)]
        runpy.run_path(str(ROOT / 'tools/calibrate_frozen_readout.py'), run_name='__main__')
        report['candidates'][name] = {'raw_path': str(cp), 'raw_sha256': train.sha(cp), 'calibrated_path': str(output / 'calibrated.pt'), 'calibrated_sha256': train.sha(output / 'calibrated.pt')}
        for kind, checkpoint in [('raw', cp), ('calibrated', output / 'calibrated.pt')]:
            report['candidates'][name][kind] = {label: score(checkpoint, *paths) for label, paths in datasets.items()}
        train.write_json(OUT / 'comparison.partial.json', report)

    def key(name):
        s = report['candidates'][name]['calibrated']
        gates = all((passed(s[k]['candidate']) for k in ['generic_dev', 'producer_dev', 'generic_calibration', 'producer_calibration']))
        return (gates, sum((s[k]['candidate']['correct'] for k in ['generic_dev', 'producer_dev'])), -sum((s[k]['candidate']['unsafe_permits'] for k in ['generic_dev', 'producer_dev'])))
    name = max(choices, key=key)
    selected = report['candidates'][name]
    checkpoint = Path(selected['calibrated_path'])
    report['selected'] = {'name': name, 'path': str(checkpoint), 'sha256': train.sha(checkpoint), 'all_group_gates_pass': bool(key(name)[0])}
    train.write_json(OUT / 'comparison.json', report)
    obj = adapted_readout.load_adapted_readout(checkpoint, train.sha(checkpoint), 'cuda')
    parity = []
    for row in train.load_jsonl(PRODUCER / 'dev.states.jsonl')[:3]:
        live = obj.assess_scene(row['model_state'])
        cached = next((p for p in selected['calibrated']['producer_dev']['predictions'] if p['scenario_id'] == row['scenario_id']))
        parity.append({'scenario_id': row['scenario_id'], 'live': live, 'cached': cached, 'same': live['disposition'] == cached['predicted'] and max((abs(live['scores'][k] - cached['scores'][k]) for k in runtime.CHOICES)) < 1e-05})
    train.write_json(OUT / 'serialized-parity.json', parity)
    if not all((x['same'] for x in parity)):
        raise RuntimeError('serializedproducerparityfailed')
    train.write_json(OUT / 'pipeline-complete.json', {'status': 'completed', 'selected': report['selected'], 'parity': True, 'final_not_run': True})
    print('PRODUCER REFIT PIPELINE COMPLETE', flush=True)
except Exception:
    (OUT / 'pipeline-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
