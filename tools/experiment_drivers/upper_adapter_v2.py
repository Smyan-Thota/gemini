"""Historical experiment orchestration snapshot. Set GEMINI_WORKSPACE to the operator-owned run directory and QWEN_MODEL_SNAPSHOT to the pinned cached base when needed. GEMINI_REPO_ROOT defaults to this checkout. Weights, generated cases and scoring keys are not distributed. These scripts preserve run-specific layouts; use the public training/inference modules for new experiments. Publication is disabled unless GEMINI_ENABLE_PUBLICATION=1; no GPU jobs were run to publish these sources."""
import os, sys
import ctypes, json, sys, time, traceback, runpy, gc
from ctypes import wintypes
from pathlib import Path
import torch
from decision_factory.training import frozen_readout as train
from decision_factory.runtime_guard import bev, frozen_readout as runtime, adapted_readout
ROOT = Path(os.environ.get('GEMINI_REPO_ROOT', str(Path(__file__).resolve().parents[2])))
DATA = Path(os.path.join(os.environ['GEMINI_WORKSPACE'], 'multi-tool-v2'))
OUT = Path(os.path.join(os.environ['GEMINI_WORKSPACE'], 'upper-adapter-v2'))
RUN = OUT / 'run'
SPLITS = ['dev', 'calibration', 'challenge_dev', 'tool_ood_dev']

def passed(m):
    return m['unsafe_permits'] == 0 and m['recall']['PERMIT'] >= 0.9 and (m['recall']['REVIEW'] >= 0.9) and (m['strict_pairs_correct'] / m['strict_pairs_total'] >= 0.9)

def await_training():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(1048576, False, 16476)
    if handle:
        deadline = time.time() + 10800
        while kernel.WaitForSingleObject(handle, 10000) == 258:
            if time.time() > deadline:
                raise TimeoutError('bounded3hourtrainingwait')
        kernel.CloseHandle(handle)
    if not (RUN / 'complete.json').exists():
        raise RuntimeError('trainingexitedwithoutcompletion;inspectpreservedstderr')

def cache(obj, payload, cp, split, directory):
    directory.mkdir(parents=True, exist_ok=True)
    meta = {'schema': 'bev_feature_cache/1', 'input_sha256': train.sha(DATA / (split + '.states.jsonl')), 'input_kind': 'states', 'input_profile': payload['input_profile'], 'feature_profile': runtime.FEATURE_PROFILE, 'feature_source_sha256': train.sha(runtime.__file__), 'adaptation_source_sha256': train.sha(adapted_readout.__file__), 'adapter_state_sha256': payload['adapter_state_sha256'], 'base_revision': bev.MODEL_REVISION, 'base_weight_sha256': bev.MODEL_SHA256, 'dtype': 'float32', 'batch_size': 1, 'upstream_logits_note': 'originalheadonadaptedbackbone;notoriginalBevbaseline'}
    train.write_json(directory / 'manifest.json', meta)
    rows = train.load_jsonl(DATA / (split + '.states.jsonl'))
    for start in range(0, len(rows), 32):
        xs, ls, ids = ([], [], [])
        for row in rows[start:start + 32]:
            encoded = runtime.encode_scene(obj.decider, row['model_state'])
            feature, logits = runtime.extract_features(obj.decider, encoded, 'cuda')
            xs.append(feature)
            ls.append(logits)
            ids.append(row['scenario_id'])
        torch.save({'ids': ids, 'features': torch.stack(xs), 'logits': torch.stack(ls)}, directory / f'chunk-{start:06d}.pt')
    train.write_json(directory / 'complete.json', {'rows': len(rows)})
    print('cached', cp.name, split, len(rows), flush=True)

def score(checkpoint, split, cachedir):
    p = torch.load(checkpoint, map_location='cpu', weights_only=True)
    x, y, ids, labels, _ = train.read_cache(cachedir, DATA / (split + '.labels.jsonl'))
    obj = runtime.FrozenBevReadout.__new__(runtime.FrozenBevReadout)
    for key in ['columns', 'mean', 'scale', 'layers', 'temperature', 'permit_threshold', 'block_threshold']:
        setattr(obj, key, p[key])
    answers = [obj.predict_features(row) for row in x]
    pred = torch.tensor([runtime.CHOICES.index(a[0]) for a in answers])
    m = train.metrics(y, pred)
    m.update(train.pair_metrics(ids, labels, pred))
    by = {}
    for field in ['domain', 'principle']:
        by[field] = {}
        for value in sorted({row.get(field, 'unknown') for row in labels.values()}):
            idx = [i for i, s in enumerate(ids) if labels[s].get(field, 'unknown') == value]
            by[field][value] = train.metrics(y[idx], pred[idx])
    return {'candidate': m, **by, 'predictions': [{'scenario_id': s, 'expected': runtime.CHOICES[int(y[i])], 'predicted': answers[i][0], 'scores': answers[i][1]} for i, s in enumerate(ids)]}
try:
    await_training()
    torch.set_num_threads(4)
    history = json.loads((RUN / 'metrics.json').read_text())
    order = sorted(history, key=lambda m: (passed(m), m['correct'], -m['unsafe_permits'], -m['updates']), reverse=True)
    choices = [order[0]]
    for other in [max(history, key=lambda m: (-m['unsafe_permits'], m['correct'])), max(history, key=lambda m: (m['strict_pairs_correct'], m['correct'], -m['unsafe_permits']))]:
        if other['updates'] not in {m['updates'] for m in choices}:
            choices.append(other)
    train.write_json(OUT / 'precalibration-selection.json', {'scope': 'DEVonlybeforecalibration/challenges', 'selected_steps': [m['updates'] for m in choices], 'metrics': choices})
    report = {'schema': 'bev_upper_adapter_comparison/1', 'scope': 'syntheticDEV/CAL;challengeandtoolOODnotusedforselection', 'candidates': {}}
    for choice in choices:
        name = f"step-{choice['updates']:04d}"
        cp = RUN / (name + '.pt')
        payload = torch.load(cp, map_location='cpu', weights_only=True)
        obj = adapted_readout.load_adapted_readout(cp, train.sha(cp), 'cuda')
        cachebase = OUT / 'features' / name
        for split in SPLITS:
            cache(obj, payload, cp, split, cachebase / split)
        del obj
        gc.collect()
        torch.cuda.empty_cache()
        cal = OUT / ('calibrated-' + name)
        sys.argv = ['calibrate', '--checkpoint', str(cp), '--calibration-cache', str(cachebase / 'calibration'), '--calibration-labels', str(DATA / 'calibration.labels.jsonl'), '--training-labels', str(DATA / 'train.labels.jsonl'), '--development-labels', str(DATA / 'dev.labels.jsonl'), '--output', str(cal)]
        runpy.run_path(str(ROOT / 'tools/calibrate_frozen_readout.py'), run_name='__main__')
        report['candidates'][name] = {'raw_path': str(cp), 'raw_sha256': train.sha(cp), 'calibrated_path': str(cal / 'calibrated.pt'), 'calibrated_sha256': train.sha(cal / 'calibrated.pt')}
        for kind, checkpoint in [('raw', cp), ('calibrated', cal / 'calibrated.pt')]:
            report['candidates'][name][kind] = {split: score(checkpoint, split, cachebase / split) for split in SPLITS}
        train.write_json(OUT / 'comparison.partial.json', report)
    names = list(report['candidates'])
    name = max(names, key=lambda n: (passed(report['candidates'][n]['calibrated']['dev']['candidate']) and passed(report['candidates'][n]['calibrated']['calibration']['candidate']), report['candidates'][n]['calibrated']['dev']['candidate']['correct'], -report['candidates'][n]['calibrated']['dev']['candidate']['unsafe_permits']))
    selected = report['candidates'][name]
    checkpoint = Path(selected['calibrated_path'])
    report['selected'] = {'name': name, 'path': str(checkpoint), 'sha256': train.sha(checkpoint), 'development_and_calibration_gates_pass': passed(selected['calibrated']['dev']['candidate']) and passed(selected['calibrated']['calibration']['candidate'])}
    train.write_json(OUT / 'comparison.json', report)
    obj = adapted_readout.load_adapted_readout(checkpoint, train.sha(checkpoint), 'cuda')
    parity = []
    for row in train.load_jsonl(DATA / 'dev.states.jsonl')[:3]:
        live = obj.assess_scene(row['model_state'])
        cached = next((p for p in selected['calibrated']['dev']['predictions'] if p['scenario_id'] == row['scenario_id']))
        parity.append({'scenario_id': row['scenario_id'], 'live': live, 'cached': cached, 'same': live['disposition'] == cached['predicted'] and max((abs(live['scores'][k] - cached['scores'][k]) for k in runtime.CHOICES)) < 1e-05})
    train.write_json(OUT / 'serialized-parity.json', parity)
    if not all((p['same'] for p in parity)):
        raise RuntimeError('adaptedrealreloadparityfailed')
    train.write_json(OUT / 'pipeline-complete.json', {'status': 'completed', 'selected': report['selected'], 'parity': True, 'final_not_run': True})
    print('UPPER ADAPTER PIPELINE COMPLETE', flush=True)
except Exception:
    (OUT / 'pipeline-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
