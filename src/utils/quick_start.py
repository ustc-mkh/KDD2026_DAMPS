"""Schedule independent hyperparameter trials, with one process per GPU."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from itertools import product
from logging import getLogger
from pathlib import Path

from utils.search_io import write_json


def parameter_combinations(config):
    names = list(dict.fromkeys(config['hyper_parameters']))
    candidates = []
    for name in names:
        value = config[name]
        if value is None or (isinstance(value, (list, tuple)) and not value):
            raise ValueError('Missing or empty search parameter: {}'.format(name))
        candidates.append(value if isinstance(value, (list, tuple)) else [value])
    return [dict(zip(names, values)) for values in product(*candidates)]


def select_best(results, bigger, field='valid_score'):
    successful = [r for r in results if r['status'] == 'completed']
    if not successful:
        return None
    # Prefer the earlier trial on ties, independent of completion order.
    return min(successful, key=lambda r: (
        -r[field] if bigger else r[field], r['trial_id']))


def _publish_best(result, run_dir, save_model):
    if save_model:
        destination = os.path.join(run_dir, 'best.pth')
        temporary = destination + '.tmp'
        try:
            shutil.copyfile(result['checkpoint'], temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    write_json(os.path.join(run_dir, 'best.json'), result)


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    from utils.configurator import Config
    from utils.logger import init_logger

    # The coordinator never initializes CUDA. Each new interpreter sees only its GPU.
    config = Config(model, dataset, dict(config_dict or {}), mg, initialize_device=False)
    combinations = parameter_combinations(config)
    if config['use_gpu']:
        gpu_ids = [part.strip() for part in str(config['gpu_id']).split(',')]
        if any(not part for part in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError('gpu_id must contain distinct GPU IDs, e.g. 0,1,2,3')
    else:
        gpu_ids = [None]
    threads = config['threads_per_worker']
    if threads is None:
        threads = 4
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('threads_per_worker must be a positive integer')

    for name in ('epochs', 'eval_step'):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError('{} must be a positive integer'.format(name))

    save_dir = os.path.abspath(config['save_dir'] or './saved_models')
    os.makedirs(save_dir, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix='{}-{}-{}-'.format(
        model, dataset, datetime.now().strftime('%Y%m%d_%H%M%S')), dir=save_dir)
    config['log_file'] = os.path.join(run_dir, 'search.log')
    init_logger(config)
    logger = getLogger()
    logger.info('Search: %d trials, devices=%s, output=%s',
                len(combinations), gpu_ids, run_dir)
    write_json(os.path.join(run_dir, 'config.json'), config.final_config_dict)

    pending = deque(enumerate(combinations))
    active = {}
    results = []
    best_id = None
    metric = config['valid_metric'].lower()
    bigger = config['valid_metric_bigger']
    src_dir = str(Path(__file__).resolve().parents[1])

    def summarize(status):
        best = select_best(results, bigger)
        test_best = select_best(results, bigger, 'test_score')
        summary = {
            'status': status, 'total_trials': len(combinations), 'run_dir': run_dir,
            'selection_metric': metric, 'selection_split': 'validation',
            'best_trial_id': best['trial_id'] if best else None,
            # Diagnostic only: this does not select the saved model.
            'best_observed_test_trial_id': test_best['trial_id'] if test_best else None,
            'results': sorted(results, key=lambda r: r['trial_id']),
        }
        write_json(os.path.join(run_dir, 'results.json'), summary)
        return summary

    summarize('running')
    try:
        while pending or active:
            for gpu in gpu_ids:
                if gpu in active or not pending:
                    continue
                trial_id, parameters = pending.popleft()
                trial_dir = os.path.join(run_dir, 'trial_{:04d}'.format(trial_id))
                os.makedirs(trial_dir)
                trial_config = dict(config.final_config_dict)
                trial_config.update(parameters)
                trial_config.update({
                    'gpu_id': gpu if gpu is not None else '0',
                    'log_file': os.path.join(trial_dir, 'train.log'),
                    'checkpoint_path': os.path.join(trial_dir, 'best.pth'),
                    'recommend_topk': os.path.join(trial_dir, 'recommend_topk'),
                    'trial_id': trial_id, 'trial_parameters': parameters,
                    'threads_per_worker': threads,
                })
                payload = {'model': model, 'dataset': dataset, 'config': trial_config,
                           'save_model': save_model, 'mg': mg,
                           'result_path': os.path.join(trial_dir, 'result.json')}
                payload_path = os.path.join(trial_dir, 'input.json')
                write_json(payload_path, payload)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu if gpu is not None else ''
                env['OMP_NUM_THREADS'] = str(threads)
                env['MKL_NUM_THREADS'] = str(threads)
                env['PYTHONPATH'] = src_dir + os.pathsep + env.get('PYTHONPATH', '')
                output = open(os.path.join(trial_dir, 'worker.log'), 'w', encoding='utf-8')
                try:
                    process = subprocess.Popen(
                        [sys.executable, '-u', '-m', 'utils.search_worker', payload_path],
                        env=env, stdout=output, stderr=subprocess.STDOUT)
                except BaseException:
                    output.close()
                    raise
                active[gpu] = (process, output, payload)
                logger.info('Started trial %d/%d on GPU %s: %s',
                            trial_id + 1, len(combinations), gpu, parameters)

            for gpu, (process, output, payload) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                output.close()
                trial = payload['config']
                if code == 0 and os.path.isfile(payload['result_path']):
                    with open(payload['result_path'], encoding='utf-8') as stream:
                        result = json.load(stream)
                else:
                    result = {'trial_id': trial['trial_id'], 'status': 'failed',
                              'parameters': trial['trial_parameters'], 'gpu_id': gpu,
                              'exit_code': code, 'log': output.name}
                results.append(result)
                del active[gpu]
                logger.info('Trial %d %s (%d/%d finished)', result['trial_id'],
                            result['status'], len(results), len(combinations))
                best = select_best(results, bigger)
                if best is not None and best['trial_id'] != best_id:
                    _publish_best(best, run_dir, save_model)
                    best_id = best['trial_id']
                    logger.info('Current BEST by validation: trial=%d, valid=%s, test=%s',
                                best_id, best['valid_result'], best['test_result'])
                summarize('running')
            if active:
                time.sleep(0.2)
    except BaseException:
        for process, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, output, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            output.close()
        summarize('interrupted')
        raise

    failed = [r for r in results if r['status'] != 'completed']
    summary = summarize('completed_with_failures' if failed else 'completed')
    logger.info('Search finished. Results: %s', run_dir)
    if failed:
        raise RuntimeError('{} trial(s) failed; inspect results.json and trial worker.log files in {}'.format(
            len(failed), run_dir))
    return summary
