"""One isolated training trial. Started by quick_start in a fresh interpreter."""
import fcntl
import json
import math
import os
import sys
import time
from logging import getLogger

import torch

from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.search_io import write_json
from utils.utils import get_model, get_trainer, init_seed


def run_trial(payload):
    started_at = time.time()
    config = Config(payload['model'], payload['dataset'], payload['config'], payload['mg'])
    if config['use_gpu'] and config['device'].type != 'cuda':
        raise RuntimeError('Requested GPU {} is not available'.format(config['gpu_id']))
    torch.set_num_threads(config['threads_per_worker'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    init_seed(config['seed'])
    dataset = RecDataset(config)
    logger.info(str(dataset))
    train, valid, test = dataset.split()
    for name, split in [('Training', train), ('Validation', valid), ('Testing', test)]:
        logger.info('%s\n%s', name, split)
    train_data = TrainDataLoader(config, train, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid, additional_dataset=train,
                               batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test, additional_dataset=train,
                              batch_size=config['eval_batch_size'])
    train_data.pretrain_setup()
    # Existing models share graph-cache files. Serialize initialization, not training,
    # to prevent simultaneous readers/writers from seeing incomplete cache files.
    with open(os.path.join(dataset.dataset_path, '.model_init.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer()(config, model, payload['mg'])
    score, valid_result, test_result = trainer.fit(
        train_data, valid_data, test_data, saved=payload['save_model'])
    if trainer.best_epoch is None or not math.isfinite(float(score)):
        raise RuntimeError('Trial produced no finite validation result')
    result = {
        'trial_id': config['trial_id'], 'status': 'completed',
        'started_at': started_at, 'finished_at': time.time(),
        'pid': os.getpid(), 'device': str(config['device']),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'parameters': config['trial_parameters'], 'gpu_id': config['gpu_id'],
        'best_epoch': trainer.best_epoch, 'valid_score': float(score),
        'test_score': float(test_result[config['valid_metric'].lower()]),
        'valid_result': valid_result, 'test_result': test_result,
        'checkpoint': config['checkpoint_path'] if payload['save_model'] else None,
    }
    write_json(payload['result_path'], result)
    return result


if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf-8') as stream:
        run_trial(json.load(stream))
