"""Run with PYTHONPATH=src python -m unittest discover -s tests -v.
Set TEST_GPU_IDS=1,2 to exercise real multi-GPU subprocess scheduling.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from common.trainer import Trainer
from utils.configurator import Config
from utils.quick_start import parameter_combinations, quick_start, select_best


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass


class SearchTests(unittest.TestCase):
    def test_combinations_and_validation_selection(self):
        config = Config('MGCN', 'baby', {}, initialize_device=False)
        self.assertEqual(len(parameter_combinations(config)), 192)
        config['knn_k'] = 10
        self.assertEqual(len(parameter_combinations(config)), 48)
        config['knn_k'] = None
        with self.assertRaises(ValueError):
            parameter_combinations(config)
        results = [
            {'trial_id': 1, 'status': 'completed', 'valid_score': .7, 'test_score': .9},
            {'trial_id': 0, 'status': 'completed', 'valid_score': .8, 'test_score': .1},
        ]
        self.assertEqual(select_best(results, True)['trial_id'], 0)
        self.assertEqual(select_best(results, False)['trial_id'], 1)
        self.assertEqual(select_best(results, True, 'test_score')['trial_id'], 1)
        results[0]['valid_score'] = .8
        self.assertEqual(select_best(results, True)['trial_id'], 0)

    def test_checkpoint_restores_best_epoch_for_both_metric_directions(self):
        for bigger in (True, False):
            with self.subTest(bigger=bigger), tempfile.TemporaryDirectory() as tmp:
                config = Config('MGCN', 'baby', {
                    'use_gpu': False, 'epochs': 3, 'eval_step': 1,
                    'checkpoint_path': str(Path(tmp) / 'best.pth'),
                    'trial_parameters': {'seed': 999},
                })
                config['valid_metric_bigger'] = bigger
                model = ToyModel()
                trainer = Trainer(config, model)
                validation = [.6, .9, .7] if bigger else [.6, .2, .4]
                testing = [.8, .1, .99]

                def train_epoch(data, epoch):
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    trainer.optimizer.step()
                    return 0., []

                def evaluate(data):
                    epoch = int(model.weight.item())
                    value = validation[epoch] if data == 'valid' else testing[epoch]
                    return value, {'recall@20': value}

                trainer._train_epoch = train_epoch
                trainer._valid_epoch = evaluate
                score, valid, test = trainer.fit(None, 'valid', 'test', saved=True, verbose=False)
                checkpoint = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=False)
                self.assertEqual(trainer.best_epoch, 1)
                self.assertEqual(model.weight.item(), 1.)
                self.assertEqual(checkpoint['model_state_dict']['weight'].item(), 1.)
                self.assertEqual(checkpoint['epoch'], 1)
                self.assertEqual(score, validation[1])
                self.assertEqual(test['recall@20'], testing[1])
                self.assertEqual(checkpoint['test_result'], test)

    def test_cpu_subprocess_search_and_reload(self):
        self._run_search(None)

    @unittest.skipUnless(os.environ.get('TEST_GPU_IDS'), 'TEST_GPU_IDS is not set')
    def test_multi_gpu_subprocess_search_and_reload(self):
        self._run_search(os.environ['TEST_GPU_IDS'])

    def test_failed_trial_does_not_stop_remaining_trials(self):
        self._run_search(None, include_failure=True)

    def test_interrupt_terminates_all_active_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            processes = [MagicMock(), MagicMock()]
            for process in processes:
                process.poll.return_value = None
            options = {
                'use_gpu': True, 'gpu_id': '1,2', 'save_dir': tmp,
                'n_ui_layers': [1], 'n_layers': [1], 'cl_loss': [.01, .02], 'knn_k': [2],
            }
            with patch('utils.quick_start.subprocess.Popen', side_effect=processes), \
                    patch('utils.quick_start.time.sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    quick_start('MGCN', 'baby', options)
            for process in processes:
                process.terminate.assert_called_once()
                process.wait.assert_called_once()
            result_path = next(Path(tmp).glob('*/results.json'))
            self.assertEqual(json.loads(result_path.read_text())['status'], 'interrupted')

    def test_vanilla_search_and_reload(self):
        self._run_search(None, vanilla=True)

    def _run_search(self, gpu_ids, include_failure=False, vanilla=False):
        from utils.dataset import RecDataset
        from utils.dataloader import TrainDataLoader, EvalDataLoader
        from models.mgcn import MGCN

        with tempfile.TemporaryDirectory(prefix='damps-search-test-') as tmp:
            data = Path(tmp) / 'baby'
            data.mkdir()
            lines = ['userID\titemID\tx_label']
            for user in range(4):
                for offset, split in [(0, 0), (1, 0), (2, 1), (3, 2)]:
                    lines.append('{}\t{}\t{}'.format(user, (2 * user + offset) % 8, split))
            (data / 'baby.inter').write_text('\n'.join(lines) + '\n')
            rng = np.random.RandomState(42)
            np.save(str(data / 'image_feat.npy'), rng.rand(8, 6).astype('float32'))
            np.save(str(data / 'text_feat.npy'), rng.rand(8, 5).astype('float32'))
            options = {
                'data_path': tmp + '/', 'save_dir': str(Path(tmp) / 'outputs'),
                'use_gpu': bool(gpu_ids), 'gpu_id': gpu_ids or '0',
                'n_ui_layers': [1], 'n_layers': [1], 'cl_loss': [.01, .02],
                'knn_k': [2], 'seed': [999], 'epochs': 2,
                'embedding_size': 8, 'topk': [1, 3], 'valid_metric': 'Recall@3',
                'metrics': ['Recall', 'NDCG'], 'train_batch_size': 4,
                'eval_batch_size': 4, 'threads_per_worker': 1,
            }
            if vanilla:
                options['use_damps'] = False
            if include_failure:
                options.update({'knn_k': [2, 99, 3], 'cl_loss': [.01]})
            # Search configuration must not initialize CUDA in the coordinator.
            with patch.object(Config, '_init_device', side_effect=AssertionError('parent CUDA init')):
                if include_failure:
                    with self.assertRaisesRegex(RuntimeError, '1 trial\\(s\\) failed'):
                        quick_start('MGCN', 'baby', options)
                    result_path = next((Path(tmp) / 'outputs').glob('*/results.json'))
                    summary = json.loads(result_path.read_text())
                else:
                    summary = quick_start('MGCN', 'baby', options)
            expected_status = 'completed_with_failures' if include_failure else 'completed'
            self.assertEqual(summary['status'], expected_status)
            if include_failure:
                self.assertEqual(len(summary['results']), 3)
                failed = summary['results'][1]
                self.assertEqual(failed['status'], 'failed')
                self.assertIn('out of range', Path(failed['log']).read_text())
            results = [r for r in summary['results'] if r['status'] == 'completed']
            self.assertEqual(len(results), 2)
            self.assertEqual(len({r['pid'] for r in results}), 2)
            self.assertTrue(all(r['pid'] != os.getpid() for r in results))
            if gpu_ids:
                devices = gpu_ids.split(',')
                self.assertEqual({r['cuda_visible_devices'] for r in results}, set(devices[:2]))
                self.assertTrue(all(r['device'] == 'cuda' for r in results))
                self.assertLess(max(r['started_at'] for r in results),
                                min(r['finished_at'] for r in results))
            best = select_best(results, True)
            self.assertEqual(summary['best_trial_id'], best['trial_id'])
            run_dir = Path(summary['run_dir'])
            checkpoint = torch.load(str(run_dir / 'best.pth'), map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['epoch'], best['best_epoch'])
            self.assertEqual(checkpoint['valid_result'], best['valid_result'])
            self.assertEqual(checkpoint['test_result'], best['test_result'])
            if vanilla:
                self.assertFalse(any(key.startswith('feature_filter.')
                                     for key in checkpoint['model_state_dict']))
            else:
                self.assertIn('feature_filter.avg_R', checkpoint['model_state_dict'])
            for result in results:
                self.assertTrue(Path(result['checkpoint']).is_file())
            self.assertEqual(json.loads((run_dir / 'results.json').read_text())['status'], expected_status)

            # A different initialization must still reproduce the saved test metrics.
            saved_config = dict(checkpoint['config'])
            saved_config['use_gpu'] = False
            config = Config('MGCN', 'baby', saved_config)
            dataset = RecDataset(config)
            str(dataset)
            train, valid, test = dataset.split()
            for split in (train, valid, test):
                str(split)
            train_data = TrainDataLoader(config, train, batch_size=4)
            test_data = EvalDataLoader(config, test, additional_dataset=train, batch_size=4)
            torch.manual_seed(12345)
            model = MGCN(config, train_data)
            model.load_state_dict(checkpoint['model_state_dict'])
            metrics = Trainer(config, model).evaluate(test_data)
            self.assertEqual(metrics, best['test_result'])


if __name__ == '__main__':
    unittest.main()
