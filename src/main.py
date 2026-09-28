import argparse
import json
from utils.quick_start import quick_start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MGCN', help='name of model')
    parser.add_argument('--dataset', '-d', default='baby', help='name of dataset')
    parser.add_argument('--gpu_id', '-g', default='0',
                        help='GPU IDs visible to workers, e.g. 0,1,2,3; one trial per GPU')
    parser.add_argument('--save_dir', default='./saved_models', help='search output directory')
    parser.add_argument('--threads_per_worker', type=int, default=4, help='CPU threads per trial')
    parser.add_argument('--epochs', type=int, help='maximum epochs per trial')
    parser.add_argument('--stopping_step', type=int, help='early stopping patience')
    parser.add_argument('--cpu', action='store_true', help='run trials sequentially on CPU')
    parser.add_argument('--vanilla', action='store_true', help='disable DAMPS in MGCN')
    parser.add_argument('--config_json', help='JSON configuration overrides (scalar search values run once)')
    args = parser.parse_args()
    config_dict = {}
    if args.config_json:
        with open(args.config_json, encoding='utf-8') as stream:
            config_dict.update(json.load(stream))
    config_dict.update({
        'gpu_id': args.gpu_id,
        'save_dir': args.save_dir,
        'threads_per_worker': args.threads_per_worker,
    })
    if args.vanilla:
        if args.model != 'MGCN':
            parser.error('--vanilla is currently supported only for MGCN')
        config_dict['use_damps'] = False
    if args.cpu:
        config_dict['use_gpu'] = False
    for name in ('epochs', 'stopping_step'):
        if getattr(args, name) is not None:
            config_dict[name] = getattr(args, name)
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
