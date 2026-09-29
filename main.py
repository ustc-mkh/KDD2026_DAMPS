import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='KSMMRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='gpu id')
    
    args, _ = parser.parse_known_args()
    
    save_dir = './saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    config_dict = {
        'gpu_id': int(args.gpu_id.split(',')[0]),
        'save_dir': save_dir  
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

