import argparse
from engine import Engine

'''
    To Run the code:

    python generate_adversarial.py --dataset ISBI --batch_size 2 --num_workers 1 --model_path 'results/1646037293/model_20.pth --eps 0.5
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--dataset', type=str, default='ISBI')
    parser.add_argument('--image_size', type=int, default=256)

    # Params
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')

    # Settings
    parser.add_argument('--model_path', type=str, default='results/1645712917/model_20.pth')
    parser.add_argument('--alpha', type=int, default=1)

    config = parser.parse_args()

    if config.dataset == 'ISBI':
        config.data_path = 'datasets/ISBI/ISBI2016_ISIC_Part1_Test_Data/'
        config.mask_path = 'datasets/ISBI/ISBI2016_ISIC_Part1_Test_GroundTruth/'
    elif config.dataset == 'JPCL':
        config.data_path = 'datasets/JPCL/fold2/images/'
        config.mask_path = 'datasets/JPCL/fold2/masks/'
    else:
        raise ValueError('Dataset not supported')

    config.adversary_dir = 'datasets/adversarial/'+config.dataset+f'_{config.eps}/'

    engine = Engine(config)
    engine.generate_adversarial_samples(adversary_dir=config.adversary_dir, eps=config.eps)