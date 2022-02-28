import argparse
from engine import Engine

'''
    To Run the code:

    python train.py --dataset JPCL --batch_size 4 --num_workers 1 --out_to_folder True --epochs 500
'''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    # parser.add_argument('--dataset', type=str, default='JPCL')
    parser.add_argument('--dataset', type=str, default='ISBI')
    parser.add_argument('--image_size', type=int, default=256)

    # Training params
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--first_momentum', type=float, default=0.9)
    parser.add_argument('--second_momentum', type=float, default=0.999)

    # Training settings
    parser.add_argument('--out_to_folder', type=str, default='False')
    parser.add_argument('--model_path', type=str, default='')

    # Other params
    parser.add_argument('--lamb', type=float, default=0.25)

    config = parser.parse_args()

    if config.dataset == 'ISBI':
        config.data_path = 'datasets/ISBI/ISBI2016_ISIC_Part1_Training_Data/'
        config.mask_path = 'datasets/ISBI/ISBI2016_ISIC_Part1_Training_GroundTruth/'
    elif config.dataset == 'JPCL':
        config.data_path = 'datasets/JPCL/images/'
        config.mask_path = 'datasets/JPCL/masks/'
    else:
        raise ValueError('Dataset not supported')

    engine = Engine(config)
    engine.train()