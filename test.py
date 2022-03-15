import argparse
from engine import Engine

'''
    To Run the code:

    python test.py --dataset JPCL --batch_size 8 --num_workers 2 --model_path 'results/1646138004/model_250.pth'
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--dataset', type=str, default='ISBI')
    parser.add_argument('--image_size', type=int, default=256)

    # Testing params
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')

    # Testing settings
    parser.add_argument('--model_path', type=str, default='results/1645712917/model_20.pth')
    parser.add_argument('--out_to_folder', type=str, default='True')
    parser.add_argument('--use_adv', type=float, default=0)

    config = parser.parse_args()

    if config.dataset == 'ISBI':
        if config.use_adv > 0:
            config.data_path = "datasets/adversarial/ISBI_" + str(config.use_adv) + "/"
        else: 
            config.data_path = 'datasets/ISBI/ISBI2016_ISIC_Part1_Test_Data/'
        config.mask_path = 'datasets/ISBI/ISBI2016_ISIC_Part1_Test_GroundTruth/'
    elif config.dataset == 'JPCL':
        if config.use_adv > 0:
            config.data_path = "datasets/adversarial/JPCL_" + str(config.use_adv) + "/"
        else:
            config.data_path = 'datasets/JPCL/fold2/images/'
        config.mask_path = 'datasets/JPCL/fold2/masks/'
    else:
        raise ValueError('Dataset not supported')

    engine = Engine(config)
    engine.test()