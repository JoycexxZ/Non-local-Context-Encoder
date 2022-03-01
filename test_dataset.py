from datasets.datasets import *
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Datasets
    # parser.add_argument('--data_path', type=str, default='datasets/JPCL/images/')
    # parser.add_argument('--mask_path', type=str, default='datasets/JPCL/masks/')
    # parser.add_argument('--dataset', type=str, default='JPCL')

    parser.add_argument('--data_path', type=str, default='datasets/ISBI/ISBI2016_ISIC_Part1_Training_Data/')
    parser.add_argument('--mask_path', type=str, default='datasets/ISBI/ISBI2016_ISIC_Part1_Training_GroundTruth/')
    parser.add_argument('--dataset', type=str, default='ISBI')

    parser.add_argument('--image_size', type=int, default=256)

    config = parser.parse_args()

    loader = get_training_loader(config, batch_size=900, num_workers=0)
    
    data = next(iter(loader))
    print(data[1].mean(dim = [0,2,3]), data[1].std(dim = [0,2,3]))

    # for i, (image, mask) in enumerate(loader):
    #     print(image.shape, mask.shape, image.max(), mask.max())
    #     if i > 10:
    #         break
