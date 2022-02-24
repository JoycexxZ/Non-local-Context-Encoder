import argparse

from engine import Engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--data_path', type=str, default='datasets/JPCL/images/')
    parser.add_argument('--mask_path', type=str, default='datasets/JPCL/masks/')
    parser.add_argument('--dataset', type=str, default='JPCL')
    parser.add_argument('--image_size', type=int, default=256)

    # Testing params
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')

    # Testing settings
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--out_to_folder', type=str, default='True')

    config = parser.parse_args()

    engine = Engine(config)
    engine.test()