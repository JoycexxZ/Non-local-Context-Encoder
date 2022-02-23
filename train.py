import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--data_path', type=str, default='datasets/JPCL/images/')
    parser.add_argument('--mask_path', type=str, default='datasets/JPCL/masks/')
    parser.add_argument('--dataset', type=str, default='JPCL')
    parser.add_argument('--image_size', type=int, default=256)

    # Training params
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--first_momentum', type=float, default=0.9)
    parser.add_argument('--second_momentum', type=float, default=0.999)

    # Training settings
    parser.add_argument('--out_to_folder', type=str, default='False')

    # Other params
    parser.add_argument('--lamb', type=float, default=0.25)