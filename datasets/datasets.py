#from random import random
#from cv2 import transform
#from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
from datasets import utils
from torchvision import transforms


class GeneralDataset(Dataset):
    def __init__(self, config, transforms=None) -> None:
        super(GeneralDataset, self).__init__()
        self.config = config
        self.dataset = config.dataset
        self.transform = transforms
        self.filenames = utils.get_filenames(config.data_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        if self.dataset == "JPCL":
            image, mask = utils.get_data_JPCL(self.config.data_path, self.config.mask_path, name)
            image = utils.norm_JPCL(image)
            #image = np.resize(image, (1, 256, 256)).astype('float32')
            #mask = np.resize(mask, (1, 256, 256)).astype('float32')
            image = utils.reshape_image(image, self.config.image_size).reshape((1, self.config.image_size, self.config.image_size)).astype('float32')
            mask = utils.reshape_image(mask, self.config.image_size).reshape((self.config.image_size, self.config.image_size)).astype('int64')
        
        elif self.dataset == "ISBI":
            image, mask = utils.get_data_ISBI(self.config.data_path, self.config.mask_path, name)
            if self.transform:
                sample = {'image': image, 'mask': mask}
                sample = self.transform(sample)
                image = sample['image']
                mask = sample['mask'].reshape((self.config.image_size, self.config.image_size)).long()

        return image, mask


def get_training_loader(config, batch_size, num_workers):
    image_stats = {'mean':[0.7331, 0.6158, 0.5599],
                   'std':[0.1522, 0.1724, 0.1930]}
    transformed_train = GeneralDataset(config, transforms= transforms.Compose([
                                            utils.RandomHorizontalFlip(),
                                            utils.RandomVerticalFlip(),
                                            utils.RandomRotate(10),
                                            utils.Scale(256),
                                            utils.CenterCrop([256, 256], [256, 256]),
                                            utils.ToTensor(),
                                            utils.Normalize(image_stats['mean'], image_stats['std'])
                                            ])
                                            )
    
    dataloader_train = DataLoader(transformed_train, batch_size, shuffle=True, num_workers=num_workers)
    return dataloader_train

def get_testing_loader(config, batch_size, num_workers):
    transformed_test = GeneralDataset(config, transforms_image=transforms.Compose([
                                            utils.Scale(256),
                                            utils.CenterCrop([256,256], [256, 256]),
                                            utils.ToTensor()
                                            ]))
    dataloader_test = DataLoader(transformed_test, batch_size, shuffle=True, num_workers=num_workers)

    return dataloader_test
