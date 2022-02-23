from cv2 import transform
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
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
        elif self.dataset == "ISBI":
            image, mask = utils.get_data_ISBI(self.config.data_path, self.config.mask_path, name)


        #image = torch.from_numpy(image.astype(np.float32) / max_pixel).contiguous()
        #mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()

        if self.transform and self.dataset == "ISBI":
            image = self.transform(image)
            mask = self.transform(mask)

        if self.dataset == "JPCL":
            image = np.resize(image, (1,256, 256))
            mask = np.resize(mask, (1,256, 256))

        return image, mask


def get_training_set(config, batch_size, num_workers):
    transformed_train = GeneralDataset(config, transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(256),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(10),
                                            transforms.ToTensor()
                                            ]))

    dataloader_train = DataLoader(transformed_train, batch_size, shuffle=True, num_workers=num_workers)
    return dataloader_train

def get_testing_set(config, batch_size, num_workers):
    transformed_test = GeneralDataset(config, transforms = transforms.Compose([transforms.ToTensor()]))
    dataloader_test = DataLoader(transformed_test, batch_size, shuffle=True, num_workers=num_workers)

    return dataloader_test