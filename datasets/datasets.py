from torch.utils.data import Dataset
import numpy as np
import torch
from datasets import utils


class GeneralDataset(Dataset):
    def __init__(self, config) -> None:
        super(GeneralDataset, self).__init__()
        self.config = config
        self.dataset = config.dataset
        
        self.filenames = utils.get_filenames(config.data_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        if self.dataset == "JPCL":
            image, mask = utils.get_data_JPCL(self.config.data_path, self.config.mask_path, name)
        elif self.dataset == "ISBI":
            image, mask = utils.get_data_ISBI(self.config.data_path, self.config.mask_path, name)

        image = utils.reshape_image(image, self.config.image_size)
        mask = utils.reshape_image(mask, self.config.image_size)

        max_pixel = image.max()

        image = torch.from_numpy(image.astype(np.float32) / max_pixel).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()

        return image, mask


def get_training_loader(config, batch_size, num_workers):
    pass

def get_testing_loader(config, batch_size, num_workers):
    pass