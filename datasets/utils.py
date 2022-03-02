import os
import numpy as np
from PIL import Image
import cv2
import random
import torch
import accimage
import scipy.ndimage as ndimage

def get_filenames(path):
    names = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            fname, _ = os.path.splitext(filespath)
            names.append(fname)
    return names

def get_data_JPCL(data_path, mask_path, name):
    data_path = os.path.join(data_path, name+".IMG")
    shape = (2048, 2048, 1)
    dtype = np.dtype('>u2') 
    fid = open(data_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    mask_left_path = os.path.join(mask_path, "left lung", name+".gif")
    mask_right_path = os.path.join(mask_path, "right lung", name+".gif")
    mask_left = np.array(Image.open(mask_left_path))
    mask_right = np.array(Image.open(mask_right_path))
    mask = np.minimum(mask_left + mask_right, 1)
    #mask = mask_left + mask_right

    return image, mask
    
def get_data_ISBI(data_path, mask_path, name):
    data_path = os.path.join(data_path, name+".jpg")
    image = Image.open(data_path)
    mask_path = os.path.join(mask_path, name+"_Segmentation.png")
    mask = Image.open(mask_path)

    return image, mask

def reshape_image(image, image_size):
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape

    if h > w:
        image = image[w//2:w//2+w, :]
    elif h < w:
        image = image[:, h//2:h//2+h]

    image = cv2.resize(image, (image_size, image_size))
    return image

def norm_JPCL(image):
    max_pixel = image.max()
    return image/max_pixel


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'mask': mask}
    
class RandomVerticalFlip(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': image, 'mask': mask}
    
class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['mask']
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        image = self.to_tensor(image)
        mask = self.to_tensor(depth).float()
        return {'image': image, 'mask': mask}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img
        
        
class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image, depth = sample['image'], sample['mask']

        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180

        image = ndimage.interpolation.rotate(
            image, angle1, reshape=self.reshape, order=self.order)
        depth = ndimage.interpolation.rotate(
            depth, angle1, reshape=self.reshape, order=self.order)

        image = Image.fromarray(image)
        depth = Image.fromarray(depth)

        return {'image': image, 'mask': depth}


class CenterCrop(object):
    def __init__(self, size_image, size_depth):
        self.size_image = size_image
        self.size_depth = size_depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['mask']

        image = self.centerCrop(image, self.size_image)
        depth = self.centerCrop(depth, self.size_image)

        ow, oh = self.size_depth
        depth = depth.resize((ow, oh))

        return {'image': image, 'mask': depth}

    def centerCrop(self, image, size):

        w1, h1 = image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image
    
class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, depth = sample['image'], sample['mask']

        image = self.changeScale(image, self.size)
        depth = self.changeScale(depth, self.size,Image.NEAREST)
 
        return {'image': image, 'mask': depth}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, depth = sample['image'], sample['mask']

        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'mask': depth}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor