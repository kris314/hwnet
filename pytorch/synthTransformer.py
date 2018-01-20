import torch
import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import io, transform

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, roi, gt = sample['image'], sample['label'], sample['roi'], sample['gt']
        image = (image-np.mean(image)) / ((np.std(image) + 0.0001) / 128.0)

        return {'image': image,
                'label': label,
                'roi': roi,
                'gt': gt}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, roi = sample['image'], sample['label'], sample['roi']

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'roi': torch.from_numpy(roi)}
