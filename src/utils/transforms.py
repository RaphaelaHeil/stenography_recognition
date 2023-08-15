from math import ceil
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image
from skimage.transform import rescale
from torchvision.transforms import Resize


class PadSequence(object):

    def __init__(self, length: int, padwith: int = 0):
        self.length = length
        self.padwith = padwith

    def __call__(self, sequence: torch.Tensor):
        sequenceLength = sequence.shape[0]
        if sequenceLength == self.length:
            return sequence
        targetLength = self.length - sequenceLength
        return F.pad(sequence, pad=(0, targetLength), mode="constant", value=self.padwith)


class ResizeToHeight(Resize):

    def __init__(self, size: int):
        super().__init__(size)
        if isinstance(size, Tuple):
            self.height = size[0]
        else:
            self.height = size

    def forward(self, img:Image):
        oldWidth, oldHeight = img.size
        if oldHeight > oldWidth:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            return tvF.resize(img, [self.height, intermediateWidth], self.interpolation, self.max_size, self.antialias)
        else:
            return super().forward(img)


class ResizeAndPad(object):
    """
    Custom transformation that maintains the image's original aspect ratio by scaling it to the given height and padding
    it to achieve the desired width.
    """

    def __init__(self, height: int, width: int, padwith: int = 1):
        self.width = width
        self.height = height
        self.padwith = padwith

    def __call__(self, img: Image):
        oldWidth, oldHeight = img.size
        if oldWidth == self.width and oldHeight == self.height:
            return img
        else:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            if intermediateWidth > self.width:
                intermediateWidth = self.width
            resized = img.resize((intermediateWidth, self.height), resample=Image.BICUBIC) #Image.Resampling.BICUBIC
            if img.mode == "RGB":
                padValue = (self.padwith, self.padwith, self.padwith) # TODO: add option to pad with mean?! O.O
            else:
                padValue = self.padwith
            preprocessed = Image.new(img.mode, (self.width, self.height), padValue)
            preprocessed.paste(resized)
            return preprocessed

    @classmethod
    def invert(cls, image: np.ndarray, targetShape: Tuple[int, int]) -> np.ndarray:
        # resize so that the height matches, then cut off at width ...
        originalHeight, originalWidth = image.shape
        scaleFactor = targetShape[0] / originalHeight
        resized = rescale(image, scaleFactor)
        return resized[:, :targetShape[1]]

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
