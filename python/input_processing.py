from dataclasses import dataclass
import numpy as np
from skimage.transform import pyramid_gaussian


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the luminance formula
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


@dataclass
class ImagePyramids:
    """
    A dataclass to hold the generated image pyramids
    """
    pyramidA: tuple[np.ndarray]
    pyramidAPrime: tuple[np.ndarray]
    pyramidB: tuple[np.ndarray]
    pyramidBPrime: list[np.ndarray]
    levels: int = 5

def inputsToPyramids(inputImgs: list[np.ndarray], pyramidLevels: int = 5) -> ImagePyramids:
    """
    Processes the raw input images and converts them to image pyramids
    for the purpose of the algorithm.
    """

    remappedImgs = [img / 255.0 for img in inputImgs]

    # Use grayscale (luminance) features for the distance metric as recommended
    # in Hertzmann et al. 2001
    grayImgs = []
    for img in remappedImgs:
        if len(img.shape) > 2 and img.shape[2] >= 3:
            grayImgs.append(rgb2gray(img[..., :3]).astype(np.float32))
        else:
            grayImgs.append(np.asarray(img, dtype=np.float32))

    # Downscale level 2 referenced from Fiser et al 2016
    pyramidA, pyramidAPrime, pyramidB = [
        tuple(pyramid_gaussian(img, pyramidLevels, downscale=2, channel_axis=None))
        for img in grayImgs[:3]
    ]
    
    pyramidBPrime = []
    
    for level in range(len(pyramidB)):
        pyramidBPrime.append(np.zeros(pyramidB[level].shape))
        
    return ImagePyramids(pyramidA, pyramidAPrime, pyramidB, pyramidBPrime, pyramidLevels)
    
    