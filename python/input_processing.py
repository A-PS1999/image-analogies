from dataclasses import dataclass
import numpy as np
from skimage.transform import pyramid_gaussian

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
    channelAxes = [-1 if len(img.shape) > 2 else None for img in remappedImgs]

    # Downscale level 2 referenced from Fiser et al 2016
    pyramidA, pyramidAPrime, pyramidB = [
        tuple(pyramid_gaussian(img, pyramidLevels, downscale=2, channel_axis=axis))
        for img, axis in zip(remappedImgs[:3], channelAxes[:3])
    ]
    
    pyramidBPrime = []
    
    for level in range(len(pyramidB)):
        pyramidBPrime.append(np.zeros(pyramidB[level].shape))
        
    return ImagePyramids(pyramidA, pyramidAPrime, pyramidB, pyramidBPrime, pyramidLevels)
    
    