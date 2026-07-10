import numpy as np
from pynndescent import NNDescent
import skimage.transform
from input_processing import ImagePyramids

COARSE_PATCH_SIZE = 3
FINE_PATCH_SIZE = 5

resizeImg = lambda img, shape: skimage.transform.resize(img, shape, anti_aliasing=True, mode='constant')

def makePatches(image: np.ndarray, dimensions: int, iArr: np.ndarray, jArr: np.ndarray) -> np.ndarray:
    numPatches = iArr.size
    pixelRange = np.arange(dimensions)
    iPix, jPix = np.meshgrid(pixelRange, pixelRange, indexing='ij')
    iPix = iPix.flatten()
    jPix = jPix.flatten()
    iPix = (iArr[:, None] + iPix[None, :]).flatten()
    jPix = (jArr[:, None] + jPix[None, :]).flatten()
    patches = image[iPix, jPix, ...]
    
    numChannels = 1
    if len(image.shape) > 2:
        numChannels = image.shape[2]
    
    return np.reshape(patches, (numPatches, dimensions*dimensions*numChannels))

def fetchCausalPatches(image: np.ndarray, dimensions: int, iArr: np.ndarray, jArr: np.ndarray) -> np.ndarray:
    patches = makePatches(image, dimensions, iArr, jArr)
    causalArea = (dimensions*dimensions) // 2
    if len(image.shape) > 2:
        causalArea *= image.shape[2]
    
    patches = patches[:, 0:causalArea]
    return patches

def preparePatches(pyramids: ImagePyramids,
                   level: int, 
                   patchSize: int, 
                   dimensions: int) -> np.ndarray:
    initialShape = pyramids.pyramidA[level].shape
    iPatch, jPatch = np.meshgrid(np.arange(initialShape[0]), np.arange(initialShape[1]), indexing='ij')
    iPatch = iPatch.flatten()
    jPatch = jPatch.flatten()
    
    paddedA = np.pad(pyramids.pyramidA[level], dimensions)
    paddedAPrime = np.pad(pyramids.pyramidAPrime[level], dimensions)
    
    patchesA = makePatches(paddedA, patchSize, iPatch, jPatch)
    patchesAPrime = fetchCausalPatches(paddedAPrime, patchSize, iPatch, jPatch)
    
    groupedPatches = np.concatenate((patchesA, patchesAPrime), 1)
    
    nextLevelA = np.array([])
    nextLevelAPrime = np.array([])
    
    if level < pyramids.levels:
        nextLevelA = np.pad(resizeImg(pyramids.pyramidA[level+1], pyramids.pyramidA[level].shape), dimensions)
        nextLevelAPrime = np.pad(resizeImg(pyramids.pyramidAPrime[level+1], pyramids.pyramidAPrime[level].shape), dimensions)
        nextAPatches = makePatches(nextLevelA, patchSize, iPatch, jPatch)
        nextAPrimePatches = makePatches(nextLevelAPrime, patchSize, iPatch, jPatch)
        groupedPatches = np.concatenate((groupedPatches, nextAPatches, nextAPrimePatches), 1)
        
    return groupedPatches
    

def generateAnalogy(pyramids: ImagePyramids, coherence: float) -> np.ndarray:
    """
    Generates the output image B' based on the input data
    """
    
    sourceMapping = []
    
    for level in range(len(pyramids.pyramidB)):
        sourceMapping.append(-1 * np.ones(pyramids.pyramidB[level].shape[:2], dtype=int))
        
    for level in range(pyramids.levels, -1, -1):
        currPatchSize = FINE_PATCH_SIZE
        if (level == pyramids.levels):
            currPatchSize = COARSE_PATCH_SIZE
        
        patchDimensions = (currPatchSize-1) // 2
        patchArea = currPatchSize * currPatchSize
        causalArea = patchArea // 2
        combinedArea = patchArea + causalArea
        
        if level < pyramids.levels:
            combinedArea += patchArea * 2
        
        preppedPatches = preparePatches(pyramids, level, currPatchSize, patchDimensions)
        
        nearestNeighbours = NNDescent(preppedPatches)
        features = np.zeros(combinedArea)
        
        paddedB = np.pad(pyramids.pyramidB[level], patchDimensions)
        paddedBPrime = np.pad(pyramids.pyramidBPrime[level], patchDimensions)
        if level < pyramids.levels:
            nextLevelB = np.pad(resizeImg(pyramids.pyramidB[level+1], pyramids.pyramidB[level].shape), patchDimensions)
            nextLevelBPrime = np.pad(resizeImg(pyramids.pyramidBPrime[level+1], pyramids.pyramidBPrime[level].shape), patchDimensions)
        
        for i in range(pyramids.pyramidBPrime[level].shape[0]):
            for j in range(pyramids.pyramidBPrime[level].shape[1]):
                features[0:patchArea] = paddedB[i:i+currPatchSize, j:j+currPatchSize].flatten()
                features[patchArea:patchArea+causalArea] = paddedBPrime[i:i+currPatchSize, j:j+currPatchSize].flatten()[0:causalArea]
                
                if level < pyramids.levels:
                    features[patchArea+causalArea:patchArea*2+causalArea] = nextLevelB[i:i+currPatchSize, j:j+currPatchSize].flatten()
                    features[patchArea*2+causalArea:] = nextLevelBPrime[i:i+currPatchSize, j:j+currPatchSize].flatten()
                    
                idx, distance = nearestNeighbours.query(nearestNeighbours, features)
                # TODO continue algorithm
                    
                
    
    return pyramids.pyramidBPrime[0]
        