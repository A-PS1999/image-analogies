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
        
    return iPatch, jPatch, groupedPatches

def findCoherenceMatch(features: np.ndarray, pyramids: ImagePyramids, level: int, bPrimeLvl: np.ndarray, dimensions: int, i: int, j: int):
    M = bPrimeLvl.shape[0]
    N = bPrimeLvl.shape[1]
    area = dimensions * dimensions
    causalArea = area // 2
    
    halfPatch = (dimensions-1) // 2
    iPix = np.arange(max(i-halfPatch, 0), min(i-halfPatch+dimensions, M))
    jPix = np.arange(max(j-halfPatch, 0), min(j-halfPatch+dimensions, N))
    [neighborRows, neighborCols] = np.meshgrid(iPix, jPix, indexing='ij')
    neighborRows = neighborRows.flatten()
    neighborCols = neighborCols.flatten()
    
    sourceMapping = bPrimeLvl[neighborRows, neighborCols]
    candidateRows = neighborRows[sourceMapping[:, 0] > -1]
    candidateCols = neighborCols[sourceMapping[:, 0] > -1]
    sourceMapping = sourceMapping[sourceMapping[:, 0] > -1, :]
    
    candidateRows = sourceMapping[:, 0] - candidateRows
    candidateCols = sourceMapping[:, 1] - candidateCols
    
    paddedA = np.pad(pyramids.pyramidA[level], dimensions)
    paddedAPrime = np.pad(pyramids.pyramidAPrime[level], dimensions)
    
    unpaddedRows = paddedAPrime.shape[0] - 2*halfPatch
    unpaddedCols = paddedAPrime.shape[1] - 2*halfPatch
    
    isValid = (candidateRows >= 0)*(candidateRows < unpaddedRows)*(candidateCols >= 0)*(candidateCols < unpaddedCols)
    candidateRows = candidateRows[isValid]
    candidateCols = candidateCols[isValid]
    
    if candidateRows.size == 0:
        return [-1, -1], np.inf
    
    patchOffsets = np.arange(dimensions)
    patchI, patchJ = np.meshgrid(patchOffsets, patchOffsets, indexing='ij')
    patchI = patchI.flatten()
    patchJ = patchJ.flatten()
    candidatePatchRows = (candidateRows[:, None] + patchI[None, :])
    patchShape = candidatePatchRows.shape
    candidatePatchCols = (candidateCols[:, None] + patchJ[None, :])
    flatRows = candidatePatchRows.flatten()
    flatCols = candidatePatchCols.flatten()
    
    squareDists = np.zeros(flatRows.size)
    Y = np.reshape(paddedA[flatRows, flatCols], patchShape)
    X = features[0:area]
    squareDists += np.sum(X**2) + np.sum(Y**2, axis=1) - 2 * (Y.dot(X)).flatten()
    
    Y = np.reshape(paddedAPrime[flatRows, flatCols], patchShape)
    X = features[area:area+causalArea]
    squareDists += np.sum(X**2) + np.sum(Y**2, axis=1) - 2 * (Y.dot(X)).flatten()
    
    parentPyrA = np.pad(resizeImg(pyramids.pyramidA[level+1],  pyramids.pyramidA[level].shape), halfPatch)
    parentPyrAPrime = np.pad(resizeImg(pyramids.pyramidAPrime[level+1],  pyramids.pyramidAPrime[level].shape), halfPatch)
    
    if (parentPyrA.size > 0):
        Y = np.reshape(parentPyrA[flatRows, flatCols], patchShape)
        X = features[area+causalArea:area*2+causalArea]
        squareDists += np.sum(X**2) + np.sum(Y**2, axis=1) - 2 * (Y.dot(X)).flatten()
        
        Y = np.reshape(parentPyrAPrime[flatRows, flatCols], patchShape)
        X = features[area*2+causalArea:]
        squareDists += np.sum(X**2) + np.sum(Y**2, axis=1) - 2 * (Y.dot(X)).flatten()
        
    bestIdx = np.argmin(squareDists)
    return [candidateRows[bestIdx], candidateCols[bestIdx]], squareDists[bestIdx]

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
        
        iPatch, jPatch, preppedPatches = preparePatches(pyramids, level, currPatchSize, patchDimensions)
        
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
                distanceSquared = distance**2
                idx = int(idx[0][0])
                idx = [iPatch[idx], jPatch[idx]]
                
                if coherence > 0.0:
                    (coherentIdx, coherentDist) = findCoherenceMatch(features, pyramids, level, sourceMapping[level], currPatchSize, i, j)
                    factorial = 1 + coherence * (2.0 ** (level - pyramids.levels))
                    if (coherentDist < distanceSquared * factorial * factorial):
                        idx = coherentIdx
                
                sourceMapping[level][i, j, :] = idx
                pyramids.pyramidBPrime[level][i, j] = pyramids.pyramidAPrime[level][idx[0], idx[1]]
    
    return pyramids.pyramidBPrime[0]
        