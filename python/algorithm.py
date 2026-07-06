import numpy as np
from input_processing import ImagePyramids

def generateAnalogy(pyramids: ImagePyramids, coherence: float) -> np.ndarray:
    """
    Generates the output image B' based on the input data
    """
    
    sourceMapping = []
    
    for level in range(len(pyramids.pyramidB)):
        sourceMapping.append(-1 * np.ones(pyramids.pyramidB[level].shape[:2], dtype=int))
        
    for level in range(pyramids.levels, -1, -1):
        # TODO implement algorithm in Python
        pass
    
    return pyramids.pyramidBPrime[0]
        