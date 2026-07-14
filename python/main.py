import imageio.v2 as imageio
import input_processing
import algorithm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageA", type=str, required=True, help="Path to input image A")
    parser.add_argument("--imageAPrime", type=str, required=True, help="Path to input image A'")
    parser.add_argument("--imageB", type=str, required=True, help="Path to input image B")
    parser.add_argument("--pyrLevels", type=int, default=5, required=False, help="Number of pyramid levels to use")
    parser.add_argument("--coherence", type=float, default=5.0, required=False, help="Coherence parameter for the algorithm")
    
    inputs = parser.parse_args()
    originalImgA = imageio.imread(inputs.imageA)
    originalImgAPrime = imageio.imread(inputs.imageAPrime)
    originalImgB = imageio.imread(inputs.imageB)
    coherence = inputs.coherence
    pyramidLevels = inputs.pyrLevels
    
    pyramids = input_processing.inputsToPyramids(
        [originalImgA, originalImgAPrime, originalImgB], 
        pyramidLevels
    )
    
    outputImgBPrime = algorithm.generateAnalogy(pyramids, coherence)
    
    imageio.imwrite("outputBPrime.png", outputImgBPrime)