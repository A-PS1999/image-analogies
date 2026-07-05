import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageA", type=str, required=True, help="Path to input image A")
    parser.add_argument("--imageAPrime", type=str, required=True, help="Path to input image A'")
    parser.add_argument("--imageB", type=str, required=True, help="Path to input image B")
    parser.add_argument("--coherence", type=float, default=5.0, help="Coherence parameter for the algorithm")
    
    inputs = parser.parse_args()
    
    