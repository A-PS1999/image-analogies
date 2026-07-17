#ifndef FEATURE_VECTOR_H
#define FEATURE_VECTOR_H

#include <vector>
#include "opencv2/core.hpp"

namespace ImageAnalogy
{
    const int FINE_SIZE = 5;
    const int COARSE_SIZE = 3;
    const int NUM_CHANNELS = 3;
    const int VEC_SIZE = (FINE_SIZE * FINE_SIZE * NUM_CHANNELS) +
                                (COARSE_SIZE * COARSE_SIZE * NUM_CHANNELS);
    struct FeatureVector
    {
        std::vector<float> features;
    };

    class FeatureVectorExtractor
    {
        public:
            static void computeFullPyramidFeatures(const std::vector<cv::Mat> &pyramid,
                                            std::vector<FeatureVector> &featureVector);
            static void computeLevelFeatures(const cv::Mat &currLevel,
                                    const cv::Mat &coarseLevel,
                                    FeatureVector &featureVec);
            static void computeSingleFeature();
            static int extractWindow(const cv::Mat &image,
                            int x,
                            int y,
                            int windowSize,
                            size_t basePos,
                            int pos,
                            FeatureVector &outVec);
    };

} // namespace ImageAnalogy

#endif // FEATURE_VECTOR_H