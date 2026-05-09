#ifndef FEATURE_VECTOR_H
#define FEATURE_VECTOR_H

#include <vector>
#include "opencv2/core.hpp"

namespace ImageAnalogy
{
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