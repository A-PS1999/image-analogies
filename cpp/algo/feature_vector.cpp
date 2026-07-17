#include "feature_vector.h"
#include <algorithm>
#include "opencv2/imgproc.hpp"

namespace ImageAnalogy
{
    int FeatureVectorExtractor::extractWindow(const cv::Mat &image,
                                              int x,
                                              int y,
                                              int windowSize,
                                              size_t basePos,
                                              int pos,
                                              FeatureVector &outVec)
    {
        int windowRadius = windowSize / 2;

        for (int dy = -windowRadius; dy <= windowRadius; ++dy)
        {
            for (int dx = -windowRadius; dx <= windowRadius; ++dx)
            {
                int sampleX = std::clamp(x + dx, 0, image.cols - 1);
                int sampleY = std::clamp(y + dy, 0, image.rows - 1);

                cv::Vec3b pixel = image.at<cv::Vec3b>(sampleY, sampleX);
                outVec.features[basePos + pos++] = pixel[0] / 255.0f;            // L
                outVec.features[basePos + pos++] = (pixel[1] - 128.0f) / 256.0f; // a
                outVec.features[basePos + pos++] = (pixel[2] - 128.0f) / 256.0f; // b
            }
        }
        return pos;
    }

    void FeatureVectorExtractor::computeLevelFeatures(const cv::Mat &currLevel,
                                                      const cv::Mat &coarseLevel,
                                                      FeatureVector &featureVec)
    {
        int width = currLevel.cols;
        int height = currLevel.rows;
        featureVec.features.resize(width * height * VEC_SIZE, 0.0f);

        // Image converted to L*a*b* for feature vectors
        cv::Mat currLevelLab;
        cv::cvtColor(currLevel, currLevelLab, cv::COLOR_BGR2Lab);

        cv::Mat coarseLevelLab;
        if (coarseLevel.data)
        {
            cv::cvtColor(coarseLevel, coarseLevelLab, cv::COLOR_BGR2Lab);
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t basePos = (static_cast<size_t>(y) * width + x) * VEC_SIZE;
                int localPixPos = 0;

                localPixPos = extractWindow(currLevelLab, x, y, FINE_SIZE, basePos, localPixPos, featureVec);

                if (coarseLevelLab.data)
                {
                    extractWindow(coarseLevelLab, x / 2, y / 2, COARSE_SIZE, basePos, localPixPos, featureVec);
                }
            }
        }
    }

    void FeatureVectorExtractor::computeFullPyramidFeatures(const std::vector<cv::Mat> &pyramid,
                                                            std::vector<FeatureVector> &featureVector)
    {
        featureVector.resize(pyramid.size());

        for (size_t i = 0; i < pyramid.size(); ++i)
        {
            const cv::Mat &currLevel = pyramid[i];
            const cv::Mat *coarseLevel = (i < pyramid.size() - 1) ? &pyramid[i + 1] : nullptr;
            computeLevelFeatures(currLevel, coarseLevel ? *coarseLevel : cv::Mat(), featureVector[i]);
        }
    }
} // namespace ImageAnalogy