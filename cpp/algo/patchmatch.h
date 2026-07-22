#ifndef PATCHMATCH_H
#define PATCHMATCH_H

#include <vector>
#include <random>
#include <limits>
#include "opencv2/core.hpp"

namespace PatchMatch
{
    struct NNF
    {
        int width = 0, height = 0;
        std::vector<cv::Point2i> offsets;
        std::vector<float> dists;

        void initRandom(int widthA, int heightA, std::mt19937& rng);
        void initDists(const cv::Mat& imageA, const cv::Mat& imageB, int patchSize);
        void upsampleFrom(const NNF& coarse, cv::Size currASize);
        void propagate(const cv::Mat& imageA, const cv::Mat& imageB,
                       cv::Point2i patchPos, int patchSize, bool isEven);
        void randomSearch(const cv::Mat& imageA, const cv::Mat& imageB,
                          cv::Point2i patchPos, int patchSize, float alpha, std::mt19937& rng);
    };

    float computePatchDistance(const cv::Mat& imageA,
                               const cv::Mat& imageB,
                               cv::Point2i posA,
                               cv::Point2i posB,
                               int patchSize,
                               float currentBestDist = std::numeric_limits<float>::max());
}

#endif // PATCHMATCH_H