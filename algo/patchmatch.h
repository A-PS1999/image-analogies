#ifndef PATCHMATCH_H
#define PATCHMATCH_H

#include <vector>
#include "opencv2/core.hpp"

namespace PatchMatch
{
    std::vector<cv::Point2i> initNNFRandom(int widthA, int heightA, int widthB, int heightB);
    std::vector<float> initNNFDists(const cv::Mat &imageA,
                                    const cv::Mat &imageB,
                                    const std::vector<cv::Point2i> &nnf,
                                    int patchSize);
    float computePatchDistance(const cv::Mat &imageA,
                               const cv::Mat &imageB,
                               cv::Point2i posA,
                               cv::Point2i posB,
                               int patchSize);
    void propagate(int width,
                   cv::Point2i patchPos,
                   std::vector<cv::Point2i> &nnf,
                   std::vector<float> &dists,
                   int patchSize,
                   int iterNum);
    void randomSearch(cv::Mat &imageA,
                      cv::Mat &imageB,
                      std::vector<cv::Point2i> &nnf,
                      std::vector<float> &dists,
                      int searchRadius,
                      int patchSize);
} // namespace PatchMatch

#endif // PATCHMATCH_H