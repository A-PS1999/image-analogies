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
    std::vector<cv::Point2i> upsampleNNF(std::vector<cv::Point2i> &nnf,
                                         cv::Size coarseSize,
                                         cv::Size currASize,
                                         cv::Size currBSize);
    // SSD between the patch centered at posA in imageA and posB in imageB.
    // If `currentBestDist` is finite, the sum aborts early as soon as it
    // exceeds that value (returning +inf); pass FLT_MAX (the default) for a
    // full compute.
    float computePatchDistance(const cv::Mat &imageA,
                               const cv::Mat &imageB,
                               cv::Point2i posA,
                               cv::Point2i posB,
                               int patchSize,
                               float currentBestDist = std::numeric_limits<float>::max());
    void propagate(const cv::Mat &imageA,
                   const cv::Mat &imageB,
                   cv::Point2i patchPos,
                   std::vector<cv::Point2i> &nnf,
                   std::vector<float> &dists,
                   int patchSize,
                   bool isEven);
    void randomSearch(const cv::Mat &imageA,
                      const cv::Mat &imageB,
                      cv::Point2i patchPos,
                      std::vector<cv::Point2i> &nnf,
                      std::vector<float> &dists,
                      int patchSize,
                      float alpha = 0.5);
} // namespace PatchMatch

#endif // PATCHMATCH_H