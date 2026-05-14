#ifndef GAUSSIAN_PYRAMIDS_H
#define GAUSSIAN_PYRAMIDS_H

#include "opencv2/core.hpp"
#include <vector>

namespace Util
{
    namespace GaussianPyramids
    {
        void buildPyramid(const cv::Mat &input, std::vector<cv::Mat> &pyramid, int levels = 5);
        void buildEmptyPyramid(const std::vector<cv::Mat> &reference, std::vector<cv::Mat> &dest);
    } // namespace GaussianPyramids
} // namespace Util

#endif // GAUSSIAN_PYRAMIDS_H