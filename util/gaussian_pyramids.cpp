#include "gaussian_pyramids.h"
#include "opencv2/imgproc.hpp"

namespace Util
{
    namespace GaussianPyramids
    {
        void buildPyramid(const cv::Mat &input, std::vector<cv::Mat> &pyramid, int levels)
        {
            int divisibilityFactor = 1 << levels;

            if (input.cols % divisibilityFactor == 0 && input.rows % divisibilityFactor == 0)
            {
                // First we push back the original image before downsampling
                pyramid.push_back(input);

                for (int i = 0; i < levels; i++)
                {
                    cv::Mat downsampled;
                    cv::pyrDown(pyramid.back(), downsampled);
                    pyramid.push_back(downsampled);
                }
            }
            else
            {
                throw std::runtime_error("Input image dimensions must be divisible by " + std::to_string(divisibilityFactor) +
                                         " for the specified number of pyramid levels (" + std::to_string(levels) + ").");
            }
        }
    } // namespace GaussianPyramids
} // namespace Util