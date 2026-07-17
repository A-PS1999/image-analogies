#include "gaussian_pyramids.h"
#include "opencv2/imgproc.hpp"

namespace Util
{
    namespace GaussianPyramids
    {
        int getPaddedSize(int originalSize, int divisibilityFactor)
        {
            int remainder = originalSize % divisibilityFactor;
            if (remainder == 0)
                return originalSize;
            return originalSize + (divisibilityFactor - remainder);
        }

        void buildPyramid(const cv::Mat &input, std::vector<cv::Mat> &pyramid, int levels)
        {
            int divisibilityFactor = 1 << levels;

            int paddedWidth = getPaddedSize(input.cols, divisibilityFactor);
            int paddedHeight = getPaddedSize(input.rows, divisibilityFactor);

            cv::Mat paddedInput = input.clone();
            if (paddedWidth != input.cols || paddedHeight != input.rows)
            {
                int padRight = paddedWidth - input.cols;
                int padBottom = paddedHeight - input.rows;
                cv::copyMakeBorder(input, paddedInput, 0, padBottom, 0, padRight, cv::BORDER_REPLICATE);
            }

            // First we push back the padded image before downsampling
            pyramid.push_back(paddedInput);

            for (int i = 0; i < levels; i++)
            {
                cv::Mat downsampled;
                cv::pyrDown(pyramid.back(), downsampled);
                pyramid.push_back(downsampled);
            }
        }

        void initSourceMappingPyr(std::vector<std::vector<cv::Point2i>> &sourceMap, const std::vector<cv::Mat> &sourcePyr) {
            int numLevels = sourcePyr.size();
            cv::Point2i defaultPoint(-1, -1);
            
            for (int l = 0; l < numLevels; ++l) {
                sourceMap[l] = std::vector<cv::Point2i>(sourcePyr[l].cols * sourcePyr[l].rows, defaultPoint);
            }
        }

        void buildEmptyPyramid(const std::vector<cv::Mat> &reference, std::vector<cv::Mat> &dest)
        {
            int numLevels = reference.size();
            
            for (int l = 0; l < numLevels; ++l)
            {
                cv::Mat newLevel;
                cv::Mat currLvlRef = reference[l];
                newLevel.zeros(currLvlRef.size(), currLvlRef.type());
                dest.push_back(newLevel);
            }
        }
    } // namespace GaussianPyramids
} // namespace Util