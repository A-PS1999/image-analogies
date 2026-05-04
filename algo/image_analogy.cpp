#include "image_analogy.h"
#include "gaussian_pyramids.h"
#include "opencv2/imgcodecs.hpp"

namespace ImageAnalogy
{
    ImageAnalogyMaker::ImageAnalogyMaker(const std::string &imageAPath,
                                         const std::string &imageAPrimePath,
                                         const std::string &imageBPath,
                                         const float coherenceWeight)
    {
        cv::Mat imageA = cv::imread(imageAPath);
        cv::Mat imageAPrime = cv::imread(imageAPrimePath);
        cv::Mat imageB = cv::imread(imageBPath);

        if (imageA.empty() || imageAPrime.empty() || imageB.empty())
        {
            throw std::runtime_error("One or more input images could not be loaded. Please check file paths.");
        }

        std::vector<cv::Mat> pyramidA, pyramidAPrime, pyramidB;
        Util::GaussianPyramids::buildPyramid(imageA, pyramidA);
        Util::GaussianPyramids::buildPyramid(imageAPrime, pyramidAPrime);
        Util::GaussianPyramids::buildPyramid(imageB, pyramidB);

        this->pyramidA = pyramidA;
        this->pyramidAPrime = pyramidAPrime;
        this->pyramidB = pyramidB;
        this->coherenceWeight = coherenceWeight;
    }

    cv::Mat ImageAnalogyMaker::generateAnalogy()
    {
        return cv::Mat(); // Placeholder for now
    }
}