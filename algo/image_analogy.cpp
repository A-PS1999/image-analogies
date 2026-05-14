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
        cv::Mat imageA = cv::imread(imageAPath, cv::IMREAD_COLOR);
        cv::Mat imageAPrime = cv::imread(imageAPrimePath, cv::IMREAD_COLOR);
        cv::Mat imageB = cv::imread(imageBPath, cv::IMREAD_COLOR);

        if (imageA.empty() || imageAPrime.empty() || imageB.empty())
        {
            throw std::runtime_error("One or more input images could not be loaded. Please check file paths.");
        }

        std::vector<cv::Mat> pyramidA, pyramidAPrime, pyramidB, pyramidBPrime;
        Util::GaussianPyramids::buildPyramid(imageA, pyramidA);
        Util::GaussianPyramids::buildPyramid(imageAPrime, pyramidAPrime);
        Util::GaussianPyramids::buildPyramid(imageB, pyramidB);
        Util::GaussianPyramids::buildEmptyPyramid(pyramidB, pyramidBPrime);

        this->pyramidA = pyramidA;
        this->pyramidAPrime = pyramidAPrime;
        this->pyramidB = pyramidB;
        this->coherenceWeight = coherenceWeight;
        this->sourcePixelMapping.resize(imageB.cols * imageB.rows);

        FeatureVectorExtractor::computeFullPyramidFeatures(pyramidA, featureVectorsA);
        FeatureVectorExtractor::computeFullPyramidFeatures(pyramidAPrime, featureVectorsAPrime);
        FeatureVectorExtractor::computeFullPyramidFeatures(pyramidB, featureVectorsB);
    }

    cv::Mat ImageAnalogyMaker::generateAnalogy()
    {
        int numLevels = pyramidB.size();

        for (int l = numLevels; l >= 0; --l)
        {
            int lvlLength = pyramidBPrime[l].cols * pyramidBPrime[l].rows;

            for (int q = 0; q < lvlLength; ++q)
            {
                int cols = pyramidBPrime[l].cols;
                int x = q % cols;
                int y = q / cols;
                cv::Point2i currQ(x, y);
                
                bestMatch(l, currQ);
            }
        }
    }

    cv::Point2i bestMatch(int currLvl, int maxLevels, float cohWeight, cv::Point2i currQ)
    {
        cv::Point2i bestApproxMatch = bestApproximateMatch(currLvl, currQ);
        cv::Point2i bestCohMatch = bestCoherenceMatch(currLvl, currQ);

        float distApprox = featureDistance(currQ, bestApproxMatch);
        float distCoherence = featureDistance(currQ, bestCohMatch);

        float factorial = 1 + std::pow(2, (currLvl - maxLevels)) * cohWeight;

        if (distCoherence <= distApprox * factorial) {
            return bestCohMatch;
        } else {
            return bestApproxMatch;
        }
    }

    cv::Point2i bestApproximateMatch(int currLvl, cv::Point2i currQ)
    {
        // TODO implement approximage match function logic using PatchMatch
    }

    cv::Point2i bestCoherenceMatch(int currLvl, cv::Point2i currQ)
    {
        // TODO implement coherence match logic
    }

    float featureDistance(cv::Point2i currQ, cv::Point2i comparisonP) {
        // Implement feature vector distance logic
    }
}