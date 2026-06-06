#include "image_analogy.h"
#include "gaussian_pyramids.h"
#include "patchmatch.h"
#include "opencv2/imgcodecs.hpp"
#include <algorithm>
#include <iostream>

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

        this->originalBWidth = imageB.cols;
        this->originalBHeight = imageB.rows;

        std::vector<cv::Mat> pyramidA, pyramidAPrime, pyramidB, pyramidBPrime;
        Util::GaussianPyramids::buildPyramid(imageA, pyramidA);
        Util::GaussianPyramids::buildPyramid(imageAPrime, pyramidAPrime);
        Util::GaussianPyramids::buildPyramid(imageB, pyramidB);
        Util::GaussianPyramids::buildEmptyPyramid(pyramidB, pyramidBPrime);

        this->pyramidA = pyramidA;
        this->pyramidAPrime = pyramidAPrime;
        this->pyramidB = pyramidB;
        this->pyramidBPrime = pyramidBPrime;
        this->coherenceWeight = coherenceWeight;
        this->sourcePixelMapping.resize(pyramidB[0].cols * pyramidB[0].rows);
        this->levelNNF.resize(pyramidB.size());
        this->levelNNFDists.resize(pyramidB.size());

        FeatureVectorExtractor::computeFullPyramidFeatures(pyramidA, featureVectorsA);
        FeatureVectorExtractor::computeFullPyramidFeatures(pyramidAPrime, featureVectorsAPrime);
        FeatureVectorExtractor::computeFullPyramidFeatures(pyramidB, featureVectorsB);
        featureVectorsBPrime = featureVectorsB;
    }

    cv::Mat ImageAnalogyMaker::generateAnalogy()
    {
        int numLevels = pyramidB.size();

        for (int l = numLevels - 1; l >= 0; --l)
        {
            computeApproximateMatchesForLevel(l);

            int lvlLength = pyramidBPrime[l].cols * pyramidBPrime[l].rows;
            int colsB = pyramidBPrime[l].cols;
            int colsA = pyramidA[l].cols;

            for (int q = 0; q < lvlLength; ++q)
            {
                int x = q % colsB;
                int y = q / colsB;
                cv::Point2i currQ(x, y);

                cv::Point2i matchPoint = bestMatch(l, currQ);
                featureVectorsBPrime[l].features[q] =
                    featureVectorsAPrime[l].features[matchPoint.y * colsA + matchPoint.x];
                sourcePixelMapping[q] = matchPoint;
                
                cv::Vec3b pixelValue = pyramidAPrime[l].at<cv::Vec3b>(matchPoint.y, matchPoint.x);
                pyramidBPrime[l].at<cv::Vec3b>(y, x) = pixelValue;
            }
        }

        cv::Mat result = pyramidBPrime.front();
        if (result.cols != originalBWidth || result.rows != originalBHeight)
        {
            cv::Rect cropRect(0, 0, originalBWidth, originalBHeight);
            result = result(cropRect).clone();
        }

        return result;
    }

    cv::Point2i ImageAnalogyMaker::bestMatch(int currLvl, cv::Point2i currQ)
    {
        int maxLevels = pyramidB.size();
        float cohWeight = coherenceWeight;
        cv::Point2i bestApproxMatch = bestApproximateMatch(currLvl, currQ);
        cv::Point2i bestCohMatch = bestCoherenceMatch(currLvl, currQ);

        float distApprox = featureDistance(currLvl, currQ, bestApproxMatch);
        float distCoherence = featureDistance(currLvl, currQ, bestCohMatch);

        float factorial = 1 + std::pow(2, (currLvl - maxLevels)) * cohWeight;

        if (distCoherence <= distApprox * factorial)
        {
            return bestCohMatch;
        }
        else
        {
            return bestApproxMatch;
        }
    }

    void ImageAnalogyMaker::computeApproximateMatchesForLevel(int currLvl)
    {
        int widthA = pyramidA[currLvl].cols;
        int heightA = pyramidA[currLvl].rows;
        int widthB = pyramidB[currLvl].cols;
        int heightB = pyramidB[currLvl].rows;

        std::vector<cv::Point2i> nnf = PatchMatch::initNNFRandom(widthB, heightB, widthA, heightA);
        std::vector<float> dists = PatchMatch::initNNFDists(pyramidB[currLvl], pyramidA[currLvl], nnf, FINE_PATCH_SIZE);

        int numIterations = 5;
        for (int iter = 0; iter < numIterations; ++iter)
        {
            // Propagation phase
            for (int y = 0; y < heightB; ++y)
            {
                for (int x = 0; x < widthB; ++x)
                {
                    PatchMatch::propagate(pyramidB[currLvl], pyramidA[currLvl],
                                          cv::Point2i(x, y), nnf, dists, FINE_PATCH_SIZE, iter);
                }
            }

            // Random search phase
            for (int y = 0; y < heightB; ++y)
            {
                for (int x = 0; x < widthB; ++x)
                {
                    PatchMatch::randomSearch(pyramidB[currLvl], pyramidA[currLvl],
                                             cv::Point2i(x, y), nnf, dists, FINE_PATCH_SIZE);
                }
            }
        }

        // Cache the computed NNF and distances for this level
        levelNNF[currLvl] = nnf;
        levelNNFDists[currLvl] = dists;
    }

    cv::Point2i ImageAnalogyMaker::bestApproximateMatch(int currLvl, cv::Point2i currQ)
    {
        int widthB = pyramidB[currLvl].cols;
        int linearPosQ = currQ.y * widthB + currQ.x;
        return levelNNF[currLvl][linearPosQ];
    }

    cv::Point2i ImageAnalogyMaker::bestCoherenceMatch(int currLvl, cv::Point2i currQ)
    {
        cv::Point2i rStar;
        float minDistance = std::numeric_limits<float>::max();

        int widthB = pyramidB[currLvl].cols;
        int heightB = pyramidB[currLvl].rows;
        int widthA = pyramidA[currLvl].cols;
        int heightA = pyramidA[currLvl].rows;
        int linearPosQ = currQ.y * widthB + currQ.x;

        for (int dy = -FINE_PATCH_SIZE; dy <= FINE_PATCH_SIZE; ++dy)
        {
            for (int dx = -FINE_PATCH_SIZE; dx <= FINE_PATCH_SIZE; ++dx)
            {
                int rX = currQ.x + dx;
                int rY = currQ.y + dy;

                if (rX < 0 || rX >= widthB || rY < 0 || rY >= heightB)
                {
                    continue;
                }

                int linearPosR = rY * widthB + rX;

                // Skip pixels r that are not before q in scan order
                if (linearPosR >= linearPosQ)
                {
                    continue;
                }

                cv::Point2i sourceR = sourcePixelMapping[linearPosR];

                // s(r) + (q - r)
                cv::Point2i candidateP = sourceR + (currQ - cv::Point2i(rX, rY));

                candidateP.x = std::clamp(candidateP.x, 0, widthA - 1);
                candidateP.y = std::clamp(candidateP.y, 0, heightA - 1);

                // ||F(s(r) + (q - r)) - F(q)||^2
                float distance = featureDistance(currLvl, currQ, candidateP);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    rStar = candidateP;
                }
            }
        }

        return rStar;
    }

    float ImageAnalogyMaker::featureDistance(int currLvl, cv::Point2i currQ, cv::Point2i comparisonP)
    {
        const int VEC_SIZE = (FINE_PATCH_SIZE * FINE_PATCH_SIZE * 4) + (COARSE_PATCH_SIZE * COARSE_PATCH_SIZE * 4);

        int widthB = pyramidB[currLvl].cols;
        int widthA = pyramidA[currLvl].cols;

        size_t qIndex = (static_cast<size_t>(currQ.y) * widthB + currQ.x) * VEC_SIZE;
        size_t pIndex = (static_cast<size_t>(comparisonP.y) * widthA + comparisonP.x) * VEC_SIZE;

        const float *pFeaturesA = &featureVectorsA[currLvl].features[pIndex];
        const float *pFeaturesAPrime = &featureVectorsAPrime[currLvl].features[pIndex];
        const float *qFeaturesB = &featureVectorsB[currLvl].features[qIndex];
        const float *qFeaturesBPrime = &featureVectorsBPrime[currLvl].features[qIndex];

        float distanceSquared = 0.0f;

        for (int i = 0; i < VEC_SIZE; ++i)
        {
            float diffAB = pFeaturesA[i] - qFeaturesB[i];
            distanceSquared += diffAB * diffAB;

            float diffAPrimeBPrime = pFeaturesAPrime[i] - qFeaturesBPrime[i];
            distanceSquared += diffAPrimeBPrime * diffAPrimeBPrime;
        }

        return distanceSquared;
    }
}