#include "image_analogy.h"
#include "gaussian_pyramids.h"
#include "patchmatch.h"
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
                int linearPosQ = y * cols + x;

                cv::Point2i matchPoint = bestMatch(l, currQ);
                featureVectorsBPrime[l].features[linearPosQ] =
                    featureVectorsAPrime[l].features[matchPoint.y * cols + matchPoint.x];
                    sourcePixelMapping[linearPosQ] = matchPoint;
            }
        }

        return pyramidBPrime.front();
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

    cv::Point2i ImageAnalogyMaker::bestApproximateMatch(int currLvl, cv::Point2i currQ)
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

        // Store the best match for currQ in sourcePixelMapping: s(q) = p
        int linearPosQ = currQ.y * widthB + currQ.x;
        cv::Point2i bestMatch = nnf[linearPosQ];
        sourcePixelMapping[linearPosQ] = bestMatch;

        return bestMatch;
    }

    cv::Point2i ImageAnalogyMaker::bestCoherenceMatch(int currLvl, cv::Point2i currQ)
    {
        cv::Point2i rStar;
        float minDistance = std::numeric_limits<float>::max();

        int width = pyramidB[currLvl].cols;
        int height = pyramidB[currLvl].rows;
        int linearPosQ = currQ.y * width + currQ.x;

        for (int dy = -FINE_PATCH_SIZE; dy <= FINE_PATCH_SIZE; ++dy)
        {
            for (int dx = -FINE_PATCH_SIZE; dx <= FINE_PATCH_SIZE; ++dx)
            {
                int rX = currQ.x + dx;
                int rY = currQ.y + dy;

                if (rX < 0 || rX >= width || rY < 0 || rY >= height)
                {
                    continue;
                }

                int linearPosR = rY * width + rX;

                // Only consider pixels r that have already been synthesized (earlier in scan)
                if (linearPosR >= linearPosQ)
                {
                    continue;
                }

                cv::Point2i sourceR = sourcePixelMapping[linearPosR];

                // s(r) + (q - r)
                cv::Point2i candidateP = sourceR + (currQ - cv::Point2i(rX, rY));

                candidateP.x = std::clamp(candidateP.x, 0, width - 1);
                candidateP.y = std::clamp(candidateP.y, 0, height - 1);

                // ||F(s(r) + (q - r)) - F(q)||^2
                float distance = featureDistance(currLvl, currQ, candidateP);

                // Track the best (minimum distance) match
                if (distance < minDistance)
                {
                    minDistance = distance;
                    rStar = sourceR;
                }
            }
        }

        int linearPosRStar = rStar.y * width + rStar.x;

        return sourcePixelMapping[linearPosRStar] + (currQ - rStar);
    }

    float ImageAnalogyMaker::featureDistance(int currLvl, cv::Point2i currQ, cv::Point2i comparisonP)
    {
        const int VEC_SIZE = (FINE_PATCH_SIZE * FINE_PATCH_SIZE * 4) + (COARSE_PATCH_SIZE * COARSE_PATCH_SIZE * 4);

        int width = pyramidB[currLvl].cols;

        size_t qIndex = (static_cast<size_t>(currQ.y) * width + currQ.x) * VEC_SIZE;
        size_t pIndex = (static_cast<size_t>(comparisonP.y) * width + comparisonP.x) * VEC_SIZE;

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