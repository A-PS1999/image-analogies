#include "image_analogy.h"
#include "gaussian_pyramids.h"
#include "patchmatch.h"
#include "opencv2/imgproc.hpp"
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
        Util::GaussianPyramids::initSourceMappingPyr(sourcePixelMapping, pyramidB);
        Util::GaussianPyramids::buildEmptyPyramid(pyramidB, pyramidBPrime);
        pyramidBPrimeLab.resize(pyramidBPrime.size());
        for (int i = 0; i < pyramidBPrime.size(); ++i)
        {
            pyramidBPrimeLab[i].create(pyramidBPrime[i].size(), pyramidBPrime[i].type());
            pyramidBPrimeLab[i].setTo(cv::Scalar::all(0));
        }

        pyramidAPrimeLab.resize(pyramidAPrime.size());
        for (int i = 0; i < pyramidAPrime.size(); ++i)
        {
            cv::cvtColor(pyramidAPrime[i], pyramidAPrimeLab[i], cv::COLOR_BGR2Lab);
        }

        this->pyramidA = pyramidA;
        this->pyramidAPrime = pyramidAPrime;
        this->pyramidB = pyramidB;
        this->pyramidBPrime = pyramidBPrime;
        this->coherenceWeight = coherenceWeight;
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
            if (l < numLevels - 1)
            {
                Util::GaussianPyramids::upsamplePyramid(l,
                                                        pyramidBPrime,
                                                        pyramidBPrimeLab);
                FeatureVectorExtractor::computeLevelFeatures(pyramidBPrime[l],
                                                             pyramidBPrime[l + 1],
                                                             featureVectorsBPrime[l]);
            }
            FeatureVectorExtractor::precomputeCoarseFeatures(l,
                                                             pyramidBPrime,
                                                             featureVectorsBPrime);

            int lvlLength = pyramidBPrime[l].cols * pyramidBPrime[l].rows;
            int colsB = pyramidBPrime[l].cols;
            int colsA = pyramidA[l].cols;

            for (int q = 0; q < lvlLength; ++q)
            {
                int x = q % colsB;
                int y = q / colsB;
                cv::Point2i currQ(x, y);

                cv::Point2i matchPoint = bestMatch(l, currQ);

                if (q < static_cast<int>(sourcePixelMapping[l].size()))
                {
                    sourcePixelMapping[l][q] = matchPoint;
                }

                cv::Vec3b pixelValue = pyramidAPrime[l].at<cv::Vec3b>(matchPoint.y, matchPoint.x);
                pyramidBPrime[l].at<cv::Vec3b>(y, x) = pixelValue;

                pyramidBPrimeLab[l].at<cv::Vec3b>(y, x) =
                    pyramidAPrimeLab[l].at<cv::Vec3b>(matchPoint.y, matchPoint.x);

                FeatureVectorExtractor::recomputeFineFeatures(l,
                                                              x,
                                                              y,
                                                              pyramidBPrimeLab,
                                                              featureVectorsBPrime);
                FeatureVectorExtractor::recomputeCausalNeighbours(l,
                                                                  x,
                                                                  y,
                                                                  pyramidBPrimeLab,
                                                                  featureVectorsBPrime);
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

        float factorial = 1 + std::pow(2, -currLvl) * cohWeight;

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

        std::vector<cv::Point2i> nnf;
        if (currLvl < pyramidA.size() - 1 && !levelNNF[currLvl + 1].empty())
        {
            nnf = PatchMatch::upsampleNNF(levelNNF[currLvl + 1],
                                          pyramidB[currLvl + 1].size(),
                                          pyramidA[currLvl].size(),
                                          pyramidB[currLvl].size());
        }
        else
        {
            nnf = PatchMatch::initNNFRandom(widthA, heightA, widthB, heightB);
        }
        std::vector<float> dists = PatchMatch::initNNFDists(pyramidA[currLvl], pyramidB[currLvl], nnf, FINE_SIZE);

        int numIterations = 5;
        for (int iter = 0; iter < numIterations; ++iter)
        {
            bool isEven = iter % 2 == 0;

            if (!isEven)
            {
                for (int y = 0; y < heightB; ++y)
                {
                    for (int x = 0; x < widthB; ++x)
                    {
                        PatchMatch::propagate(pyramidA[currLvl], pyramidB[currLvl],
                                              cv::Point2i(x, y), nnf, dists, FINE_SIZE, isEven);
                        PatchMatch::randomSearch(pyramidA[currLvl], pyramidB[currLvl],
                                                 cv::Point2i(x, y), nnf, dists, FINE_SIZE);
                    }
                }
            }
            else
            {
                for (int y = heightB - 1; y >= 0; --y)
                {
                    for (int x = widthB - 1; x >= 0; --x)
                    {
                        PatchMatch::propagate(pyramidA[currLvl], pyramidB[currLvl],
                                              cv::Point2i(x, y), nnf, dists, FINE_SIZE, isEven);
                        PatchMatch::randomSearch(pyramidA[currLvl], pyramidB[currLvl],
                                                 cv::Point2i(x, y), nnf, dists, FINE_SIZE);
                    }
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
        size_t linearPosQ = static_cast<size_t>(currQ.y) * static_cast<size_t>(widthB) + static_cast<size_t>(currQ.x);

        if (linearPosQ >= levelNNF[currLvl].size())
        {
            return cv::Point2i(0, 0);
        }

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
        size_t linearPosQ = static_cast<size_t>(currQ.y) * static_cast<size_t>(widthB) + static_cast<size_t>(currQ.x);

        int searchRadius = (FINE_SIZE - 1) / 2;

        for (int dy = -searchRadius; dy <= searchRadius; ++dy)
        {
            for (int dx = -searchRadius; dx <= searchRadius; ++dx)
            {
                int rX = currQ.x + dx;
                int rY = currQ.y + dy;

                if (rX < 0 || rX >= widthB || rY < 0 || rY >= heightB)
                {
                    continue;
                }

                size_t linearPosR = static_cast<size_t>(rY) * static_cast<size_t>(widthB) + static_cast<size_t>(rX);

                // Skip pixels r that are not before q in scan order
                if (linearPosR >= linearPosQ)
                {
                    continue;
                }

                if (linearPosR >= sourcePixelMapping[currLvl].size())
                {
                    continue;
                }

                cv::Point2i sourceR = sourcePixelMapping[currLvl][linearPosR];

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
        int widthB = pyramidB[currLvl].cols;
        int widthA = pyramidA[currLvl].cols;

        size_t qIndex = (static_cast<size_t>(currQ.y) * static_cast<size_t>(widthB) + static_cast<size_t>(currQ.x)) * VEC_SIZE;
        size_t pIndex = (static_cast<size_t>(comparisonP.y) * static_cast<size_t>(widthA) + static_cast<size_t>(comparisonP.x)) * VEC_SIZE;

        if (qIndex >= featureVectorsB[currLvl].features.size() ||
            pIndex >= featureVectorsA[currLvl].features.size())
        {
            return std::numeric_limits<float>::max(); // Return max distance if out of bounds
        }

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