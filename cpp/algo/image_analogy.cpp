#include "image_analogy.h"
#include "gaussian_pyramids.h"
#include "patchmatch.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <algorithm>
#include <iostream>

namespace ImageAnalogy
{
    ImageAnalogyMaker::ImageAnalogyMaker(const std::string& imageAPath,
                                         const std::string& imageAPrimePath,
                                         const std::string& imageBPath,
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

        int numLevels = pyramidB.size();
        levels.resize(numLevels);

        for (int i = 0; i < numLevels; ++i)
        {
            levels[i].A = pyramidA[i];
            levels[i].APrime = pyramidAPrime[i];
            levels[i].B = pyramidB[i];
            levels[i].BPrime = pyramidBPrime[i];

            levels[i].BPrimeLab.create(pyramidBPrime[i].size(), pyramidBPrime[i].type());
            levels[i].BPrimeLab.setTo(cv::Scalar::all(0));

            cv::cvtColor(levels[i].APrime, levels[i].APrimeLab, cv::COLOR_BGR2Lab);

            levels[i].sourceMap.assign(pyramidB[i].cols * pyramidB[i].rows, cv::Point2i(-1, -1));
        }

        this->coherenceWeight = coherenceWeight;

        FeatureVectorExtractor::computeFullPyramidFeatures(levels, FeatureSelector::FEAT_A);
        FeatureVectorExtractor::computeFullPyramidFeatures(levels, FeatureSelector::FEAT_A_PRIME);
        FeatureVectorExtractor::computeFullPyramidFeatures(levels, FeatureSelector::FEAT_B);
        for (int i = 0; i < numLevels; ++i)
        {
            levels[i].featBPrime = levels[i].featB;
        }
    }

    cv::Mat ImageAnalogyMaker::generateAnalogy()
    {
        int numLevels = levels.size();

        std::vector<cv::Mat> bPrimeMats(numLevels);
        std::vector<cv::Mat> bPrimeLabMats(numLevels);
        for (int i = 0; i < numLevels; ++i)
        {
            bPrimeMats[i] = levels[i].BPrime;
            bPrimeLabMats[i] = levels[i].BPrimeLab;
        }

        for (int l = numLevels - 1; l >= 0; --l)
        {
            computeApproximateMatchesForLevel(l);
            if (l < numLevels - 1)
            {
                Util::GaussianPyramids::upsamplePyramid(l, bPrimeMats, bPrimeLabMats);
                levels[l].BPrime = bPrimeMats[l];
                levels[l].BPrimeLab = bPrimeLabMats[l];
                FeatureVectorExtractor::computeLevelFeatures(levels, l, FeatureSelector::FEAT_B_PRIME);
            }
            FeatureVectorExtractor::precomputeCoarseFeatures(levels, l);

            int lvlLength = levels[l].BPrime.cols * levels[l].BPrime.rows;
            int colsB = levels[l].BPrime.cols;
            int colsA = levels[l].A.cols;

            for (int q = 0; q < lvlLength; ++q)
            {
                int x = q % colsB;
                int y = q / colsB;
                cv::Point2i currQ(x, y);

                cv::Point2i matchPoint = bestMatch(l, currQ);

                if (q < static_cast<int>(levels[l].sourceMap.size()))
                {
                    levels[l].sourceMap[q] = matchPoint;
                }

                cv::Vec3b pixelValue = levels[l].APrime.at<cv::Vec3b>(matchPoint.y, matchPoint.x);
                levels[l].BPrime.at<cv::Vec3b>(y, x) = pixelValue;

                levels[l].BPrimeLab.at<cv::Vec3b>(y, x) =
                    levels[l].APrimeLab.at<cv::Vec3b>(matchPoint.y, matchPoint.x);

                FeatureVectorExtractor::recomputeFineFeatures(levels, l, x, y);
                FeatureVectorExtractor::recomputeCausalNeighbours(levels, l, x, y);
            }
        }

        cv::Mat result = levels[0].BPrime;
        if (result.cols != originalBWidth || result.rows != originalBHeight)
        {
            cv::Rect cropRect(0, 0, originalBWidth, originalBHeight);
            result = result(cropRect).clone();
        }

        return result;
    }

    cv::Point2i ImageAnalogyMaker::bestMatch(int currLvl, cv::Point2i currQ)
    {
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
        int widthA = levels[currLvl].A.cols;
        int heightA = levels[currLvl].A.rows;
        int widthB = levels[currLvl].B.cols;
        int heightB = levels[currLvl].B.rows;

        PatchMatch::NNF nnf;
        nnf.width = widthB;
        nnf.height = heightB;
        nnf.offsets.resize(widthB * heightB);
        nnf.dists.resize(widthB * heightB);

        if (currLvl < static_cast<int>(levels.size()) - 1 && !levels[currLvl + 1].nnf.offsets.empty())
        {
            nnf.upsampleFrom(levels[currLvl + 1].nnf, levels[currLvl].A.size());
        }
        else
        {
            nnf.initRandom(widthA, heightA, rng);
        }
        nnf.initDists(levels[currLvl].A, levels[currLvl].B, FINE_SIZE);

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
                        nnf.propagate(levels[currLvl].A, levels[currLvl].B,
                                      cv::Point2i(x, y), FINE_SIZE, isEven);
                        nnf.randomSearch(levels[currLvl].A, levels[currLvl].B,
                                         cv::Point2i(x, y), FINE_SIZE, 0.5f, rng);
                    }
                }
            }
            else
            {
                for (int y = heightB - 1; y >= 0; --y)
                {
                    for (int x = widthB - 1; x >= 0; --x)
                    {
                        nnf.propagate(levels[currLvl].A, levels[currLvl].B,
                                      cv::Point2i(x, y), FINE_SIZE, isEven);
                        nnf.randomSearch(levels[currLvl].A, levels[currLvl].B,
                                         cv::Point2i(x, y), FINE_SIZE, 0.5f, rng);
                    }
                }
            }
        }

        levels[currLvl].nnf = std::move(nnf);
    }

    cv::Point2i ImageAnalogyMaker::bestApproximateMatch(int currLvl, cv::Point2i currQ)
    {
        int widthB = levels[currLvl].B.cols;
        size_t linearPosQ = static_cast<size_t>(currQ.y) * static_cast<size_t>(widthB) + static_cast<size_t>(currQ.x);

        if (linearPosQ >= levels[currLvl].nnf.offsets.size())
        {
            return cv::Point2i(0, 0);
        }

        return levels[currLvl].nnf.offsets[linearPosQ];
    }

    cv::Point2i ImageAnalogyMaker::bestCoherenceMatch(int currLvl, cv::Point2i currQ)
    {
        cv::Point2i rStar;
        float minDistance = std::numeric_limits<float>::max();

        int widthB = levels[currLvl].B.cols;
        int heightB = levels[currLvl].B.rows;
        int widthA = levels[currLvl].A.cols;
        int heightA = levels[currLvl].A.rows;
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

                if (linearPosR >= linearPosQ)
                {
                    continue;
                }

                if (linearPosR >= levels[currLvl].sourceMap.size())
                {
                    continue;
                }

                cv::Point2i sourceR = levels[currLvl].sourceMap[linearPosR];

                cv::Point2i candidateP = sourceR + (currQ - cv::Point2i(rX, rY));

                candidateP.x = std::clamp(candidateP.x, 0, widthA - 1);
                candidateP.y = std::clamp(candidateP.y, 0, heightA - 1);

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
        int widthB = levels[currLvl].B.cols;
        int widthA = levels[currLvl].A.cols;

        size_t qIndex = (static_cast<size_t>(currQ.y) * static_cast<size_t>(widthB) + static_cast<size_t>(currQ.x)) * VEC_SIZE;
        size_t pIndex = (static_cast<size_t>(comparisonP.y) * static_cast<size_t>(widthA) + static_cast<size_t>(comparisonP.x)) * VEC_SIZE;

        if (qIndex >= levels[currLvl].featB.features.size() ||
            pIndex >= levels[currLvl].featA.features.size())
        {
            return std::numeric_limits<float>::max();
        }

        const float* pFeaturesA = &levels[currLvl].featA.features[pIndex];
        const float* pFeaturesAPrime = &levels[currLvl].featAPrime.features[pIndex];
        const float* qFeaturesB = &levels[currLvl].featB.features[qIndex];
        const float* qFeaturesBPrime = &levels[currLvl].featBPrime.features[qIndex];

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