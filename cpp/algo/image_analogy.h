#ifndef IMAGE_ANALOGY_H
#define IMAGE_ANALOGY_H

#include "feature_vector.h"

namespace ImageAnalogy
{
    enum PyramidOption
    {
        PYRAMID_A,
        PYRAMID_A_PRIME,
        PYRAMID_B,
        PYRAMID_B_PRIME
    };

    class ImageAnalogyMaker
    {
    public:
        ImageAnalogyMaker(const std::string &imageAPath,
                          const std::string &imageAPrimePath,
                          const std::string &imageBPath,
                          const float coherenceWeight);
        cv::Mat generateAnalogy();

    private:
        float coherenceWeight = 5.0;
        std::vector<cv::Mat> pyramidA, pyramidAPrime, pyramidB, pyramidBPrime, pyramidBPrimeLab;
        std::vector<cv::Mat> pyramidAPrimeLab;
        std::vector<FeatureVector> featureVectorsA, featureVectorsAPrime, featureVectorsB, featureVectorsBPrime;
        std::vector<std::vector<cv::Point2i>> sourcePixelMapping;
        std::vector<std::vector<cv::Point2i>> levelNNF;
        std::vector<std::vector<float>> levelNNFDists;
        int originalBWidth = 0, originalBHeight = 0;
        cv::Point2i bestMatch(int currLvl, cv::Point2i currQ);
        cv::Point2i bestApproximateMatch(int currLvl, cv::Point2i currQ);
        cv::Point2i bestCoherenceMatch(int currLvl, cv::Point2i currQ);
        void computeApproximateMatchesForLevel(int currLvl);
        float featureDistance(int currLvl, cv::Point2i currQ, cv::Point2i comparisonP);
    };

}

#endif // IMAGE_ANALOGY_H