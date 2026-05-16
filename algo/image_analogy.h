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
        const int FINE_PATCH_SIZE = 5;
        const int COARSE_PATCH_SIZE = 3;
        ImageAnalogyMaker(const std::string &imageAPath,
                          const std::string &imageAPrimePath,
                          const std::string &imageBPath,
                          const float coherenceWeight);
        cv::Mat generateAnalogy();

    private:
        float coherenceWeight = 5.0;
        std::vector<cv::Mat> pyramidA, pyramidAPrime, pyramidB, pyramidBPrime;
        std::vector<FeatureVector> featureVectorsA, featureVectorsAPrime, featureVectorsB, featureVectorsBPrime;
        std::vector<cv::Point2i> sourcePixelMapping;
        std::vector<cv::Point2i> nearestNeighbourField;
        std::vector<float> nearestNeighbourDistances;
        cv::Point2i bestMatch(int currLvl, cv::Point2i currQ);
        cv::Point2i bestApproximateMatch(int currLvl, cv::Point2i currQ);
        cv::Point2i bestCoherenceMatch(int currLvl, cv::Point2i currQ);
        float featureDistance(int currLvl, cv::Point2i currQ, cv::Point2i comparisonP);
    };

}

#endif // IMAGE_ANALOGY_H