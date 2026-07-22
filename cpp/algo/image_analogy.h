#ifndef IMAGE_ANALOGY_H
#define IMAGE_ANALOGY_H

#include <string>
#include <vector>
#include <random>
#include "level_state.h"

namespace ImageAnalogy
{
    class ImageAnalogyMaker
    {
    public:
        ImageAnalogyMaker(const std::string& imageAPath,
                          const std::string& imageAPrimePath,
                          const std::string& imageBPath,
                          float coherenceWeight);
        cv::Mat generateAnalogy();

    private:
        std::vector<LevelState> levels;
        float coherenceWeight = 5.0f;
        int originalBWidth = 0, originalBHeight = 0;
        std::mt19937 rng{std::random_device{}()};

        cv::Point2i bestMatch(int currLvl, cv::Point2i currQ);
        cv::Point2i bestApproximateMatch(int currLvl, cv::Point2i currQ);
        cv::Point2i bestCoherenceMatch(int currLvl, cv::Point2i currQ);
        void computeApproximateMatchesForLevel(int currLvl);
        float featureDistance(int currLvl, cv::Point2i currQ, cv::Point2i comparisonP);
    };
}

#endif // IMAGE_ANALOGY_H