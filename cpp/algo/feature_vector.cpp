#include "feature_vector.h"
#include "level_state.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include "opencv2/imgproc.hpp"

namespace ImageAnalogy
{
    // Gaussian weight table generation
    namespace
    {
        template <int WindowSize>
        static constexpr int BlockSize = WindowSize * WindowSize * NUM_CHANNELS;

        template <int WindowSize>
        static std::array<float, BlockSize<WindowSize>> computeGaussianWeights()
        {
            std::array<float, BlockSize<WindowSize>> weights{};
            constexpr int radius = WindowSize / 2;
            const float sigma = WindowSize / 3.0f;
            const float twoSigmaSq = 2.0f * sigma * sigma;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    float g = std::exp(-(dx * dx + dy * dy) / twoSigmaSq);
                    int pixelIdx = (dy + radius) * WindowSize + (dx + radius);
                    for (int c = 0; c < NUM_CHANNELS; ++c)
                    {
                        weights[pixelIdx * NUM_CHANNELS + c] = g;
                    }
                }
            }
            return weights;
        }
    }

    extern const std::array<float, FINE_BLOCK_SIZE> FINE_WEIGHTS = computeGaussianWeights<FINE_SIZE>();
    extern const std::array<float, COARSE_BLOCK_SIZE> COARSE_WEIGHTS = computeGaussianWeights<COARSE_SIZE>();
    const float FINE_WEIGHT_SUM   = std::accumulate(FINE_WEIGHTS.begin(),   FINE_WEIGHTS.end(),   0.0f);
    const float COARSE_WEIGHT_SUM = std::accumulate(COARSE_WEIGHTS.begin(), COARSE_WEIGHTS.end(), 0.0f);

    // Window & feature extraction
    int FeatureVectorExtractor::extractWindow(const cv::Mat &image,
                                              int x, int y, int windowSize,
                                              size_t basePos, int pos,
                                              FeatureVector &outVec)
    {
        int windowRadius = windowSize / 2;

        for (int dy = -windowRadius; dy <= windowRadius; ++dy)
        {
            for (int dx = -windowRadius; dx <= windowRadius; ++dx)
            {
                int sampleX = std::clamp(x + dx, 0, image.cols - 1);
                int sampleY = std::clamp(y + dy, 0, image.rows - 1);

                cv::Vec3b pixel = image.at<cv::Vec3b>(sampleY, sampleX);
                outVec.features[basePos + pos++] = pixel[0] / 255.0f; // B
                outVec.features[basePos + pos++] = pixel[1] / 255.0f; // G
                outVec.features[basePos + pos++] = pixel[2] / 255.0f; // R
            }
        }
        return pos;
    }

    static cv::Mat &selectImage(LevelState &lvlState, FeatureSelector which)
    {
        switch (which)
        {
        case FeatureSelector::FEAT_A:
            return lvlState.A;
        case FeatureSelector::FEAT_A_PRIME:
            return lvlState.APrime;
        case FeatureSelector::FEAT_B:
            return lvlState.B;
        case FeatureSelector::FEAT_B_PRIME:
            return lvlState.BPrime;
        }
        return lvlState.A; // unreachable
    }

    static FeatureVector &selectFeatures(LevelState &lvlState, FeatureSelector which)
    {
        switch (which)
        {
        case FeatureSelector::FEAT_A:
            return lvlState.featA;
        case FeatureSelector::FEAT_A_PRIME:
            return lvlState.featAPrime;
        case FeatureSelector::FEAT_B:
            return lvlState.featB;
        case FeatureSelector::FEAT_B_PRIME:
            return lvlState.featBPrime;
        }
        return lvlState.featA; // unreachable
    }

    void FeatureVectorExtractor::computeFullPyramidFeatures(std::vector<LevelState> &levels,
                                                            FeatureSelector which)
    {
        for (size_t i = 0; i < levels.size(); ++i)
        {
            cv::Mat &currLevel = selectImage(levels[i], which);
            FeatureVector &featureVec = selectFeatures(levels[i], which);

            int width = currLevel.cols;
            int height = currLevel.rows;
            featureVec.features.resize(width * height * VEC_SIZE, 0.0f);

            bool hasCoarser = (i + 1 < levels.size());

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    size_t basePos = (static_cast<size_t>(y) * width + x) * VEC_SIZE;
                    int localPixPos = 0;

                    localPixPos = extractWindow(currLevel, x, y, FINE_SIZE, basePos, localPixPos, featureVec);

                    if (hasCoarser)
                    {
                        cv::Mat &coarseLevel = selectImage(levels[i + 1], which);
                        extractWindow(coarseLevel, x / 2, y / 2, COARSE_SIZE, basePos, localPixPos, featureVec);
                    }
                }
            }
        }
    }

    void FeatureVectorExtractor::computeLevelFeatures(std::vector<LevelState> &levels,
                                                      int level, FeatureSelector which)
    {
        cv::Mat &currLevel = selectImage(levels[level], which);
        FeatureVector &featureVec = selectFeatures(levels[level], which);

        int width = currLevel.cols;
        int height = currLevel.rows;
        featureVec.features.resize(width * height * VEC_SIZE, 0.0f);

        bool hasCoarser = (level + 1 < static_cast<int>(levels.size()));

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t basePos = (static_cast<size_t>(y) * width + x) * VEC_SIZE;
                int localPixPos = 0;

                localPixPos = extractWindow(currLevel, x, y, FINE_SIZE, basePos, localPixPos, featureVec);

                if (hasCoarser)
                {
                    cv::Mat &coarseLevel = selectImage(levels[level + 1], which);
                    extractWindow(coarseLevel, x / 2, y / 2, COARSE_SIZE, basePos, localPixPos, featureVec);
                }
            }
        }
    }

    // Specific to B'
    void FeatureVectorExtractor::precomputeCoarseFeatures(std::vector<LevelState> &levels, int level)
    {
        int width = levels[level].BPrime.cols;
        int height = levels[level].BPrime.rows;

        bool hasCoarser = (level + 1 < static_cast<int>(levels.size()));

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t basePos = (static_cast<size_t>(y) * width + x) * VEC_SIZE;
                if (hasCoarser)
                {
                    extractWindow(levels[level + 1].BPrime, x / 2, y / 2, COARSE_SIZE,
                                  basePos, FINE_BLOCK_SIZE, levels[level].featBPrime);
                }
                else
                {
                    std::fill_n(&levels[level].featBPrime.features[basePos + FINE_BLOCK_SIZE],
                                COARSE_BLOCK_SIZE, 0.0f);
                }
            }
        }
    }

    // Specific to B'
    void FeatureVectorExtractor::recomputeFineFeatures(std::vector<LevelState> &levels,
                                                       int level, int x, int y)
    {
        const cv::Mat &fineImg = levels[level].BPrime;
        int width = fineImg.cols;

        size_t basePos = (static_cast<size_t>(y) * width + x) * VEC_SIZE;
        extractWindow(fineImg, x, y, FINE_SIZE, basePos, 0, levels[level].featBPrime);
    }

    // Specific to B'
    void FeatureVectorExtractor::recomputeCausalNeighbours(std::vector<LevelState> &levels,
                                                           int level, int x, int y)
    {
        const std::vector<std::pair<int, int>> directions = {
            {0, -1}, {-1, -1}, {-1, 0}, {1, -1}};

        int width = levels[level].BPrime.cols;
        int height = levels[level].BPrime.rows;

        size_t linearPosQ = static_cast<size_t>(y) * width + x;

        for (const auto &[dx, dy] : directions)
        {
            int newX = x + dx;
            int newY = y + dy;

            if (newX < 0 || newY < 0 || newX >= width || newY >= height)
                continue;

            size_t linearPosR = static_cast<size_t>(newY) * width + newX;
            if (linearPosR >= linearPosQ)
                continue;

            recomputeFineFeatures(levels, level, newX, newY);
        }
    }
}