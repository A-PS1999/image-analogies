#ifndef FEATURE_VECTOR_H
#define FEATURE_VECTOR_H

#include <vector>
#include "opencv2/core.hpp"

namespace ImageAnalogy
{
    const int FINE_SIZE = 5;
    const int COARSE_SIZE = 3;
    const int NUM_CHANNELS = 3;
    const int FINE_BLOCK_SIZE = (FINE_SIZE * FINE_SIZE * NUM_CHANNELS);
    const int COARSE_BLOCK_SIZE = (COARSE_SIZE * COARSE_SIZE * NUM_CHANNELS);
    const int VEC_SIZE = FINE_BLOCK_SIZE + COARSE_BLOCK_SIZE;

    enum class FeatureSelector { FEAT_A, FEAT_A_PRIME, FEAT_B, FEAT_B_PRIME };

    struct FeatureVector
    {
        std::vector<float> features;
    };

    // Forward declaration — LevelState is defined in level_state.h
    struct LevelState;

    class FeatureVectorExtractor
    {
    public:
        static void computeFullPyramidFeatures(std::vector<LevelState>& levels,
                                               FeatureSelector which);
        static void computeLevelFeatures(std::vector<LevelState>& levels,
                                        int level, FeatureSelector which);
        static void precomputeCoarseFeatures(std::vector<LevelState>& levels, int level);
        static void recomputeFineFeatures(std::vector<LevelState>& levels,
                                          int level, int x, int y);
        static void recomputeCausalNeighbours(std::vector<LevelState>& levels,
                                              int level, int x, int y);
        static int extractWindow(const cv::Mat& image,
                                 int x, int y, int windowSize,
                                 size_t basePos, int pos, FeatureVector& outVec);
    };
}

#endif // FEATURE_VECTOR_H