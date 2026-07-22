#ifndef LEVEL_STATE_H
#define LEVEL_STATE_H

#include <vector>
#include "opencv2/core.hpp"
#include "patchmatch.h"
#include "feature_vector.h"

namespace ImageAnalogy
{
    struct LevelState
    {
        cv::Mat A, APrime, B, BPrime, BPrimeLab, APrimeLab;
        FeatureVector featA, featAPrime, featB, featBPrime;
        std::vector<cv::Point2i> sourceMap;
        PatchMatch::NNF nnf;
    };
}

#endif // LEVEL_STATE_H
