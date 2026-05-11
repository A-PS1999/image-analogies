#include "patchmatch.h"
#include <random>

namespace PatchMatch
{
    std::vector<cv::Point2i> initNNFRandom(int widthA, int heightA, int widthB, int heightB)
    {
        std::vector<cv::Point2i> nnf(widthB * heightB);
        std::random_device randDevice;
        std::mt19937 generator(randDevice());
        std::uniform_int_distribution<> distX(0, widthB - 1);
        std::uniform_int_distribution<> distY(0, heightB - 1);

        for (int y = 0; y < heightA; ++y)
        {
            for (int x = 0; x < widthA; ++x)
            {
                nnf[y * widthA + x] = cv::Point2i(distX(generator), distY(generator));
            }
        }
        return nnf;
    }

    std::vector<float> initNNFDists(const cv::Mat &imageA,
                                    const cv::Mat &imageB,
                                    const std::vector<cv::Vec2i> &nnf,
                                    int patchSize)
    {
        int width = imageB.cols;
        int height = imageB.rows;
        std::vector<float> dists(width * height, std::numeric_limits<float>::max());
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = y * width + x;
                dists[idx] = computePatchDistance(imageA, imageB, cv::Point2i(x, y), nnf[idx], patchSize);
            }
        }
        return dists;
    }

    float computePatchDistance(const cv::Mat &imageA,
                               const cv::Mat &imageB,
                               cv::Point2i posA,
                               cv::Point2i posB,
                               int patchSize)
    {
        float dist = 0.0f;
        int patchRadius = patchSize / 2;

        for (int dy = -patchRadius; dy <= patchRadius; ++dy)
        {
            for (int dx = -patchRadius; dx <= patchRadius; ++dx)
            {
                int sampleAX = std::clamp(posA.x + dx, 0, imageA.cols - 1);
                int sampleAY = std::clamp(posA.y + dy, 0, imageA.rows - 1);
                int sampleBX = std::clamp(posB.x + dx, 0, imageB.cols - 1);
                int sampleBY = std::clamp(posB.y + dy, 0, imageB.rows - 1);

                cv::Vec3b pixelA = imageA.at<cv::Vec3b>(sampleAY, sampleAX);
                cv::Vec3b pixelB = imageB.at<cv::Vec3b>(sampleBY, sampleBX);

                dist += std::pow(pixelB[0] - pixelA[0], 2) + std::pow(pixelB[1] - pixelA[1], 2) +
                        std::pow(pixelB[2] - pixelA[2], 2);
            }
        }
        return dist;
    }

    enum class Source
    {
        Left,
        Up,
        Current,
        Right,
        Down
    };

    void propagate(const cv::Mat &imageA,
                   const cv::Mat &imageB,
                   cv::Point2i patchPos,
                   std::vector<cv::Point2i> &nnf,
                   std::vector<float> &dists,
                   int patchSize,
                   int iterNum)
    {
        int width = imageB.cols;
        bool isEven = iterNum % 2 == 0;
        int idx = patchPos.y * width + patchPos.x;

        if (!isEven)
        {
            std::vector<std::pair<float, Source>> candidates;

            candidates.push_back({dists[idx], Source::Current});

            if (patchPos.x > 0)
            {
                candidates.push_back({dists[patchPos.y * width + (patchPos.x - 1)], Source::Left});
            }

            if (patchPos.y > 0)
            {
                candidates.push_back({dists[(patchPos.y - 1) * width + patchPos.x], Source::Up});
            }

            auto minPair = *std::min_element(candidates.begin(), candidates.end(),
                                             [](const auto &a, const auto &b)
                                             { return a.first < b.first; });

            Source minDistSrc = minPair.second;

            if (minDistSrc == Source::Left)
            {
                cv::Point2i candidateMatch = nnf[patchPos.y * width + (patchPos.x - 1)];
                float newDist = computePatchDistance(imageA, imageB, patchPos, candidateMatch, patchSize);
                if (newDist < dists[idx])
                {
                    nnf[idx] = candidateMatch;
                    dists[idx] = newDist;
                }
            }
            else if (minDistSrc == Source::Up)
            {
                cv::Point2i candidateMatch = nnf[(patchPos.y - 1) * width + patchPos.x];
                float newDist = computePatchDistance(imageA, imageB, patchPos, candidateMatch, patchSize);
                if (newDist < dists[idx])
                {
                    nnf[idx] = candidateMatch;
                    dists[idx] = newDist;
                }
            }
        }
        else
        {
            std::vector<std::pair<float, Source>> candidates;

            candidates.push_back({dists[idx], Source::Current});

            if (patchPos.x < imageB.cols - 1)
            {
                candidates.push_back({dists[patchPos.y * width + (patchPos.x + 1)], Source::Right});
            }

            if (patchPos.y < imageB.rows - 1)
            {
                candidates.push_back({dists[(patchPos.y + 1) * width + patchPos.x], Source::Down});
            }

            auto minPair = *std::min_element(candidates.begin(), candidates.end(),
                                             [](const auto &a, const auto &b)
                                             { return a.first < b.first; });

            Source minDistSrc = minPair.second;

            if (minDistSrc == Source::Right)
            {
                cv::Point2i candidateMatch = nnf[patchPos.y * width + (patchPos.x + 1)];
                float newDist = computePatchDistance(imageA, imageB, patchPos, candidateMatch, patchSize);
                if (newDist < dists[idx])
                {
                    nnf[idx] = candidateMatch;
                    dists[idx] = newDist;
                }
            }
            else if (minDistSrc == Source::Down)
            {
                cv::Point2i candidateMatch = nnf[(patchPos.y + 1) * width + patchPos.x];
                float newDist = computePatchDistance(imageA, imageB, patchPos, candidateMatch, patchSize);
                if (newDist < dists[idx])
                {
                    nnf[idx] = candidateMatch;
                    dists[idx] = newDist;
                }
            }
        }
    }

    void randomSearch(cv::Mat &imageA,
                      cv::Mat &imageB,
                      cv::Point2i patchPos,
                      std::vector<cv::Point2i> &nnf,
                      std::vector<float> &dists,
                      int patchSize,
                      float alpha = 0.5)
    {
        int img_width = imageB.cols;
        int img_height = imageB.rows;

        int searchRadius = patchSize / 2;
        int i_param = 4;
        int search_height = img_height * std::pow(alpha, i_param);
        int search_width = img_width * std::pow(alpha, i_param);

        cv::Point2i currPoint = nnf[patchPos.y * img_width + patchPos.x];
        // TODO continue random search fun implementation
    }
}