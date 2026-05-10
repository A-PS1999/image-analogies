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

    void propagate(int width,
                   cv::Point2i patchPos,
                   std::vector<cv::Point2i> &nnf,
                   std::vector<float> &dists,
                   int patchSize,
                   int iterNum)
    {
        bool isEven = iterNum % 2 == 0;

        if (!isEven)
        {
            auto leftDist = std::make_pair(dists[patchPos.y * width + std::max(patchPos.x - 1, 0)],
                                            Source::Left);
            auto upDist = std::make_pair(dists[std::max(patchPos.y - 1, 0) * width + patchPos.x],
                                          Source::Up);
            auto currDist = std::make_pair(dists[patchPos.y * width + patchPos.x], Source::Current);

            auto minPair = std::min({leftDist, upDist, currDist}, [](const auto &a, const auto &b){
                return a.first < b.first;
            });
            
            float minDist = minPair.first;
            Source minDistSrc = minPair.second;

            // Todo continue propagation logic
        }
        else
        {
        }
    }
}