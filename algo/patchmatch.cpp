#include "patchmatch.h"
#include <random>
#include <algorithm>

namespace PatchMatch
{
    static std::mt19937 g_generator(std::random_device{}());

    std::vector<cv::Point2i> initNNFRandom(int widthB, int heightB, int widthA, int heightA)
    {
        std::vector<cv::Point2i> nnf(widthB * heightB);
        std::uniform_int_distribution<> distX(0, widthA - 1);
        std::uniform_int_distribution<> distY(0, heightA - 1);

        for (int y = 0; y < heightB; ++y)
        {
            for (int x = 0; x < widthB; ++x)
            {
                nnf[y * widthB + x] = cv::Point2i(distX(g_generator), distY(g_generator));
            }
        }
        return nnf;
    }

    std::vector<float> initNNFDists(const cv::Mat &imageA,
                                    const cv::Mat &imageB,
                                    const std::vector<cv::Point2i> &nnf,
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
        int height = imageB.rows;
        bool isEven = iterNum % 2 == 0;
        
        if (patchPos.x < 0 || patchPos.x >= width || patchPos.y < 0 || patchPos.y >= height)
        {
            return;
        }
        
        int idx = patchPos.y * width + patchPos.x;
        
        if (idx < 0 || idx >= static_cast<int>(dists.size()))
        {
            return;
        }

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
                      float alpha)
    {
        int img_width = imageB.cols;
        int img_height = imageB.rows;
        
        if (patchPos.x < 0 || patchPos.x >= img_width || patchPos.y < 0 || patchPos.y >= img_height)
        {
            return;
        }
        
        int idx = patchPos.y * img_width + patchPos.x;
        
        if (idx < 0 || idx >= static_cast<int>(dists.size()))
        {
            return;
        }

        std::uniform_real_distribution<float> range(-1.0f, 1.0f);

        float current_radius = std::max(img_height, img_width);
        int i_param = 1;

        cv::Point2i currBestPoint = nnf[idx];
        float currBestDist = dists[idx];
        
        while (current_radius > 1.0f) {
            float randX = range(g_generator);
            float randY = range(g_generator);

            int candidateX = currBestPoint.x + (int)(current_radius * randX);
            int candidateY = currBestPoint.y + (int)(current_radius * randY);
            candidateX = std::max(0, std::min(img_width - 1, candidateX));
            candidateY = std::max(0, std::min(img_height - 1, candidateY));
            cv::Point2i candidatePos(candidateX, candidateY);

            float candidateDist = computePatchDistance(imageA, imageB, patchPos, candidatePos, patchSize);
            if (candidateDist < currBestDist) {
                currBestDist = candidateDist;
                currBestPoint = candidatePos;
            }

            current_radius *= std::pow(alpha, i_param);
            i_param += 1;
        }

        nnf[idx] = currBestPoint;
        dists[idx] = currBestDist;
    }
}