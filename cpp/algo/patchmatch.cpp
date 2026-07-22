#include "patchmatch.h"
#include <random>
#include <algorithm>

namespace PatchMatch
{
    static std::mt19937 g_generator(std::random_device{}());

    std::vector<cv::Point2i> initNNFRandom(int widthA, int heightA, int widthB, int heightB)
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
                size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                if (idx >= dists.size())
                {
                    continue; // Bounds check to prevent overflow access
                }
                dists[idx] = computePatchDistance(imageA, imageB, nnf[idx], cv::Point2i(x, y), patchSize);
            }
        }
        return dists;
    }

    std::vector<cv::Point2i> upsampleNNF(std::vector<cv::Point2i> &nnf,
                                         cv::Size coarseSize,
                                         cv::Size currASize,
                                         cv::Size currBSize)
    {
        std::vector<cv::Point2i> upsampledNNF(currBSize.width * currBSize.height);

        for (int y = 0; y < currBSize.height; ++y)
        {
            for (int x = 0; x < currBSize.width; ++x)
            {
                int coarseX = std::min(x / 2, coarseSize.width - 1);
                int coarseY = std::min(y / 2, coarseSize.height - 1);
                cv::Point2i coarseOffset = nnf[coarseY * coarseSize.width + coarseX];

                int fineX = coarseOffset.x * 2 + (x % 2);
                int fineY = coarseOffset.y * 2 + (y % 2);
                fineX = std::clamp(fineX, 0, currASize.width - 1);
                fineY = std::clamp(fineY, 0, currASize.height - 1);

                upsampledNNF[y * currBSize.width + x] = cv::Point2i(fineX, fineY);
            }
        }

        return upsampledNNF;
    }

    float computePatchDistance(const cv::Mat &imageA,
                               const cv::Mat &imageB,
                               cv::Point2i posA,
                               cv::Point2i posB,
                               int patchSize,
                               float currentBestDist)
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

                float db0 = pixelB[0] - pixelA[0];
                float db1 = pixelB[1] - pixelA[1];
                float db2 = pixelB[2] - pixelA[2];

                dist += db0 * db0 + db1 * db1 + db2 * db2;
                if (dist >= currentBestDist)
                {
                    return std::numeric_limits<float>::max();
                }
            }
        }
        return dist;
    }

    static bool tryPropagateFromNeighbor(
        const cv::Mat &imageA,
        const cv::Mat &imageB,
        cv::Point2i patchPos,
        cv::Point2i neighborPos,
        std::vector<cv::Point2i> &nnf,
        std::vector<float> &dists,
        int patchSize)
    {
        int width = imageB.cols;
        int neighborIdx = neighborPos.y * width + neighborPos.x;
        int idx = patchPos.y * width + patchPos.x;

        if (neighborIdx < 0 || neighborIdx >= static_cast<int>(nnf.size()))
            return false;

        cv::Point2i candidateMatch = nnf[neighborIdx] + (patchPos - neighborPos);
        float newDist = computePatchDistance(imageA, imageB, candidateMatch, patchPos, patchSize, dists[idx]);
        if (newDist < dists[idx])
        {
            nnf[idx] = candidateMatch;
            dists[idx] = newDist;
            return true;
        }
        return false;
    }

    void propagate(const cv::Mat &imageA,
                   const cv::Mat &imageB,
                   cv::Point2i patchPos,
                   std::vector<cv::Point2i> &nnf,
                   std::vector<float> &dists,
                   int patchSize,
                   bool isEven)
    {
        int width = imageB.cols;
        int height = imageB.rows;

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
            if (patchPos.x > 0) {
                tryPropagateFromNeighbor(imageA, imageB, patchPos, {patchPos.x - 1, patchPos.y}, nnf, dists, patchSize);
            }
            if (patchPos.y > 0) {
                tryPropagateFromNeighbor(imageA, imageB, patchPos, {patchPos.x, patchPos.y - 1}, nnf, dists, patchSize);
            }
        }
        else
        {
            if (patchPos.x < width - 1) {
                tryPropagateFromNeighbor(imageA, imageB, patchPos, {patchPos.x + 1, patchPos.y}, nnf, dists, patchSize);
            }
            if (patchPos.y < height - 1) {
                tryPropagateFromNeighbor(imageA, imageB, patchPos, {patchPos.x, patchPos.y + 1}, nnf, dists, patchSize);
            }
        }
    }

    void randomSearch(const cv::Mat &imageA,
                      const cv::Mat &imageB,
                      cv::Point2i patchPos,
                      std::vector<cv::Point2i> &nnf,
                      std::vector<float> &dists,
                      int patchSize,
                      float alpha)
    {
        int img_width = imageA.cols;
        int img_height = imageA.rows;

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

        cv::Point2i currBestPoint = nnf[idx];
        float currBestDist = dists[idx];

        while (current_radius > 1.0f)
        {
            float randX = range(g_generator);
            float randY = range(g_generator);

            int candidateX = currBestPoint.x + (int)(current_radius * randX);
            int candidateY = currBestPoint.y + (int)(current_radius * randY);
            candidateX = std::max(0, std::min(img_width - 1, candidateX));
            candidateY = std::max(0, std::min(img_height - 1, candidateY));
            cv::Point2i candidatePos(candidateX, candidateY);

            float candidateDist = computePatchDistance(imageA, imageB, candidatePos, patchPos, patchSize, currBestDist);
            if (candidateDist < currBestDist)
            {
                currBestDist = candidateDist;
                currBestPoint = candidatePos;
            }

            current_radius *= alpha;
        }

        nnf[idx] = currBestPoint;
        dists[idx] = currBestDist;
    }
}