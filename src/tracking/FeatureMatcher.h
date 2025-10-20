#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <memory>

namespace VirtualAd {
namespace Tracking {

/**
 * @brief Matcher type for descriptor matching
 */
enum class MatcherType {
    BRUTE_FORCE,      // Brute force matcher (for binary descriptors)
    FLANN             // FLANN-based matcher (faster, for float descriptors)
};

/**
 * @brief Feature matcher for frame-to-frame tracking
 * 
 * This class performs descriptor matching between two frames,
 * estimates homography transformation using RANSAC, and provides
 * robust feature correspondence for camera tracking.
 */
class FeatureMatcher {
public:
    /**
     * @brief Construct a new Feature Matcher
     * @param matcherType Type of matcher to use
     * @param ratioThreshold Lowe's ratio test threshold (default: 0.75)
     * @param ransacThreshold RANSAC reprojection threshold in pixels (default: 3.0)
     */
    explicit FeatureMatcher(MatcherType matcherType = MatcherType::BRUTE_FORCE,
                           float ratioThreshold = 0.75f,
                           float ransacThreshold = 3.0f);

    /**
     * @brief Destroy the Feature Matcher
     */
    ~FeatureMatcher() = default;

    /**
     * @brief Match features between two frames
     * @param keypoints1 Keypoints from first frame
     * @param descriptors1 Descriptors from first frame
     * @param keypoints2 Keypoints from second frame
     * @param descriptors2 Descriptors from second frame
     * @param matches Output matches (after ratio test)
     * @return true if matching succeeded
     */
    bool matchFeatures(const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& descriptors1,
                      const std::vector<cv::KeyPoint>& keypoints2,
                      const cv::Mat& descriptors2,
                      std::vector<cv::DMatch>& matches);

    /**
     * @brief Estimate homography between matched features using RANSAC
     * @param keypoints1 Keypoints from first frame
     * @param keypoints2 Keypoints from second frame
     * @param matches Input matches
     * @param homography Output homography matrix (3x3)
     * @param inlierMask Output mask indicating inliers (1) and outliers (0)
     * @return true if homography was successfully estimated
     */
    bool estimateHomography(const std::vector<cv::KeyPoint>& keypoints1,
                           const std::vector<cv::KeyPoint>& keypoints2,
                           const std::vector<cv::DMatch>& matches,
                           cv::Mat& homography,
                           std::vector<uchar>& inlierMask);

    /**
     * @brief Match features and estimate homography in one call
     * @param keypoints1 Keypoints from first frame
     * @param descriptors1 Descriptors from first frame
     * @param keypoints2 Keypoints from second frame
     * @param descriptors2 Descriptors from second frame
     * @param homography Output homography matrix (3x3)
     * @param inlierMatches Output matches that are inliers
     * @return true if matching and homography estimation succeeded
     */
    bool matchAndEstimateHomography(const std::vector<cv::KeyPoint>& keypoints1,
                                   const cv::Mat& descriptors1,
                                   const std::vector<cv::KeyPoint>& keypoints2,
                                   const cv::Mat& descriptors2,
                                   cv::Mat& homography,
                                   std::vector<cv::DMatch>& inlierMatches);

    /**
     * @brief Get the number of inliers from last homography estimation
     * @return Number of inliers
     */
    int getInlierCount() const { return m_inlierCount; }

    /**
     * @brief Get the inlier ratio from last homography estimation
     * @return Inlier ratio (0.0 to 1.0)
     */
    float getInlierRatio() const { return m_inlierRatio; }

    /**
     * @brief Set Lowe's ratio test threshold
     * @param threshold Ratio threshold (0.0 to 1.0, typically 0.7-0.8)
     */
    void setRatioThreshold(float threshold);

    /**
     * @brief Set RANSAC reprojection threshold
     * @param threshold Threshold in pixels (typically 1.0-5.0)
     */
    void setRansacThreshold(float threshold);

    /**
     * @brief Get matcher type
     * @return Current matcher type
     */
    MatcherType getMatcherType() const { return m_matcherType; }

private:
    MatcherType m_matcherType;
    float m_ratioThreshold;
    float m_ransacThreshold;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;
    int m_inlierCount;
    float m_inlierRatio;

    /**
     * @brief Initialize the matcher based on type
     */
    void initializeMatcher();

    /**
     * @brief Apply Lowe's ratio test to filter matches
     * @param knnMatches KNN matches (k=2)
     * @param goodMatches Output filtered matches
     */
    void applyRatioTest(const std::vector<std::vector<cv::DMatch>>& knnMatches,
                       std::vector<cv::DMatch>& goodMatches);

    /**
     * @brief Convert matches to point correspondences
     * @param keypoints1 Keypoints from first frame
     * @param keypoints2 Keypoints from second frame
     * @param matches Input matches
     * @param points1 Output points from first frame
     * @param points2 Output points from second frame
     */
    void matchesToPoints(const std::vector<cv::KeyPoint>& keypoints1,
                        const std::vector<cv::KeyPoint>& keypoints2,
                        const std::vector<cv::DMatch>& matches,
                        std::vector<cv::Point2f>& points1,
                        std::vector<cv::Point2f>& points2);
};

} // namespace Tracking
} // namespace VirtualAd
