#include "FeatureMatcher.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace VirtualAd {
namespace Tracking {

FeatureMatcher::FeatureMatcher(MatcherType matcherType, 
                              float ratioThreshold, 
                              float ransacThreshold)
    : m_matcherType(matcherType)
    , m_ratioThreshold(ratioThreshold)
    , m_ransacThreshold(ransacThreshold)
    , m_inlierCount(0)
    , m_inlierRatio(0.0f)
{
    if (ratioThreshold <= 0.0f || ratioThreshold >= 1.0f) {
        throw std::invalid_argument("ratioThreshold must be between 0 and 1");
    }
    if (ransacThreshold <= 0.0f) {
        throw std::invalid_argument("ransacThreshold must be greater than 0");
    }
    initializeMatcher();
}

void FeatureMatcher::initializeMatcher() {
    switch (m_matcherType) {
        case MatcherType::BRUTE_FORCE:
            // BFMatcher with Hamming distance for binary descriptors (ORB, AKAZE)
            m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
            break;

        case MatcherType::FLANN:
            // FLANN matcher for float descriptors
            // Note: For binary descriptors, FLANN needs special configuration
            m_matcher = cv::FlannBasedMatcher::create();
            break;

        default:
            throw std::runtime_error("Unknown matcher type");
    }

    if (!m_matcher) {
        throw std::runtime_error("Failed to create descriptor matcher");
    }
}

bool FeatureMatcher::matchFeatures(const std::vector<cv::KeyPoint>& keypoints1,
                                  const cv::Mat& descriptors1,
                                  const std::vector<cv::KeyPoint>& keypoints2,
                                  const cv::Mat& descriptors2,
                                  std::vector<cv::DMatch>& matches) {
    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "[FeatureMatcher] Error: Empty descriptors" << std::endl;
        return false;
    }

    if (descriptors1.cols != descriptors2.cols) {
        std::cerr << "[FeatureMatcher] Error: Descriptor dimensions mismatch" << std::endl;
        return false;
    }

    if (!m_matcher) {
        std::cerr << "[FeatureMatcher] Error: Matcher not initialized" << std::endl;
        return false;
    }

    try {
        // Perform KNN matching with k=2 for ratio test
        std::vector<std::vector<cv::DMatch>> knnMatches;
        m_matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

        // Apply Lowe's ratio test
        matches.clear();
        applyRatioTest(knnMatches, matches);

        return !matches.empty();
    }
    catch (const cv::Exception& e) {
        std::cerr << "[FeatureMatcher] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "[FeatureMatcher] Exception: " << e.what() << std::endl;
        return false;
    }
}

void FeatureMatcher::applyRatioTest(const std::vector<std::vector<cv::DMatch>>& knnMatches,
                                   std::vector<cv::DMatch>& goodMatches) {
    goodMatches.clear();
    
    for (const auto& match : knnMatches) {
        // Need at least 2 matches for ratio test
        if (match.size() < 2) {
            continue;
        }

        // Lowe's ratio test: distance of best match / distance of second best match
        if (match[0].distance < m_ratioThreshold * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
}

bool FeatureMatcher::estimateHomography(const std::vector<cv::KeyPoint>& keypoints1,
                                       const std::vector<cv::KeyPoint>& keypoints2,
                                       const std::vector<cv::DMatch>& matches,
                                       cv::Mat& homography,
                                       std::vector<uchar>& inlierMask) {
    // Need at least 4 points to estimate homography
    if (matches.size() < 4) {
        std::cerr << "[FeatureMatcher] Error: Need at least 4 matches to estimate homography (got " 
                  << matches.size() << ")" << std::endl;
        return false;
    }

    // Convert matches to point correspondences
    std::vector<cv::Point2f> points1, points2;
    matchesToPoints(keypoints1, keypoints2, matches, points1, points2);

    try {
        // Estimate homography using RANSAC
        homography = cv::findHomography(points1, points2, cv::RANSAC, 
                                       m_ransacThreshold, inlierMask);

        if (homography.empty()) {
            std::cerr << "[FeatureMatcher] Error: Failed to estimate homography" << std::endl;
            return false;
        }

        // Calculate inlier statistics
        m_inlierCount = cv::countNonZero(inlierMask);
        m_inlierRatio = static_cast<float>(m_inlierCount) / static_cast<float>(matches.size());

        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[FeatureMatcher] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
}

bool FeatureMatcher::matchAndEstimateHomography(const std::vector<cv::KeyPoint>& keypoints1,
                                               const cv::Mat& descriptors1,
                                               const std::vector<cv::KeyPoint>& keypoints2,
                                               const cv::Mat& descriptors2,
                                               cv::Mat& homography,
                                               std::vector<cv::DMatch>& inlierMatches) {
    // Match features
    std::vector<cv::DMatch> matches;
    if (!matchFeatures(keypoints1, descriptors1, keypoints2, descriptors2, matches)) {
        return false;
    }

    // Estimate homography
    std::vector<uchar> inlierMask;
    if (!estimateHomography(keypoints1, keypoints2, matches, homography, inlierMask)) {
        return false;
    }

    // Extract inlier matches
    inlierMatches.clear();
    for (size_t i = 0; i < matches.size(); i++) {
        if (inlierMask[i]) {
            inlierMatches.push_back(matches[i]);
        }
    }

    return true;
}

void FeatureMatcher::matchesToPoints(const std::vector<cv::KeyPoint>& keypoints1,
                                    const std::vector<cv::KeyPoint>& keypoints2,
                                    const std::vector<cv::DMatch>& matches,
                                    std::vector<cv::Point2f>& points1,
                                    std::vector<cv::Point2f>& points2) {
    points1.clear();
    points2.clear();
    points1.reserve(matches.size());
    points2.reserve(matches.size());

    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
}

void FeatureMatcher::setRatioThreshold(float threshold) {
    if (threshold <= 0.0f || threshold >= 1.0f) {
        throw std::invalid_argument("ratioThreshold must be between 0 and 1");
    }
    m_ratioThreshold = threshold;
}

void FeatureMatcher::setRansacThreshold(float threshold) {
    if (threshold <= 0.0f) {
        throw std::invalid_argument("ransacThreshold must be greater than 0");
    }
    m_ransacThreshold = threshold;
}

} // namespace Tracking
} // namespace VirtualAd
