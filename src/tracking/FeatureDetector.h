#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <memory>
#include <string>

namespace VirtualAd {
namespace Tracking {

/**
 * @brief Feature detector types supported
 */
enum class FeatureDetectorType {
    ORB,
    AKAZE
};

/**
 * @brief Feature detector wrapper for ORB/AKAZE algorithms
 * 
 * This class provides a unified interface for feature detection and description
 * extraction using either ORB or AKAZE algorithms. It's designed for real-time
 * tracking in the METABALL Virtual Ad system.
 */
class FeatureDetector {
public:
    /**
     * @brief Construct a new Feature Detector
     * @param type Detector type (ORB or AKAZE)
     * @param maxFeatures Maximum number of features to detect (default: 1000)
     */
    explicit FeatureDetector(FeatureDetectorType type = FeatureDetectorType::ORB, 
                            int maxFeatures = 1000);

    /**
     * @brief Destroy the Feature Detector
     */
    ~FeatureDetector() = default;

    /**
     * @brief Detect features in an image
     * @param image Input image (grayscale or color)
     * @param keypoints Output keypoints
     * @param descriptors Output descriptors
     * @return true if features were successfully detected
     */
    bool detectFeatures(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors);

    /**
     * @brief Get the detector type
     * @return Current detector type
     */
    FeatureDetectorType getType() const { return m_type; }

    /**
     * @brief Get the name of the current detector
     * @return Detector name as string
     */
    std::string getName() const;

    /**
     * @brief Set maximum number of features to detect
     * @param maxFeatures Maximum features (must be > 0)
     */
    void setMaxFeatures(int maxFeatures);

    /**
     * @brief Get maximum number of features
     * @return Maximum features setting
     */
    int getMaxFeatures() const { return m_maxFeatures; }

private:
    FeatureDetectorType m_type;
    int m_maxFeatures;
    cv::Ptr<cv::Feature2D> m_detector;

    /**
     * @brief Initialize the detector based on type
     */
    void initializeDetector();
};

} // namespace Tracking
} // namespace VirtualAd
