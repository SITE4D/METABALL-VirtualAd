#pragma once

#include "FeatureDetector.h"
#include "FeatureMatcher.h"
#include "PnPSolver.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace VirtualAd {
namespace Tracking {

/**
 * @brief Tracking state
 */
enum class TrackingState {
    NOT_INITIALIZED,  // Tracker not initialized
    TRACKING,         // Successfully tracking
    LOST              // Tracking lost
};

/**
 * @brief Feature-based tracker for planar targets
 * 
 * This class integrates feature detection, matching, and PnP solving
 * to track a planar target (e.g., backnet) across video frames.
 */
class FeatureTracker {
public:
    /**
     * @brief Construct a new Feature Tracker
     * @param detector_type Feature detector type (ORB or AKAZE)
     * @param intrinsics Camera intrinsics
     * @param max_features Maximum features to detect (default: 1000)
     */
    explicit FeatureTracker(FeatureDetectorType detector_type,
                           const CameraIntrinsics& intrinsics,
                           int max_features = 1000);

    /**
     * @brief Destroy the Feature Tracker
     */
    ~FeatureTracker() = default;

    /**
     * @brief Initialize tracker with reference frame and 3D points
     * @param reference_frame Reference image (template)
     * @param object_points 3D points in world coordinates (planar target)
     * @return true if initialization succeeded
     */
    bool initialize(const cv::Mat& reference_frame,
                   const std::vector<cv::Point3f>& object_points);

    /**
     * @brief Track in current frame
     * @param frame Current frame
     * @param pose Output camera pose
     * @return true if tracking succeeded
     */
    bool track(const cv::Mat& frame, CameraPose& pose);

    /**
     * @brief Get current tracking state
     * @return Tracking state
     */
    TrackingState getState() const { return state_; }

    /**
     * @brief Get number of tracked features in last frame
     * @return Number of tracked features
     */
    int getTrackedFeatureCount() const { return tracked_feature_count_; }

    /**
     * @brief Get inlier ratio from last frame
     * @return Inlier ratio (0.0 to 1.0)
     */
    float getInlierRatio() const { return inlier_ratio_; }

    /**
     * @brief Get processing time of last frame
     * @return Processing time in milliseconds
     */
    double getLastProcessingTime() const { return last_processing_time_; }

    /**
     * @brief Reset tracker (clear reference frame)
     */
    void reset();

    /**
     * @brief Check if tracker is initialized
     * @return true if initialized
     */
    bool isInitialized() const { return state_ != TrackingState::NOT_INITIALIZED; }

    /**
     * @brief Get reference keypoints (for debugging/visualization)
     * @return Reference keypoints
     */
    const std::vector<cv::KeyPoint>& getReferenceKeypoints() const { 
        return reference_keypoints_; 
    }

    /**
     * @brief Get current keypoints (for debugging/visualization)
     * @return Current keypoints
     */
    const std::vector<cv::KeyPoint>& getCurrentKeypoints() const { 
        return current_keypoints_; 
    }

    /**
     * @brief Get current matches (for debugging/visualization)
     * @return Current matches
     */
    const std::vector<cv::DMatch>& getCurrentMatches() const { 
        return current_matches_; 
    }

private:
    // Components
    std::unique_ptr<FeatureDetector> detector_;
    std::unique_ptr<FeatureMatcher> matcher_;
    std::shared_ptr<PnPSolver> pnp_solver_;

    // Tracking state
    TrackingState state_;

    // Reference frame data
    cv::Mat reference_frame_;
    std::vector<cv::KeyPoint> reference_keypoints_;
    cv::Mat reference_descriptors_;
    std::vector<cv::Point3f> object_points_;

    // Current frame data
    std::vector<cv::KeyPoint> current_keypoints_;
    cv::Mat current_descriptors_;
    std::vector<cv::DMatch> current_matches_;

    // Statistics
    int tracked_feature_count_;
    float inlier_ratio_;
    double last_processing_time_;

    /**
     * @brief Compute 3D-2D correspondences from matches
     * @param matches Feature matches
     * @param matched_3d_points Output 3D points
     * @param matched_2d_points Output 2D points
     */
    void computeCorrespondences(const std::vector<cv::DMatch>& matches,
                               std::vector<cv::Point3f>& matched_3d_points,
                               std::vector<cv::Point2f>& matched_2d_points);
};

} // namespace Tracking
} // namespace VirtualAd
