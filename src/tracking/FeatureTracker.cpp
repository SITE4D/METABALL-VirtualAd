#include "FeatureTracker.h"
#include <chrono>

namespace VirtualAd {
namespace Tracking {

FeatureTracker::FeatureTracker(FeatureDetectorType detector_type,
                               const CameraIntrinsics& intrinsics,
                               int max_features)
    : state_(TrackingState::NOT_INITIALIZED)
    , tracked_feature_count_(0)
    , inlier_ratio_(0.0f)
    , last_processing_time_(0.0)
{
    // Initialize feature detector
    detector_ = std::make_unique<FeatureDetector>(detector_type, max_features);
    
    // Initialize feature matcher
    // Use BRUTE_FORCE for binary descriptors (ORB/AKAZE)
    matcher_ = std::make_unique<FeatureMatcher>(
        MatcherType::BRUTE_FORCE,
        0.75f,  // Lowe's ratio test threshold
        3.0f    // RANSAC threshold
    );
    
    // Initialize PnP solver
    pnp_solver_ = std::make_shared<PnPSolver>(
        intrinsics,
        PnPSolver::Algorithm::ITERATIVE,
        true,   // Use RANSAC
        8.0f    // RANSAC threshold
    );
}

bool FeatureTracker::initialize(const cv::Mat& reference_frame,
                                const std::vector<cv::Point3f>& object_points)
{
    if (reference_frame.empty()) {
        state_ = TrackingState::NOT_INITIALIZED;
        return false;
    }
    
    if (object_points.empty()) {
        state_ = TrackingState::NOT_INITIALIZED;
        return false;
    }
    
    // Store reference frame and object points
    reference_frame_ = reference_frame.clone();
    object_points_ = object_points;
    
    // Detect features in reference frame
    if (!detector_->detectFeatures(reference_frame_, 
                                   reference_keypoints_, 
                                   reference_descriptors_)) {
        state_ = TrackingState::NOT_INITIALIZED;
        return false;
    }
    
    if (reference_keypoints_.empty()) {
        state_ = TrackingState::NOT_INITIALIZED;
        return false;
    }
    
    // Update state
    state_ = TrackingState::TRACKING;
    return true;
}

bool FeatureTracker::track(const cv::Mat& frame, CameraPose& pose)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (frame.empty()) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    if (!isInitialized()) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    // Detect features in current frame
    if (!detector_->detectFeatures(frame, current_keypoints_, current_descriptors_)) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    if (current_keypoints_.empty()) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    // Match features
    if (!matcher_->matchFeatures(reference_keypoints_,
                                 reference_descriptors_,
                                 current_keypoints_,
                                 current_descriptors_,
                                 current_matches_)) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    if (current_matches_.empty() || current_matches_.size() < 4) {
        // Need at least 4 matches for PnP
        state_ = TrackingState::LOST;
        return false;
    }
    
    // Compute 3D-2D correspondences
    std::vector<cv::Point3f> matched_3d_points;
    std::vector<cv::Point2f> matched_2d_points;
    computeCorrespondences(current_matches_, matched_3d_points, matched_2d_points);
    
    if (matched_3d_points.size() < 4) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    // Solve PnP
    std::vector<uchar> inlier_mask;
    if (!pnp_solver_->solvePnP(matched_3d_points, matched_2d_points, pose, inlier_mask)) {
        state_ = TrackingState::LOST;
        return false;
    }
    
    // Update statistics
    tracked_feature_count_ = static_cast<int>(matched_3d_points.size());
    inlier_ratio_ = pnp_solver_->getInlierRatio();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Check tracking quality
    if (inlier_ratio_ < 0.3f) {
        // Too few inliers, tracking lost
        state_ = TrackingState::LOST;
        return false;
    }
    
    state_ = TrackingState::TRACKING;
    return true;
}

void FeatureTracker::reset()
{
    state_ = TrackingState::NOT_INITIALIZED;
    
    reference_frame_.release();
    reference_keypoints_.clear();
    reference_descriptors_.release();
    object_points_.clear();
    
    current_keypoints_.clear();
    current_descriptors_.release();
    current_matches_.clear();
    
    tracked_feature_count_ = 0;
    inlier_ratio_ = 0.0f;
    last_processing_time_ = 0.0;
}

void FeatureTracker::computeCorrespondences(const std::vector<cv::DMatch>& matches,
                                           std::vector<cv::Point3f>& matched_3d_points,
                                           std::vector<cv::Point2f>& matched_2d_points)
{
    matched_3d_points.clear();
    matched_2d_points.clear();
    
    // For planar target tracking, we need to establish correspondence
    // between reference frame keypoints and 3D object points.
    // 
    // Strategy: Use homography to map reference keypoints to object plane
    // This assumes the reference frame is a frontal view of the planar target.
    
    // For now, we'll use a simple approach:
    // Map reference keypoints uniformly to the object plane based on their positions
    
    for (const auto& match : matches) {
        // Get reference keypoint position
        const cv::KeyPoint& ref_kp = reference_keypoints_[match.queryIdx];
        const cv::KeyPoint& cur_kp = current_keypoints_[match.trainIdx];
        
        // Map reference keypoint to 3D object point
        // Assume reference frame dimensions match object plane dimensions
        float ref_x = ref_kp.pt.x / reference_frame_.cols;  // Normalize to [0, 1]
        float ref_y = ref_kp.pt.y / reference_frame_.rows;  // Normalize to [0, 1]
        
        // Map to object plane (assuming object_points_ defines a rectangular plane)
        // For a simple planar target, interpolate based on corner points
        if (object_points_.size() >= 4) {
            // Bilinear interpolation
            // Assume object_points_[0-3] are corners: TL, TR, BR, BL
            cv::Point3f tl = object_points_[0];
            cv::Point3f tr = object_points_[1];
            cv::Point3f br = object_points_[2];
            cv::Point3f bl = object_points_[3];
            
            // Interpolate
            cv::Point3f top = tl + (tr - tl) * ref_x;
            cv::Point3f bottom = bl + (br - bl) * ref_x;
            cv::Point3f point_3d = top + (bottom - top) * ref_y;
            
            matched_3d_points.push_back(point_3d);
            matched_2d_points.push_back(cur_kp.pt);
        }
    }
}

} // namespace Tracking
} // namespace VirtualAd
