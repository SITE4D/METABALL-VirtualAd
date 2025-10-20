#include "PnPSolver.h"
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace VirtualAd {
namespace Tracking {

// ============================================================================
// CameraIntrinsics Implementation
// ============================================================================

cv::Mat CameraIntrinsics::toCameraMatrix() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    return K;
}

cv::Mat CameraIntrinsics::toDistortionCoeffs() const {
    if (distCoeffs.empty()) {
        return cv::Mat();
    }
    
    cv::Mat dist(static_cast<int>(distCoeffs.size()), 1, CV_64F);
    for (size_t i = 0; i < distCoeffs.size(); i++) {
        dist.at<double>(static_cast<int>(i), 0) = distCoeffs[i];
    }
    return dist;
}

CameraIntrinsics CameraIntrinsics::createDefaultHD() {
    CameraIntrinsics intrinsics;
    // Typical HD video parameters (1280x720)
    intrinsics.fx = 1000.0;
    intrinsics.fy = 1000.0;
    intrinsics.cx = 640.0;
    intrinsics.cy = 360.0;
    intrinsics.distCoeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    return intrinsics;
}

CameraIntrinsics CameraIntrinsics::createDefaultFullHD() {
    CameraIntrinsics intrinsics;
    // Typical Full HD video parameters (1920x1080)
    intrinsics.fx = 1500.0;
    intrinsics.fy = 1500.0;
    intrinsics.cx = 960.0;
    intrinsics.cy = 540.0;
    intrinsics.distCoeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    return intrinsics;
}

// ============================================================================
// CameraPose Implementation
// ============================================================================

void CameraPose::updateRotationMatrix() {
    if (!rvec.empty()) {
        cv::Rodrigues(rvec, rotationMatrix);
    }
}

cv::Mat CameraPose::getTransformationMatrix() const {
    if (!isValid()) {
        return cv::Mat();
    }

    cv::Mat R = rotationMatrix;
    if (R.empty() && !rvec.empty()) {
        cv::Rodrigues(rvec, R);
    }

    // Create 4x4 transformation matrix [R|t; 0 0 0 1]
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    
    // Copy rotation (3x3)
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    
    // Copy translation (3x1)
    tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));
    
    return T;
}

// ============================================================================
// PnPSolver Implementation
// ============================================================================

PnPSolver::PnPSolver(const CameraIntrinsics& intrinsics,
                     Algorithm algorithm,
                     bool useRansac,
                     float ransacThreshold)
    : m_intrinsics(intrinsics)
    , m_algorithm(algorithm)
    , m_useRansac(useRansac)
    , m_ransacThreshold(ransacThreshold)
    , m_inlierCount(0)
    , m_inlierRatio(0.0f)
{
    if (ransacThreshold <= 0.0f) {
        throw std::invalid_argument("ransacThreshold must be greater than 0");
    }
    updateMatrices();
}

void PnPSolver::updateMatrices() {
    m_cameraMatrix = m_intrinsics.toCameraMatrix();
    m_distCoeffs = m_intrinsics.toDistortionCoeffs();
}

int PnPSolver::algorithmToFlag(Algorithm algorithm) const {
    switch (algorithm) {
        case Algorithm::ITERATIVE:
            return cv::SOLVEPNP_ITERATIVE;
        case Algorithm::EPNP:
            return cv::SOLVEPNP_EPNP;
        case Algorithm::P3P:
            return cv::SOLVEPNP_P3P;
        case Algorithm::DLS:
            return cv::SOLVEPNP_DLS;
        case Algorithm::UPNP:
            return cv::SOLVEPNP_UPNP;
        case Algorithm::IPPE:
            return cv::SOLVEPNP_IPPE;
        case Algorithm::IPPE_SQUARE:
            return cv::SOLVEPNP_IPPE_SQUARE;
        default:
            return cv::SOLVEPNP_ITERATIVE;
    }
}

bool PnPSolver::solvePnP(const std::vector<cv::Point3f>& objectPoints,
                         const std::vector<cv::Point2f>& imagePoints,
                         CameraPose& pose,
                         std::vector<uchar>& inlierMask) {
    // Validate input
    if (objectPoints.size() != imagePoints.size()) {
        std::cerr << "[PnPSolver] Error: Number of object points and image points mismatch" << std::endl;
        return false;
    }

    if (objectPoints.size() < 4) {
        std::cerr << "[PnPSolver] Error: Need at least 4 point correspondences (got " 
                  << objectPoints.size() << ")" << std::endl;
        return false;
    }

    try {
        bool success;
        
        if (m_useRansac) {
            // Use RANSAC for robust estimation
            success = cv::solvePnPRansac(
                objectPoints,
                imagePoints,
                m_cameraMatrix,
                m_distCoeffs,
                pose.rvec,
                pose.tvec,
                false,                  // useExtrinsicGuess
                100,                    // iterationsCount
                m_ransacThreshold,      // reprojectionError
                0.99,                   // confidence
                inlierMask,             // inliers
                algorithmToFlag(m_algorithm)
            );

            // Calculate inlier statistics
            if (success && !inlierMask.empty()) {
                m_inlierCount = cv::countNonZero(inlierMask);
                m_inlierRatio = static_cast<float>(m_inlierCount) / 
                               static_cast<float>(objectPoints.size());
            } else {
                m_inlierCount = 0;
                m_inlierRatio = 0.0f;
            }
        } else {
            // Use standard solvePnP without RANSAC
            success = cv::solvePnP(
                objectPoints,
                imagePoints,
                m_cameraMatrix,
                m_distCoeffs,
                pose.rvec,
                pose.tvec,
                false,                  // useExtrinsicGuess
                algorithmToFlag(m_algorithm)
            );

            // All points are inliers
            m_inlierCount = static_cast<int>(objectPoints.size());
            m_inlierRatio = 1.0f;
            inlierMask.clear();
            inlierMask.resize(objectPoints.size(), 1);
        }

        if (success) {
            // Update rotation matrix
            pose.updateRotationMatrix();
        }

        return success;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[PnPSolver] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "[PnPSolver] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool PnPSolver::refinePose(const std::vector<cv::Point3f>& objectPoints,
                           const std::vector<cv::Point2f>& imagePoints,
                           const CameraPose& initialPose,
                           CameraPose& pose) {
    if (!initialPose.isValid()) {
        std::cerr << "[PnPSolver] Error: Initial pose is invalid" << std::endl;
        return false;
    }

    if (objectPoints.size() != imagePoints.size()) {
        std::cerr << "[PnPSolver] Error: Number of object points and image points mismatch" << std::endl;
        return false;
    }

    if (objectPoints.size() < 4) {
        std::cerr << "[PnPSolver] Error: Need at least 4 point correspondences" << std::endl;
        return false;
    }

    try {
        // Copy initial pose
        pose.rvec = initialPose.rvec.clone();
        pose.tvec = initialPose.tvec.clone();

        // Refine pose using iterative method with initial guess
        bool success = cv::solvePnP(
            objectPoints,
            imagePoints,
            m_cameraMatrix,
            m_distCoeffs,
            pose.rvec,
            pose.tvec,
            true,                       // useExtrinsicGuess = true
            cv::SOLVEPNP_ITERATIVE      // Use iterative refinement
        );

        if (success) {
            // Update rotation matrix
            pose.updateRotationMatrix();
        }

        return success;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[PnPSolver] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
}

void PnPSolver::projectPoints(const std::vector<cv::Point3f>& objectPoints,
                              const CameraPose& pose,
                              std::vector<cv::Point2f>& imagePoints) {
    if (!pose.isValid()) {
        std::cerr << "[PnPSolver] Error: Pose is invalid" << std::endl;
        imagePoints.clear();
        return;
    }

    try {
        cv::projectPoints(
            objectPoints,
            pose.rvec,
            pose.tvec,
            m_cameraMatrix,
            m_distCoeffs,
            imagePoints
        );
    }
    catch (const cv::Exception& e) {
        std::cerr << "[PnPSolver] OpenCV exception: " << e.what() << std::endl;
        imagePoints.clear();
    }
}

double PnPSolver::calculateReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                             const std::vector<cv::Point2f>& imagePoints,
                                             const CameraPose& pose) {
    if (!pose.isValid()) {
        std::cerr << "[PnPSolver] Error: Pose is invalid" << std::endl;
        return -1.0;
    }

    if (objectPoints.size() != imagePoints.size()) {
        std::cerr << "[PnPSolver] Error: Number of object points and image points mismatch" << std::endl;
        return -1.0;
    }

    if (objectPoints.empty()) {
        return 0.0;
    }

    // Project 3D points to 2D
    std::vector<cv::Point2f> projectedPoints;
    projectPoints(objectPoints, pose, projectedPoints);

    if (projectedPoints.size() != imagePoints.size()) {
        std::cerr << "[PnPSolver] Error: Projection failed" << std::endl;
        return -1.0;
    }

    // Calculate mean squared error
    double totalError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); i++) {
        cv::Point2f diff = imagePoints[i] - projectedPoints[i];
        totalError += std::sqrt(diff.x * diff.x + diff.y * diff.y);
    }

    return totalError / static_cast<double>(imagePoints.size());
}

void PnPSolver::setIntrinsics(const CameraIntrinsics& intrinsics) {
    m_intrinsics = intrinsics;
    updateMatrices();
}

void PnPSolver::setRansacParams(bool useRansac, float threshold) {
    if (threshold <= 0.0f) {
        throw std::invalid_argument("ransacThreshold must be greater than 0");
    }
    m_useRansac = useRansac;
    m_ransacThreshold = threshold;
}

} // namespace Tracking
} // namespace VirtualAd
