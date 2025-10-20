#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <optional>

namespace VirtualAd {
namespace Tracking {

/**
 * @brief Camera intrinsic parameters
 */
struct CameraIntrinsics {
    double fx;              // Focal length in x
    double fy;              // Focal length in y
    double cx;              // Principal point x
    double cy;              // Principal point y
    std::vector<double> distCoeffs;  // Distortion coefficients [k1, k2, p1, p2, k3]

    /**
     * @brief Convert to OpenCV camera matrix
     */
    cv::Mat toCameraMatrix() const;

    /**
     * @brief Convert to OpenCV distortion coefficients
     */
    cv::Mat toDistortionCoeffs() const;

    /**
     * @brief Create default intrinsics for HD video (1920x1080)
     */
    static CameraIntrinsics createDefaultHD();

    /**
     * @brief Create default intrinsics for Full HD video (1920x1080)
     */
    static CameraIntrinsics createDefaultFullHD();
};

/**
 * @brief Camera pose (position and orientation)
 */
struct CameraPose {
    cv::Mat rvec;           // Rotation vector (3x1)
    cv::Mat tvec;           // Translation vector (3x1)
    cv::Mat rotationMatrix; // Rotation matrix (3x3)
    
    bool isValid() const {
        return !rvec.empty() && !tvec.empty();
    }

    /**
     * @brief Update rotation matrix from rotation vector
     */
    void updateRotationMatrix();

    /**
     * @brief Get 4x4 transformation matrix [R|t]
     */
    cv::Mat getTransformationMatrix() const;
};

/**
 * @brief PnP solver for camera pose estimation
 * 
 * This class uses OpenCV's solvePnP to estimate camera pose from
 * 2D-3D point correspondences. It supports multiple algorithms
 * including RANSAC for robust estimation.
 */
class PnPSolver {
public:
    /**
     * @brief PnP algorithm types
     */
    enum class Algorithm {
        ITERATIVE,      // Iterative method
        EPNP,          // Efficient PnP
        P3P,           // P3P method (requires exactly 4 points)
        DLS,           // DLS method
        UPNP,          // UPnP method
        IPPE,          // IPPE method
        IPPE_SQUARE    // IPPE for square targets
    };

    /**
     * @brief Construct a new PnP Solver
     * @param intrinsics Camera intrinsic parameters
     * @param algorithm PnP algorithm to use (default: ITERATIVE)
     * @param useRansac Use RANSAC for robust estimation (default: true)
     * @param ransacThreshold RANSAC reprojection threshold in pixels (default: 8.0)
     */
    explicit PnPSolver(const CameraIntrinsics& intrinsics,
                      Algorithm algorithm = Algorithm::ITERATIVE,
                      bool useRansac = true,
                      float ransacThreshold = 8.0f);

    /**
     * @brief Destroy the PnP Solver
     */
    ~PnPSolver() = default;

    /**
     * @brief Solve PnP to estimate camera pose
     * @param objectPoints 3D points in world coordinates (Nx3)
     * @param imagePoints 2D points in image coordinates (Nx2)
     * @param pose Output camera pose
     * @param inlierMask Output inlier mask (only if RANSAC is used)
     * @return true if pose was successfully estimated
     */
    bool solvePnP(const std::vector<cv::Point3f>& objectPoints,
                 const std::vector<cv::Point2f>& imagePoints,
                 CameraPose& pose,
                 std::vector<uchar>& inlierMask);

    /**
     * @brief Solve PnP with initial guess for refinement
     * @param objectPoints 3D points in world coordinates
     * @param imagePoints 2D points in image coordinates
     * @param initialPose Initial pose estimate
     * @param pose Output refined camera pose
     * @return true if pose was successfully refined
     */
    bool refinePose(const std::vector<cv::Point3f>& objectPoints,
                   const std::vector<cv::Point2f>& imagePoints,
                   const CameraPose& initialPose,
                   CameraPose& pose);

    /**
     * @brief Project 3D points to 2D using camera pose
     * @param objectPoints 3D points in world coordinates
     * @param pose Camera pose
     * @param imagePoints Output 2D points in image coordinates
     */
    void projectPoints(const std::vector<cv::Point3f>& objectPoints,
                      const CameraPose& pose,
                      std::vector<cv::Point2f>& imagePoints);

    /**
     * @brief Calculate reprojection error
     * @param objectPoints 3D points in world coordinates
     * @param imagePoints 2D points in image coordinates
     * @param pose Camera pose
     * @return Mean reprojection error in pixels
     */
    double calculateReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                     const std::vector<cv::Point2f>& imagePoints,
                                     const CameraPose& pose);

    /**
     * @brief Get the number of inliers from last solvePnP call
     * @return Number of inliers
     */
    int getInlierCount() const { return m_inlierCount; }

    /**
     * @brief Get the inlier ratio from last solvePnP call
     * @return Inlier ratio (0.0 to 1.0)
     */
    float getInlierRatio() const { return m_inlierRatio; }

    /**
     * @brief Set camera intrinsics
     * @param intrinsics New camera intrinsics
     */
    void setIntrinsics(const CameraIntrinsics& intrinsics);

    /**
     * @brief Get camera intrinsics
     * @return Current camera intrinsics
     */
    const CameraIntrinsics& getIntrinsics() const { return m_intrinsics; }

    /**
     * @brief Set PnP algorithm
     * @param algorithm New algorithm
     */
    void setAlgorithm(Algorithm algorithm) { m_algorithm = algorithm; }

    /**
     * @brief Set RANSAC parameters
     * @param useRansac Enable/disable RANSAC
     * @param threshold RANSAC reprojection threshold in pixels
     */
    void setRansacParams(bool useRansac, float threshold);

private:
    CameraIntrinsics m_intrinsics;
    cv::Mat m_cameraMatrix;
    cv::Mat m_distCoeffs;
    Algorithm m_algorithm;
    bool m_useRansac;
    float m_ransacThreshold;
    int m_inlierCount;
    float m_inlierRatio;

    /**
     * @brief Update OpenCV matrices from intrinsics
     */
    void updateMatrices();

    /**
     * @brief Convert Algorithm enum to OpenCV flag
     */
    int algorithmToFlag(Algorithm algorithm) const;
};

} // namespace Tracking
} // namespace VirtualAd
