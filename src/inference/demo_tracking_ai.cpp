/**
 * @file demo_tracking_ai.cpp
 * @brief AI-powered tracking demo application
 * 
 * This demo integrates:
 * - Video playback (FilePlaybackSource)
 * - PnP solver (traditional pose estimation)
 * - AI-powered pose refinement (CameraPoseRefiner)
 */

#include "../core/FilePlaybackSource.h"
#include "CameraPoseRefiner.h"
#include "../tracking/PnPSolver.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace VirtualAd;
using namespace VirtualAd::Tracking;
using namespace metaball;

/**
 * @brief Define reference plane corners (planar target)
 * 
 * Assume a 1.0 x 0.75 meter planar target (e.g., poster)
 */
std::vector<cv::Point3f> createReferencePlane() {
    std::vector<cv::Point3f> points;
    // Define corners in world coordinates (Z=0 plane)
    points.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));      // Top-left
    points.push_back(cv::Point3f(1.0f, 0.0f, 0.0f));      // Top-right
    points.push_back(cv::Point3f(1.0f, 0.75f, 0.0f));     // Bottom-right
    points.push_back(cv::Point3f(0.0f, 0.75f, 0.0f));     // Bottom-left
    return points;
}

/**
 * @brief Detect reference plane corners in image
 * 
 * Simplified version: manually specify or use simple detection
 * In real application, use feature detection + matching
 * 
 * @param frame Input frame
 * @return Detected corners (or predefined for testing)
 */
std::vector<cv::Point2f> detectImagePoints(const cv::Mat& frame) {
    // For demo: use predefined points in center of frame
    // Real implementation would use FeatureDetector + FeatureMatcher
    int cx = frame.cols / 2;
    int cy = frame.rows / 2;
    int w = 400;  // Width of detected region
    int h = 300;  // Height of detected region
    
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(cx - w/2, cy - h/2));  // Top-left
    points.push_back(cv::Point2f(cx + w/2, cy - h/2));  // Top-right
    points.push_back(cv::Point2f(cx + w/2, cy + h/2));  // Bottom-right
    points.push_back(cv::Point2f(cx - w/2, cy + h/2));  // Bottom-left
    
    return points;
}

/**
 * @brief Convert pose to human-readable string
 */
std::string poseToString(const CameraPose& pose) {
    if (!pose.isValid()) {
        return "Invalid pose";
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "T: [" << pose.tvec.at<double>(0) << ", "
        << pose.tvec.at<double>(1) << ", "
        << pose.tvec.at<double>(2) << "] ";
    oss << "R: [" << pose.rvec.at<double>(0) << ", "
        << pose.rvec.at<double>(1) << ", "
        << pose.rvec.at<double>(2) << "]";
    return oss.str();
}

/**
 * @brief Print statistics
 */
void printStats(int frame_num, double fps, const CameraPose& pose, 
                double processing_time, const std::string& mode) {
    std::cout << "Frame " << std::setw(4) << frame_num 
              << " | FPS: " << std::setw(5) << std::fixed << std::setprecision(1) << fps
              << " | Mode: " << std::setw(10) << mode
              << " | Time: " << std::setw(6) << std::setprecision(2) << processing_time << "ms"
              << " | " << poseToString(pose) << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== METABALL Virtual Ad - AI Tracking Demo ===" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    std::string video_path = "data/samples/test_video.mp4";
    std::string model_path = "models/camera_pose_net.onnx";
    std::string mode_str = "BLENDED";
    float blend_alpha = 0.5f;
    
    if (argc >= 2) {
        video_path = argv[1];
    }
    if (argc >= 3) {
        model_path = argv[2];
    }
    if (argc >= 4) {
        mode_str = argv[3];
    }
    if (argc >= 5) {
        blend_alpha = std::stof(argv[4]);
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Video: " << video_path << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Mode: " << mode_str << std::endl;
    std::cout << "  Blend Alpha: " << blend_alpha << std::endl;
    std::cout << std::endl;
    
    // Initialize video source
    FilePlaybackSource video_source(video_path);
    if (!video_source.start()) {
        std::cerr << "ERROR: Failed to open video: " << video_path << std::endl;
        return -1;
    }
    
    std::cout << "Video loaded: " << video_source.getWidth() << "x" 
              << video_source.getHeight() << " @ " 
              << video_source.getFrameRate() << " FPS" << std::endl;
    
    // Initialize camera intrinsics
    CameraIntrinsics intrinsics = CameraIntrinsics::createDefaultFullHD();
    
    // Initialize PnP solver
    auto pnp_solver = std::make_shared<PnPSolver>(
        intrinsics,
        PnPSolver::Algorithm::ITERATIVE,
        true,   // Use RANSAC
        8.0f    // RANSAC threshold
    );
    
    // Initialize pose refiner
    CameraPoseRefiner refiner(pnp_solver);
    
    // Set mode
    CameraPoseRefiner::Mode mode = CameraPoseRefiner::Mode::BLENDED;
    if (mode_str == "PNP_ONLY") {
        mode = CameraPoseRefiner::Mode::PNP_ONLY;
    } else if (mode_str == "AI_ONLY") {
        mode = CameraPoseRefiner::Mode::AI_ONLY;
    }
    refiner.setMode(mode);
    refiner.setBlendAlpha(blend_alpha);
    
    // Load AI model (if needed)
    if (mode == CameraPoseRefiner::Mode::AI_ONLY || 
        mode == CameraPoseRefiner::Mode::BLENDED) {
        std::cout << "Loading AI model..." << std::endl;
        if (!refiner.loadModel(model_path)) {
            std::cerr << "WARNING: Failed to load AI model: " << model_path << std::endl;
            std::cerr << "         Falling back to PNP_ONLY mode" << std::endl;
            refiner.setMode(CameraPoseRefiner::Mode::PNP_ONLY);
        } else {
            std::cout << "AI model loaded successfully" << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Processing video..." << std::endl;
    std::cout << "Press 'q' or ESC to quit" << std::endl;
    std::cout << std::endl;
    
    // Create reference 3D points
    std::vector<cv::Point3f> object_points = createReferencePlane();
    
    // Processing loop
    int frame_count = 0;
    double total_time = 0.0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (video_source.isOpened()) {
        // Get frame
        cv::Mat frame = video_source.getFrame();
        if (frame.empty()) {
            break;
        }
        
        frame_count++;
        
        // Detect image points (simplified)
        std::vector<cv::Point2f> image_points = detectImagePoints(frame);
        
        // Refine pose
        auto process_start = std::chrono::high_resolution_clock::now();
        
        CameraPose pose;
        std::vector<uchar> inlier_mask;
        bool success = refiner.refinePose(frame, object_points, image_points, pose, inlier_mask);
        
        auto process_end = std::chrono::high_resolution_clock::now();
        double processing_time = std::chrono::duration<double, std::milli>(process_end - process_start).count();
        total_time += processing_time;
        
        // Calculate FPS
        auto current_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        double fps = frame_count / elapsed;
        
        // Print statistics
        if (success) {
            printStats(frame_count, fps, pose, processing_time, mode_str);
        } else {
            std::cout << "Frame " << std::setw(4) << frame_count 
                      << " | FAILED: " << refiner.getLastError() << std::endl;
        }
        
        // Check for user input (non-blocking)
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        }
    }
    
    // Print summary
    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Average processing time: " 
              << std::fixed << std::setprecision(2) 
              << (total_time / frame_count) << " ms/frame" << std::endl;
    std::cout << "Target: <= 10 ms/frame" << std::endl;
    
    if (total_time / frame_count <= 10.0) {
        std::cout << "PASSED: Performance target achieved!" << std::endl;
    } else {
        std::cout << "FAILED: Performance target not met" << std::endl;
    }
    
    video_source.stop();
    return 0;
}
