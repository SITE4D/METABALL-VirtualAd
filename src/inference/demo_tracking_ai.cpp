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

/**
 * @brief Draw 3D coordinate axes on frame
 * 
 * @param frame Input/output frame
 * @param pose Camera pose
 * @param camera_matrix Camera intrinsic matrix
 * @param dist_coeffs Distortion coefficients
 */
void drawAxes(cv::Mat& frame, const CameraPose& pose, 
              const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    if (!pose.isValid()) {
        return;
    }
    
    // Define 3D axes points (origin + 3 axis endpoints)
    std::vector<cv::Point3f> axes_points;
    axes_points.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));  // Origin
    axes_points.push_back(cv::Point3f(0.3f, 0.0f, 0.0f));  // X-axis (red)
    axes_points.push_back(cv::Point3f(0.0f, 0.3f, 0.0f));  // Y-axis (green)
    axes_points.push_back(cv::Point3f(0.0f, 0.0f, 0.3f));  // Z-axis (blue)
    
    // Project to image
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axes_points, pose.rvec, pose.tvec, 
                     camera_matrix, dist_coeffs, image_points);
    
    // Draw axes
    cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 3);  // X: Red
    cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 3);  // Y: Green
    cv::line(frame, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 3);  // Z: Blue
    
    // Draw origin point
    cv::circle(frame, image_points[0], 5, cv::Scalar(255, 255, 255), -1);
}

/**
 * @brief Draw detected corners on frame
 */
void drawCorners(cv::Mat& frame, const std::vector<cv::Point2f>& corners) {
    for (size_t i = 0; i < corners.size(); i++) {
        cv::circle(frame, corners[i], 8, cv::Scalar(0, 255, 255), -1);
        cv::putText(frame, std::to_string(i), corners[i] + cv::Point2f(10, 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    }
    
    // Draw outline
    if (corners.size() >= 4) {
        for (size_t i = 0; i < corners.size(); i++) {
            cv::line(frame, corners[i], corners[(i + 1) % corners.size()], 
                    cv::Scalar(0, 255, 255), 2);
        }
    }
}

/**
 * @brief Draw statistics overlay on frame
 */
void drawStatsOverlay(cv::Mat& frame, int frame_num, double fps, 
                     const std::string& mode, double processing_time,
                     float blend_alpha) {
    int y = 30;
    int line_height = 30;
    cv::Scalar color(0, 255, 0);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.7;
    int thickness = 2;
    
    // Draw semi-transparent background
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(450, 150), 
                 cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(450, 150), 
                 cv::Scalar(255, 255, 255), 2);
    
    // Draw text
    std::ostringstream oss;
    oss << "Frame: " << frame_num;
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    oss.str("");
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    oss.str("");
    oss << "Mode: " << mode;
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    oss.str("");
    oss << "Alpha: " << std::fixed << std::setprecision(2) << blend_alpha;
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    oss.str("");
    oss << "Time: " << std::fixed << std::setprecision(2) << processing_time << " ms";
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
}

/**
 * @brief Draw help overlay
 */
void drawHelpOverlay(cv::Mat& frame) {
    int y = frame.rows - 150;
    int line_height = 25;
    cv::Scalar color(255, 255, 0);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    
    // Draw semi-transparent background
    cv::rectangle(frame, cv::Point(10, y - 30), cv::Point(400, frame.rows - 10), 
                 cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, cv::Point(10, y - 30), cv::Point(400, frame.rows - 10), 
                 cv::Scalar(255, 255, 255), 2);
    
    cv::putText(frame, "Controls:", cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    cv::putText(frame, "1: PNP_ONLY", cv::Point(20, y), font, font_scale, cv::Scalar(200, 200, 200), thickness);
    y += line_height;
    cv::putText(frame, "2: AI_ONLY", cv::Point(20, y), font, font_scale, cv::Scalar(200, 200, 200), thickness);
    y += line_height;
    cv::putText(frame, "3: BLENDED", cv::Point(20, y), font, font_scale, cv::Scalar(200, 200, 200), thickness);
    y += line_height;
    cv::putText(frame, "Q/ESC: Quit", cv::Point(20, y), font, font_scale, cv::Scalar(200, 200, 200), thickness);
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
    std::cout << "Press '1/2/3' to switch modes, 'q' or ESC to quit" << std::endl;
    std::cout << std::endl;
    
    // Create reference 3D points
    std::vector<cv::Point3f> object_points = createReferencePlane();
    
    // Create display window
    const std::string window_name = "METABALL Virtual Ad - AI Tracking Demo";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);
    
    // Get camera matrix for visualization
    cv::Mat camera_matrix = intrinsics.toCameraMatrix();
    cv::Mat dist_coeffs = intrinsics.toDistortionCoeffs();
    
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
        
        // Create display frame (clone for drawing)
        cv::Mat display_frame = frame.clone();
        
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
        
        // Draw visualizations
        if (success) {
            // Draw detected corners
            drawCorners(display_frame, image_points);
            
            // Draw 3D coordinate axes
            drawAxes(display_frame, pose, camera_matrix, dist_coeffs);
            
            // Print statistics to console
            printStats(frame_count, fps, pose, processing_time, mode_str);
        } else {
            std::cout << "Frame " << std::setw(4) << frame_count 
                      << " | FAILED: " << refiner.getLastError() << std::endl;
        }
        
        // Draw overlay statistics
        drawStatsOverlay(display_frame, frame_count, fps, mode_str, 
                        processing_time, refiner.getBlendAlpha());
        
        // Draw help overlay
        drawHelpOverlay(display_frame);
        
        // Display frame
        cv::imshow(window_name, display_frame);
        
        // Check for user input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == '1') {
            // Switch to PNP_ONLY
            mode = CameraPoseRefiner::Mode::PNP_ONLY;
            mode_str = "PNP_ONLY";
            refiner.setMode(mode);
            std::cout << "Switched to PNP_ONLY mode" << std::endl;
        } else if (key == '2') {
            // Switch to AI_ONLY
            if (refiner.isModelLoaded()) {
                mode = CameraPoseRefiner::Mode::AI_ONLY;
                mode_str = "AI_ONLY";
                refiner.setMode(mode);
                std::cout << "Switched to AI_ONLY mode" << std::endl;
            } else {
                std::cout << "AI model not loaded, cannot switch to AI_ONLY" << std::endl;
            }
        } else if (key == '3') {
            // Switch to BLENDED
            if (refiner.isModelLoaded()) {
                mode = CameraPoseRefiner::Mode::BLENDED;
                mode_str = "BLENDED";
                refiner.setMode(mode);
                std::cout << "Switched to BLENDED mode" << std::endl;
            } else {
                std::cout << "AI model not loaded, cannot switch to BLENDED" << std::endl;
            }
        }
    }
    
    // Cleanup
    cv::destroyAllWindows();
    
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
