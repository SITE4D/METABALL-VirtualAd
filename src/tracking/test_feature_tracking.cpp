/**
 * @file test_feature_tracking.cpp
 * @brief Test program for FeatureTracker
 * 
 * This program tests the feature-based tracking system by:
 * 1. Loading a video file
 * 2. Initializing tracker with first frame as reference
 * 3. Tracking across subsequent frames
 * 4. Visualizing tracking results
 */

#include "../core/FilePlaybackSource.h"
#include "FeatureTracker.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace VirtualAd;
using namespace VirtualAd::Tracking;

/**
 * @brief Define reference plane corners (planar target)
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
 * @brief Draw tracking information on frame
 */
void drawTrackingInfo(cv::Mat& frame, const FeatureTracker& tracker,
                     const CameraPose& pose, int frame_num, double fps) {
    // Draw statistics overlay
    int y = 30;
    int line_height = 30;
    cv::Scalar color(0, 255, 0);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.7;
    int thickness = 2;
    
    // Draw semi-transparent background
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(450, 180), 
                 cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(450, 180), 
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
    oss << "Features: " << tracker.getTrackedFeatureCount();
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    oss.str("");
    oss << "Inliers: " << std::fixed << std::setprecision(2) << (tracker.getInlierRatio() * 100) << "%";
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    oss.str("");
    oss << "Time: " << std::fixed << std::setprecision(2) << tracker.getLastProcessingTime() << " ms";
    cv::putText(frame, oss.str(), cv::Point(20, y), font, font_scale, color, thickness);
    y += line_height;
    
    // Draw state
    std::string state_str;
    cv::Scalar state_color;
    switch (tracker.getState()) {
        case TrackingState::NOT_INITIALIZED:
            state_str = "NOT INITIALIZED";
            state_color = cv::Scalar(0, 0, 255);
            break;
        case TrackingState::TRACKING:
            state_str = "TRACKING";
            state_color = cv::Scalar(0, 255, 0);
            break;
        case TrackingState::LOST:
            state_str = "LOST";
            state_color = cv::Scalar(0, 165, 255);
            break;
    }
    cv::putText(frame, state_str, cv::Point(20, y), font, font_scale, state_color, thickness);
}

/**
 * @brief Draw feature matches on frame
 */
void drawMatches(cv::Mat& frame, const FeatureTracker& tracker) {
    const auto& keypoints = tracker.getCurrentKeypoints();
    const auto& matches = tracker.getCurrentMatches();
    
    // Draw all detected keypoints
    for (const auto& kp : keypoints) {
        cv::circle(frame, kp.pt, 3, cv::Scalar(255, 0, 0), 1);
    }
    
    // Draw matched keypoints
    for (const auto& match : matches) {
        const auto& kp = keypoints[match.trainIdx];
        cv::circle(frame, kp.pt, 5, cv::Scalar(0, 255, 255), 2);
    }
}

/**
 * @brief Draw 3D coordinate axes
 */
void drawAxes(cv::Mat& frame, const CameraPose& pose,
              const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    if (!pose.isValid()) {
        return;
    }
    
    // Define 3D axes points
    std::vector<cv::Point3f> axes_points;
    axes_points.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));  // Origin
    axes_points.push_back(cv::Point3f(0.3f, 0.0f, 0.0f));  // X-axis
    axes_points.push_back(cv::Point3f(0.0f, 0.3f, 0.0f));  // Y-axis
    axes_points.push_back(cv::Point3f(0.0f, 0.0f, 0.3f));  // Z-axis
    
    // Project to image
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axes_points, pose.rvec, pose.tvec,
                     camera_matrix, dist_coeffs, image_points);
    
    // Draw axes
    cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 3);  // X: Red
    cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 3);  // Y: Green
    cv::line(frame, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 3);  // Z: Blue
    
    // Draw origin
    cv::circle(frame, image_points[0], 5, cv::Scalar(255, 255, 255), -1);
}

int main(int argc, char** argv) {
    std::cout << "=== Feature Tracking Test ===" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    std::string video_path = "data/samples/test_video.mp4";
    std::string detector_type_str = "ORB";
    
    if (argc >= 2) {
        video_path = argv[1];
    }
    if (argc >= 3) {
        detector_type_str = argv[2];
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Video: " << video_path << std::endl;
    std::cout << "  Detector: " << detector_type_str << std::endl;
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
    cv::Mat camera_matrix = intrinsics.toCameraMatrix();
    cv::Mat dist_coeffs = intrinsics.toDistortionCoeffs();
    
    // Initialize feature tracker
    FeatureDetectorType detector_type = (detector_type_str == "AKAZE") 
        ? FeatureDetectorType::AKAZE 
        : FeatureDetectorType::ORB;
    
    FeatureTracker tracker(detector_type, intrinsics, 1000);
    
    // Create display window
    const std::string window_name = "Feature Tracking Test";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);
    
    // Get first frame as reference
    cv::Mat reference_frame = video_source.getFrame();
    if (reference_frame.empty()) {
        std::cerr << "ERROR: Failed to get reference frame" << std::endl;
        return -1;
    }
    
    // Initialize tracker
    std::vector<cv::Point3f> object_points = createReferencePlane();
    if (!tracker.initialize(reference_frame, object_points)) {
        std::cerr << "ERROR: Failed to initialize tracker" << std::endl;
        return -1;
    }
    
    std::cout << "Tracker initialized" << std::endl;
    std::cout << "Press 'q' or ESC to quit" << std::endl;
    std::cout << std::endl;
    
    // Processing loop
    int frame_count = 0;
    int tracked_frames = 0;
    int lost_frames = 0;
    double total_time = 0.0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (video_source.isOpened()) {
        // Get frame
        cv::Mat frame = video_source.getFrame();
        if (frame.empty()) {
            break;
        }
        
        frame_count++;
        
        // Create display frame
        cv::Mat display_frame = frame.clone();
        
        // Track
        CameraPose pose;
        bool success = tracker.track(frame, pose);
        
        if (success) {
            tracked_frames++;
            total_time += tracker.getLastProcessingTime();
            
            // Draw matches
            drawMatches(display_frame, tracker);
            
            // Draw 3D axes
            drawAxes(display_frame, pose, camera_matrix, dist_coeffs);
            
            std::cout << "Frame " << std::setw(4) << frame_count
                      << " | Features: " << std::setw(4) << tracker.getTrackedFeatureCount()
                      << " | Inliers: " << std::fixed << std::setprecision(1) << std::setw(5) 
                      << (tracker.getInlierRatio() * 100) << "%"
                      << " | Time: " << std::fixed << std::setprecision(2) << std::setw(6)
                      << tracker.getLastProcessingTime() << " ms" << std::endl;
        } else {
            lost_frames++;
            std::cout << "Frame " << std::setw(4) << frame_count
                      << " | LOST" << std::endl;
        }
        
        // Calculate FPS
        auto current_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        double fps = frame_count / elapsed;
        
        // Draw tracking info
        drawTrackingInfo(display_frame, tracker, pose, frame_count, fps);
        
        // Display frame
        cv::imshow(window_name, display_frame);
        
        // Check for user input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' or ESC
            break;
        }
    }
    
    // Print summary
    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Tracked frames: " << tracked_frames 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * tracked_frames / frame_count) << "%)" << std::endl;
    std::cout << "Lost frames: " << lost_frames 
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * lost_frames / frame_count) << "%)" << std::endl;
    
    if (tracked_frames > 0) {
        std::cout << "Average processing time: "
                  << std::fixed << std::setprecision(2)
                  << (total_time / tracked_frames) << " ms/frame" << std::endl;
    }
    
    cv::destroyAllWindows();
    video_source.stop();
    
    return 0;
}
