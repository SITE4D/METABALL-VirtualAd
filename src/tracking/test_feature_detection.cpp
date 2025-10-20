#include "tracking/FeatureDetector.h"
#include "core/FilePlaybackSource.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace VirtualAd;
using namespace VirtualAd::Tracking;

/**
 * @brief Draw keypoints on an image
 */
cv::Mat drawKeypointsColored(const cv::Mat& image, 
                             const std::vector<cv::KeyPoint>& keypoints,
                             const std::string& title) {
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0),
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    // Add title and stats
    cv::putText(output, title, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    std::string stats = "Features: " + std::to_string(keypoints.size());
    cv::putText(output, stats, cv::Point(10, 70),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    return output;
}

/**
 * @brief Test feature detector with a single image
 */
void testWithImage(const std::string& imagePath) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing with image: " << imagePath << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << std::endl;
        return;
    }

    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

    // Test ORB detector
    {
        std::cout << "\n--- Testing ORB Detector ---" << std::endl;
        FeatureDetector orbDetector(FeatureDetectorType::ORB, 1000);
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = orbDetector.detectFeatures(image, keypoints, descriptors);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (success) {
            std::cout << "Detected " << keypoints.size() << " features" << std::endl;
            std::cout << "Descriptor size: " << descriptors.rows << "x" << descriptors.cols << std::endl;
            std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
            
            cv::Mat outputOrb = drawKeypointsColored(image, keypoints, "ORB Features");
            cv::imshow("ORB Features", outputOrb);
        } else {
            std::cerr << "Failed to detect ORB features" << std::endl;
        }
    }

    // Test AKAZE detector
    {
        std::cout << "\n--- Testing AKAZE Detector ---" << std::endl;
        FeatureDetector akazeDetector(FeatureDetectorType::AKAZE, 1000);
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = akazeDetector.detectFeatures(image, keypoints, descriptors);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (success) {
            std::cout << "Detected " << keypoints.size() << " features" << std::endl;
            std::cout << "Descriptor size: " << descriptors.rows << "x" << descriptors.cols << std::endl;
            std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
            
            cv::Mat outputAkaze = drawKeypointsColored(image, keypoints, "AKAZE Features");
            cv::imshow("AKAZE Features", outputAkaze);
        } else {
            std::cerr << "Failed to detect AKAZE features" << std::endl;
        }
    }

    std::cout << "\nPress any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

/**
 * @brief Test feature detector with video
 */
void testWithVideo(const std::string& videoPath) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing with video: " << videoPath << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create video source
    auto videoSource = std::make_shared<FilePlaybackSource>(videoPath);
    if (!videoSource->start()) {
        std::cerr << "Error: Could not open video: " << videoPath << std::endl;
        return;
    }

    std::cout << "Video size: " << videoSource->getWidth() << "x" 
              << videoSource->getHeight() << std::endl;
    std::cout << "Frame rate: " << videoSource->getFrameRate() << " fps" << std::endl;
    std::cout << "\nPress 'q' to quit, 'o' for ORB, 'a' for AKAZE" << std::endl;

    // Create detectors
    FeatureDetector orbDetector(FeatureDetectorType::ORB, 1000);
    FeatureDetector akazeDetector(FeatureDetectorType::AKAZE, 1000);
    
    bool useOrb = true;
    int frameCount = 0;
    double totalTime = 0.0;

    while (true) {
        // Get frame
        cv::Mat frame = videoSource->getFrame();
        if (frame.empty()) {
            break;
        }

        frameCount++;

        // Select detector
        FeatureDetector& detector = useOrb ? orbDetector : akazeDetector;
        std::string detectorName = useOrb ? "ORB" : "AKAZE";

        // Detect features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = detector.detectFeatures(frame, keypoints, descriptors);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double ms = duration.count() / 1000.0;
        totalTime += ms;

        if (success) {
            // Draw features
            cv::Mat output = drawKeypointsColored(frame, keypoints, detectorName);
            
            // Add performance info
            std::string perfInfo = "Time: " + std::to_string(ms) + " ms";
            cv::putText(output, perfInfo, cv::Point(10, 110),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            std::string avgInfo = "Avg: " + std::to_string(totalTime / frameCount) + " ms";
            cv::putText(output, avgInfo, cv::Point(10, 150),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("Feature Detection", output);
        }

        // Handle key press
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == 'o') {
            useOrb = true;
            std::cout << "Switched to ORB detector" << std::endl;
        } else if (key == 'a') {
            useOrb = false;
            std::cout << "Switched to AKAZE detector" << std::endl;
        }
    }

    std::cout << "\nProcessed " << frameCount << " frames" << std::endl;
    std::cout << "Average processing time: " << (totalTime / frameCount) << " ms/frame" << std::endl;

    videoSource->stop();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "METABALL Virtual Ad - Feature Detection Test" << std::endl;
    std::cout << "========================================" << std::endl;

    if (argc < 2) {
        std::cerr << "\nUsage: " << argv[0] << " <image_or_video_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " sample.jpg" << std::endl;
        std::cerr << "  " << argv[0] << " sample.mp4" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];

    // Check if input is image or video
    std::string extension = inputPath.substr(inputPath.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == "jpg" || extension == "jpeg" || extension == "png" || 
        extension == "bmp" || extension == "tiff") {
        testWithImage(inputPath);
    } else if (extension == "mp4" || extension == "avi" || extension == "mov" || 
               extension == "mkv") {
        testWithVideo(inputPath);
    } else {
        // Try to load as image first
        cv::Mat testImage = cv::imread(inputPath);
        if (!testImage.empty()) {
            testWithImage(inputPath);
        } else {
            // Try as video
            testWithVideo(inputPath);
        }
    }

    return 0;
}
