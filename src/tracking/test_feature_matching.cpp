#include "tracking/FeatureDetector.h"
#include "tracking/FeatureMatcher.h"
#include "core/FilePlaybackSource.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace VirtualAd;
using namespace VirtualAd::Tracking;

/**
 * @brief Draw matches between two frames
 */
cv::Mat drawMatchesWithStats(const cv::Mat& img1, 
                             const std::vector<cv::KeyPoint>& keypoints1,
                             const cv::Mat& img2, 
                             const std::vector<cv::KeyPoint>& keypoints2,
                             const std::vector<cv::DMatch>& matches,
                             int inlierCount,
                             float inlierRatio,
                             double processingTime) {
    cv::Mat output;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, output,
                   cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Add statistics
    std::string matchesStr = "Matches: " + std::to_string(matches.size());
    std::string inliersStr = "Inliers: " + std::to_string(inlierCount) + 
                            " (" + std::to_string(static_cast<int>(inlierRatio * 100)) + "%)";
    std::string timeStr = "Time: " + std::to_string(processingTime) + " ms";
    
    cv::putText(output, matchesStr, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(output, inliersStr, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(output, timeStr, cv::Point(10, 90),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    return output;
}

/**
 * @brief Apply homography and draw warped result
 */
cv::Mat visualizeHomography(const cv::Mat& img1, 
                           const cv::Mat& img2, 
                           const cv::Mat& homography) {
    if (homography.empty()) {
        return img2.clone();
    }
    
    // Warp img1 to img2's perspective
    cv::Mat warped;
    cv::warpPerspective(img1, warped, homography, img2.size());
    
    // Blend with img2
    cv::Mat blended;
    cv::addWeighted(warped, 0.5, img2, 0.5, 0, blended);
    
    // Draw border of img1 warped to img2
    std::vector<cv::Point2f> corners1(4);
    corners1[0] = cv::Point2f(0, 0);
    corners1[1] = cv::Point2f(static_cast<float>(img1.cols), 0);
    corners1[2] = cv::Point2f(static_cast<float>(img1.cols), static_cast<float>(img1.rows));
    corners1[3] = cv::Point2f(0, static_cast<float>(img1.rows));
    
    std::vector<cv::Point2f> corners2;
    cv::perspectiveTransform(corners1, corners2, homography);
    
    // Draw warped border
    for (size_t i = 0; i < corners2.size(); i++) {
        cv::line(blended, corners2[i], corners2[(i+1) % corners2.size()], 
                cv::Scalar(0, 255, 0), 2);
    }
    
    return blended;
}

/**
 * @brief Test feature matching with video
 */
void testWithVideo(const std::string& videoPath) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing Feature Matching with video: " << videoPath << std::endl;
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
    std::cout << "\nControls:" << std::endl;
    std::cout << "  'q' / ESC - Quit" << std::endl;
    std::cout << "  'o' - ORB detector" << std::endl;
    std::cout << "  'a' - AKAZE detector" << std::endl;
    std::cout << "  'h' - Toggle homography visualization" << std::endl;
    std::cout << "  SPACE - Pause/Resume\n" << std::endl;

    // Create detector and matcher
    FeatureDetector orbDetector(FeatureDetectorType::ORB, 1000);
    FeatureDetector akazeDetector(FeatureDetectorType::AKAZE, 1000);
    FeatureMatcher matcher(MatcherType::BRUTE_FORCE, 0.75f, 3.0f);
    
    bool useOrb = true;
    bool showHomography = false;
    bool paused = false;
    
    // Previous frame data
    cv::Mat prevFrame;
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Mat prevDescriptors;
    
    int frameCount = 0;
    double totalTime = 0.0;
    int totalMatches = 0;
    int totalInliers = 0;

    // Get first frame
    prevFrame = videoSource->getFrame();
    if (prevFrame.empty()) {
        std::cerr << "Error: Could not read first frame" << std::endl;
        return;
    }
    
    FeatureDetector& detector = useOrb ? orbDetector : akazeDetector;
    detector.detectFeatures(prevFrame, prevKeypoints, prevDescriptors);

    while (true) {
        if (!paused) {
            // Get current frame
            cv::Mat currFrame = videoSource->getFrame();
            if (currFrame.empty()) {
                break;
            }

            frameCount++;

            // Detect features in current frame
            std::vector<cv::KeyPoint> currKeypoints;
            cv::Mat currDescriptors;
            
            auto detectStart = std::chrono::high_resolution_clock::now();
            detector.detectFeatures(currFrame, currKeypoints, currDescriptors);
            auto detectEnd = std::chrono::high_resolution_clock::now();

            // Match features and estimate homography
            cv::Mat homography;
            std::vector<cv::DMatch> inlierMatches;
            
            auto matchStart = std::chrono::high_resolution_clock::now();
            bool success = matcher.matchAndEstimateHomography(
                prevKeypoints, prevDescriptors,
                currKeypoints, currDescriptors,
                homography, inlierMatches);
            auto matchEnd = std::chrono::high_resolution_clock::now();

            // Calculate timing
            auto detectDuration = std::chrono::duration_cast<std::chrono::microseconds>(detectEnd - detectStart);
            auto matchDuration = std::chrono::duration_cast<std::chrono::microseconds>(matchEnd - matchStart);
            double totalMs = (detectDuration.count() + matchDuration.count()) / 1000.0;
            totalTime += totalMs;

            if (success) {
                totalMatches += static_cast<int>(inlierMatches.size());
                totalInliers += matcher.getInlierCount();

                // Visualize
                cv::Mat output;
                if (showHomography) {
                    output = visualizeHomography(prevFrame, currFrame, homography);
                    
                    // Add statistics to homography view
                    std::string inliersStr = "Inliers: " + std::to_string(matcher.getInlierCount()) + 
                                           " (" + std::to_string(static_cast<int>(matcher.getInlierRatio() * 100)) + "%)";
                    std::string timeStr = "Time: " + std::to_string(totalMs) + " ms";
                    
                    cv::putText(output, inliersStr, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                    cv::putText(output, timeStr, cv::Point(10, 60),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                } else {
                    output = drawMatchesWithStats(prevFrame, prevKeypoints, 
                                                 currFrame, currKeypoints,
                                                 inlierMatches,
                                                 matcher.getInlierCount(),
                                                 matcher.getInlierRatio(),
                                                 totalMs);
                }
                
                cv::imshow("Feature Matching", output);
            } else {
                std::cerr << "Frame " << frameCount << ": Matching failed" << std::endl;
            }

            // Update previous frame
            prevFrame = currFrame.clone();
            prevKeypoints = currKeypoints;
            prevDescriptors = currDescriptors.clone();
        } else {
            // Paused - just wait
            cv::waitKey(10);
        }

        // Handle key press
        int key = cv::waitKey(paused ? 0 : 1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == 'o') {
            useOrb = true;
            detector = orbDetector;
            std::cout << "Switched to ORB detector" << std::endl;
        } else if (key == 'a') {
            useOrb = false;
            detector = akazeDetector;
            std::cout << "Switched to AKAZE detector" << std::endl;
        } else if (key == 'h') {
            showHomography = !showHomography;
            std::cout << "Homography visualization: " << (showHomography ? "ON" : "OFF") << std::endl;
        } else if (key == ' ') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << std::endl;
        }
    }

    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processed frames: " << frameCount << std::endl;
    std::cout << "Average processing time: " << (totalTime / frameCount) << " ms/frame" << std::endl;
    std::cout << "Average matches: " << (totalMatches / frameCount) << std::endl;
    std::cout << "Average inliers: " << (totalInliers / frameCount) << std::endl;
    std::cout << "Average inlier ratio: " 
              << (100.0 * totalInliers / totalMatches) << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    videoSource->stop();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "METABALL Virtual Ad - Feature Matching Test" << std::endl;
    std::cout << "========================================" << std::endl;

    if (argc < 2) {
        std::cerr << "\nUsage: " << argv[0] << " <video_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " sample.mp4" << std::endl;
        return 1;
    }

    std::string videoPath = argv[1];
    testWithVideo(videoPath);

    return 0;
}
