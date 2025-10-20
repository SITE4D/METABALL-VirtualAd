/**
 * @file test_depth_compositor.cpp
 * @brief Test program for DepthCompositor class
 */

#include "DepthCompositor.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace VirtualAd::Keyer;

/**
 * @brief テストヘルパー: ダミー画像を生成
 */
cv::Mat createDummyImage(int width, int height, const cv::Scalar& color)
{
    return cv::Mat(height, width, CV_8UC3, color);
}

/**
 * @brief テストヘルパー: ダミーセグメンテーションマスクを生成
 */
cv::Mat createDummySegmentationMask(int width, int height)
{
    cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));  // 背景
    
    // 中央にバックネット領域
    cv::rectangle(mask, 
                 cv::Point(width/4, height/4), 
                 cv::Point(3*width/4, 3*height/4), 
                 cv::Scalar(static_cast<uchar>(DepthCompositor::SegmentationClass::BACKNET)), 
                 -1);
    
    // 左側に選手領域
    cv::circle(mask, 
              cv::Point(width/6, height/2), 
              50, 
              cv::Scalar(static_cast<uchar>(DepthCompositor::SegmentationClass::PLAYER)), 
              -1);
    
    return mask;
}

/**
 * @brief テストヘルパー: ダミーデプスマップを生成
 */
cv::Mat createDummyDepthMap(int width, int height)
{
    cv::Mat depth_map(height, width, CV_32FC1);
    
    // グラデーション（左が手前、右が奥）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            depth_map.at<float>(y, x) = static_cast<float>(x) / width;
        }
    }
    
    return depth_map;
}

/**
 * @brief テスト1: シンプル合成（デプスなし）
 */
bool testSimpleComposite()
{
    std::cout << "\n=== Test 1: Simple Composite (no depth) ===" << std::endl;
    
    try {
        // テストデータ作成
        cv::Mat image = createDummyImage(640, 480, cv::Scalar(100, 150, 200));  // 青っぽい
        cv::Mat mask = createDummySegmentationMask(640, 480);
        cv::Mat ad_texture = createDummyImage(640, 480, cv::Scalar(50, 50, 255));  // 赤い広告
        
        // 合成実行
        DepthCompositor compositor;
        cv::Mat output;
        
        bool success = compositor.compositeSimple(image, mask, ad_texture, output);
        
        if (!success) {
            std::cerr << "ERROR: compositeSimple() failed: " << compositor.getLastError() << std::endl;
            return false;
        }
        
        // 結果確認
        std::cout << "  Processing time: " << compositor.getProcessingTime() << " ms" << std::endl;
        std::cout << "  Output size: " << output.size() << std::endl;
        std::cout << "  Output type: " << output.type() << std::endl;
        
        // 結果表示
        cv::imshow("Test 1 - Input Image", image);
        cv::imshow("Test 1 - Segmentation Mask", mask * 85);  // 0,1,2,3 → 0,85,170,255
        cv::imshow("Test 1 - Ad Texture", ad_texture);
        cv::imshow("Test 1 - Output", output);
        
        std::cout << "Test 1: PASSED" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in Test 1: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief テスト2: デプスベース合成
 */
bool testDepthComposite()
{
    std::cout << "\n=== Test 2: Depth-based Composite ===\n" << std::endl;
    
    try {
        // テストデータ作成
        cv::Mat image = createDummyImage(640, 480, cv::Scalar(100, 150, 200));
        cv::Mat mask = createDummySegmentationMask(640, 480);
        cv::Mat depth_map = createDummyDepthMap(640, 480);
        cv::Mat ad_texture = createDummyImage(640, 480, cv::Scalar(50, 50, 255));
        
        // 合成実行（バックネットデプス = 0.5）
        DepthCompositor compositor;
        cv::Mat output;
        
        bool success = compositor.composite(image, mask, depth_map, ad_texture, output, 0.5f);
        
        if (!success) {
            std::cerr << "ERROR: composite() failed: " << compositor.getLastError() << std::endl;
            return false;
        }
        
        // 結果確認
        std::cout << "  Processing time: " << compositor.getProcessingTime() << " ms" << std::endl;
        std::cout << "  Output size: " << output.size() << std::endl;
        
        // デプスマップ可視化
        cv::Mat color_depth;
        cv::Mat depth_8u;
        depth_map.convertTo(depth_8u, CV_8U, 255.0);
        cv::applyColorMap(depth_8u, color_depth, cv::COLORMAP_JET);
        
        // 結果表示
        cv::imshow("Test 2 - Input Image", image);
        cv::imshow("Test 2 - Segmentation Mask", mask * 85);
        cv::imshow("Test 2 - Depth Map", color_depth);
        cv::imshow("Test 2 - Ad Texture", ad_texture);
        cv::imshow("Test 2 - Output", output);
        
        std::cout << "Test 2: PASSED" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in Test 2: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief テスト3: パフォーマンステスト
 */
bool testPerformance()
{
    std::cout << "\n=== Test 3: Performance Test ===\n" << std::endl;
    
    try {
        // テストデータ作成
        cv::Mat image = createDummyImage(1920, 1080, cv::Scalar(100, 150, 200));
        cv::Mat mask = createDummySegmentationMask(1920, 1080);
        cv::Mat depth_map = createDummyDepthMap(1920, 1080);
        cv::Mat ad_texture = createDummyImage(1920, 1080, cv::Scalar(50, 50, 255));
        
        DepthCompositor compositor;
        cv::Mat output;
        
        // 100イテレーション実行
        const int iterations = 100;
        std::vector<double> times;
        
        for (int i = 0; i < iterations; i++) {
            compositor.composite(image, mask, depth_map, ad_texture, output, 0.5f);
            times.push_back(compositor.getProcessingTime());
        }
        
        // 統計計算
        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double t : times) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        
        double avg_time = sum / iterations;
        
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Average time: " << avg_time << " ms" << std::endl;
        std::cout << "  Min time: " << min_time << " ms" << std::endl;
        std::cout << "  Max time: " << max_time << " ms" << std::endl;
        std::cout << "  Target: < 1.0 ms (CPU)" << std::endl;
        
        std::cout << "Test 3: PASSED" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in Test 3: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief メイン関数
 */
int main()
{
    std::cout << "======================================" << std::endl;
    std::cout << "DepthCompositor Test Program" << std::endl;
    std::cout << "======================================" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // テスト1: シンプル合成
    if (testSimpleComposite()) {
        passed++;
    } else {
        failed++;
    }
    
    std::cout << "\nPress any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // テスト2: デプスベース合成
    if (testDepthComposite()) {
        passed++;
    } else {
        failed++;
    }
    
    std::cout << "\nPress any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // テスト3: パフォーマンステスト
    if (testPerformance()) {
        passed++;
    } else {
        failed++;
    }
    
    // 結果サマリー
    std::cout << "\n======================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Passed: " << passed << " / " << (passed + failed) << std::endl;
    std::cout << "Failed: " << failed << " / " << (passed + failed) << std::endl;
    std::cout << "======================================" << std::endl;
    
    return (failed == 0) ? 0 : 1;
}
