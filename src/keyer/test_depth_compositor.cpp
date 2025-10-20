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

// 他のテスト関数は次のパートで実装
