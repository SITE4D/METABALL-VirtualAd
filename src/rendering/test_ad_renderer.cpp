/**
 * @file test_ad_renderer.cpp
 * @brief Test program for AdRenderer class
 */

#include "AdRenderer.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace VirtualAd::Rendering;

/**
 * @brief テストヘルパー: ダミーカメラ行列を生成
 */
cv::Mat createDummyCameraMatrix()
{
    // 仮想カメラ行列（1920x1080）
    // fx, fy: 焦点距離（ピクセル単位）
    // cx, cy: 画像中心（ピクセル単位）
    double fx = 1200.0;
    double fy = 1200.0;
    double cx = 960.0;
    double cy = 540.0;
    
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    );
    
    return camera_matrix;
}

/**
 * @brief テストヘルパー: ダミー歪み係数を生成
 */
cv::Mat createDummyDistCoeffs()
{
    // 歪みなし
    return cv::Mat::zeros(5, 1, CV_64FC1);
}

/**
 * @brief テストヘルパー: ダミーカメラポーズを生成
 */
void createDummyPose(cv::Mat& rvec, cv::Mat& tvec)
{
    // 回転なし、カメラから10m離れた位置
    rvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 10.0);
}

/**
 * @brief テストヘルパー: ダミーバックネット3D座標を生成
 */
std::vector<cv::Point3f> createDummyBacknet()
{
    // 10m × 5mのバックネット、カメラから10m離れた位置
    std::vector<cv::Point3f> corners = {
        cv::Point3f(-5.0f, 2.5f, 10.0f),  // 左上
        cv::Point3f( 5.0f, 2.5f, 10.0f),  // 右上
        cv::Point3f( 5.0f, -2.5f, 10.0f), // 右下
        cv::Point3f(-5.0f, -2.5f, 10.0f)  // 左下
    };
    return corners;
}

/**
 * @brief テストヘルパー: ダミー広告テクスチャを生成
 */
cv::Mat createDummyAdTexture()
{
    // 赤い広告（640x480）
    return cv::Mat(480, 640, CV_8UC3, cv::Scalar(50, 50, 255));
}

/**
 * @brief テストヘルパー: ダミー入力画像を生成
 */
cv::Mat createDummyImage()
{
    // 青っぽい背景（1920x1080）
    return cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(200, 150, 100));
}

/**
 * @brief テスト1: 初期化テスト
 */
bool testInitialization()
{
    std::cout << "\n=== Test 1: Initialization ===" << std::endl;
    
    try {
        // カメラ行列・歪み係数生成
        cv::Mat camera_matrix = createDummyCameraMatrix();
        cv::Mat dist_coeffs = createDummyDistCoeffs();
        
        // 初期化
        AdRenderer renderer;
        
        // 初期化前状態確認
        if (renderer.isInitialized()) {
            std::cerr << "ERROR: Renderer should not be initialized yet" << std::endl;
            return false;
        }
        
        // 初期化実行
        bool success = renderer.initialize(camera_matrix, dist_coeffs);
        
        if (!success) {
            std::cerr << "ERROR: initialize() failed: " << renderer.getLastError() << std::endl;
            return false;
        }
        
        // 初期化後状態確認
        if (!renderer.isInitialized()) {
            std::cerr << "ERROR: Renderer should be initialized" << std::endl;
            return false;
        }
        
        std::cout << "Test 1: PASSED" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in Test 1: " << e.what() << std::endl;
        return false;
    }
}

// 他のテスト関数は次のパートで実装
