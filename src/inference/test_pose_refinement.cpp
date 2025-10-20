#include "CameraPoseRefiner.h"
#include "../tracking/PnPSolver.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <memory>

using namespace metaball;
using namespace VirtualAd::Tracking;

int main(int argc, char** argv) {
    std::cout << "=== Camera Pose Refinement Test ===" << std::endl;
    
    // コマンドライン引数チェック
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [image_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/camera_pose_net.onnx data/test_image.jpg" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = (argc >= 3) ? argv[2] : "";
    
    // カメラ内部パラメータ設定（デフォルトHD）
    std::cout << "\n1. Setting up camera intrinsics..." << std::endl;
    CameraIntrinsics intrinsics = CameraIntrinsics::createDefaultHD();
    std::cout << "   Camera matrix: fx=" << intrinsics.fx << ", fy=" << intrinsics.fy 
              << ", cx=" << intrinsics.cx << ", cy=" << intrinsics.cy << std::endl;
    
    // PnPソルバー作成
    std::cout << "\n2. Creating PnP solver..." << std::endl;
    auto pnp_solver = std::make_shared<PnPSolver>(
        intrinsics,
        PnPSolver::Algorithm::ITERATIVE,
        true,  // Use RANSAC
        8.0f   // RANSAC threshold
    );
    
    // CameraPoseRefiner作成
    std::cout << "\n3. Creating CameraPoseRefiner..." << std::endl;
    CameraPoseRefiner refiner(pnp_solver);
    
    // AIモデル読み込み
    std::cout << "\n4. Loading AI model: " << model_path << std::endl;
    if (!refiner.loadModel(model_path)) {
        std::cerr << "Error: Failed to load model: " << refiner.getLastError() << std::endl;
        std::cerr << "Continuing with PNP_ONLY mode..." << std::endl;
        refiner.setMode(CameraPoseRefiner::Mode::PNP_ONLY);
    }
    
    // テスト用3D点群（バックネットの4隅を想定）
    std::cout << "\n5. Creating test data..." << std::endl;
    std::vector<cv::Point3f> object_points = {
        cv::Point3f(-1.0f,  1.0f, 0.0f),  // 左上
        cv::Point3f( 1.0f,  1.0f, 0.0f),  // 右上
        cv::Point3f( 1.0f, -1.0f, 0.0f),  // 右下
        cv::Point3f(-1.0f, -1.0f, 0.0f)   // 左下
    };
    
    // テスト用2D点群（画像上の対応点）
    std::vector<cv::Point2f> image_points = {
        cv::Point2f(400.0f, 200.0f),
        cv::Point2f(880.0f, 200.0f),
        cv::Point2f(880.0f, 680.0f),
        cv::Point2f(400.0f, 680.0f)
    };
    
    std::cout << "   Object points: " << object_points.size() << " points" << std::endl;
    std::cout << "   Image points:  " << image_points.size() << " points" << std::endl;
    
    // テスト画像準備
    cv::Mat test_image;
    if (!image_path.empty()) {
        test_image = cv::imread(image_path);
        if (test_image.empty()) {
            std::cerr << "Warning: Failed to load image, creating dummy image" << std::endl;
            test_image = cv::Mat::zeros(720, 1280, CV_8UC3);
        }
    } else {
        std::cout << "   No image provided, creating dummy image" << std::endl;
        test_image = cv::Mat::zeros(720, 1280, CV_8UC3);
    }
    
    std::cout << "   Test image size: " << test_image.cols << "x" << test_image.rows << std::endl;
    
    // モード別テスト
    std::cout << "\n6. Testing different refinement modes..." << std::endl;
    
    const char* mode_names[] = {"PNP_ONLY", "AI_ONLY", "BLENDED"};
    CameraPoseRefiner::Mode modes[] = {
        CameraPoseRefiner::Mode::PNP_ONLY,
        CameraPoseRefiner::Mode::AI_ONLY,
        CameraPoseRefiner::Mode::BLENDED
    };
    
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n   Mode: " << mode_names[i] << std::endl;
        refiner.setMode(modes[i]);
        
        // AI_ONLYモードでモデル未ロードの場合はスキップ
        if (modes[i] == CameraPoseRefiner::Mode::AI_ONLY && !refiner.isModelLoaded()) {
            std::cout << "     Skipped (AI model not loaded)" << std::endl;
            continue;
        }
        
        CameraPose pose;
        std::vector<uchar> inlier_mask;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = refiner.refinePose(test_image, object_points, image_points, pose, inlier_mask);
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (success) {
            std::cout << "     Success!" << std::endl;
            std::cout << "     rvec: [" << pose.rvec.at<double>(0) << ", " 
                      << pose.rvec.at<double>(1) << ", " 
                      << pose.rvec.at<double>(2) << "]" << std::endl;
            std::cout << "     tvec: [" << pose.tvec.at<double>(0) << ", " 
                      << pose.tvec.at<double>(1) << ", " 
                      << pose.tvec.at<double>(2) << "]" << std::endl;
            
            if (modes[i] != CameraPoseRefiner::Mode::AI_ONLY) {
                std::cout << "     PnP error: " << refiner.getLastPnPError() << " pixels" << std::endl;
                int inlier_count = std::count(inlier_mask.begin(), inlier_mask.end(), 1);
                std::cout << "     Inliers: " << inlier_count << "/" << object_points.size() << std::endl;
            }
            
            std::cout << "     Processing time: " << elapsed_ms << " ms" << std::endl;
        } else {
            std::cout << "     Failed: " << refiner.getLastError() << std::endl;
        }
    }
    
    // ブレンディング係数テスト
    if (refiner.isModelLoaded()) {
        std::cout << "\n7. Testing blend alpha variations..." << std::endl;
        refiner.setMode(CameraPoseRefiner::Mode::BLENDED);
        
        float alphas[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
        for (float alpha : alphas) {
            refiner.setBlendAlpha(alpha);
            
            CameraPose pose;
            std::vector<uchar> inlier_mask;
            
            if (refiner.refinePose(test_image, object_points, image_points, pose, inlier_mask)) {
                std::cout << "   Alpha=" << alpha << ": rvec=[" 
                          << pose.rvec.at<double>(0) << ", " 
                          << pose.rvec.at<double>(1) << ", " 
                          << pose.rvec.at<double>(2) << "]" << std::endl;
            }
        }
    }
    
    // パフォーマンステスト
    std::cout << "\n8. Performance test (100 iterations)..." << std::endl;
    refiner.setMode(CameraPoseRefiner::Mode::BLENDED);
    refiner.setBlendAlpha(0.5f);
    
    const int num_iterations = 100;
    std::vector<double> times;
    
    for (int i = 0; i < num_iterations; ++i) {
        CameraPose pose;
        std::vector<uchar> inlier_mask;
        
        auto start = std::chrono::high_resolution_clock::now();
        refiner.refinePose(test_image, object_points, image_points, pose, inlier_mask);
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed_ms);
    }
    
    // 統計計算
    double total = 0.0, min_time = times[0], max_time = times[0];
    for (double t : times) {
        total += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    
    double avg_time = total / num_iterations;
    
    std::cout << "   Average: " << avg_time << " ms" << std::endl;
    std::cout << "   Min:     " << min_time << " ms" << std::endl;
    std::cout << "   Max:     " << max_time << " ms" << std::endl;
    std::cout << "   FPS:     " << (1000.0 / avg_time) << std::endl;
    
    // 目標達成チェック
    if (avg_time <= 10.0) {
        std::cout << "\n✓ PASS: Processing time meets target (<= 10ms)" << std::endl;
    } else {
        std::cout << "\n✗ WARNING: Processing time exceeds target (> 10ms)" << std::endl;
    }
    
    std::cout << "\n=== Test completed successfully ===" << std::endl;
    
    return 0;
}
