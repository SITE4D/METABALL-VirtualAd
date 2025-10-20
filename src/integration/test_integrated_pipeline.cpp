/**
 * @file test_integrated_pipeline.cpp
 * @brief Test program for IntegratedPipeline
 */

#include "IntegratedPipeline.h"
#include "PipelineConfig.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace VirtualAd::Integration;

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "IntegratedPipeline Test Program" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // 設定作成
    // ========================================================================
    PipelineConfig config;
    config.single_thread_mode = true;  // シングルスレッドモードでテスト
    
    // ONNXモデルパス（存在しない場合はダミーデータ使用）
    config.camera_pose_model_path = "models/camera_pose.onnx";
    config.segmentation_model_path = "models/segmentation.onnx";
    config.depth_model_path = "models/depth.onnx";
    
    std::cout << "[Test] Configuration:" << std::endl;
    std::cout << "  Mode: SINGLE_THREAD" << std::endl;
    std::cout << "  Camera Pose Model: " << config.camera_pose_model_path << std::endl;
    std::cout << "  Segmentation Model: " << config.segmentation_model_path << std::endl;
    std::cout << "  Depth Model: " << config.depth_model_path << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // IntegratedPipeline初期化
    // ========================================================================
    std::cout << "[Test] Initializing IntegratedPipeline..." << std::endl;
    IntegratedPipeline pipeline;
    
    if (!pipeline.initialize(config)) {
        std::cerr << "[Test] ERROR: Failed to initialize pipeline" << std::endl;
        return 1;
    }
    
    std::cout << "[Test] Pipeline initialized successfully!" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // テスト画像作成（ダミー画像）
    // ========================================================================
    std::cout << "[Test] Creating test image (1920x1080)..." << std::endl;
    cv::Mat test_image(1080, 1920, CV_8UC3, cv::Scalar(100, 150, 200));
    
    // テスト用の模様を追加
    cv::circle(test_image, cv::Point(960, 540), 200, cv::Scalar(0, 255, 0), 5);
    cv::putText(test_image, "METABALL VirtualAd Test", 
                cv::Point(700, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
    
    std::cout << "[Test] Test image created: " << test_image.cols << "x" 
              << test_image.rows << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // フレーム処理テスト（複数フレーム）
    // ========================================================================
    const int num_test_frames = 5;
    std::cout << "[Test] Processing " << num_test_frames << " test frames..." << std::endl;
    std::cout << std::endl;
    
    std::vector<double> processing_times;
    
    for (int i = 0; i < num_test_frames; ++i) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "[Test] Frame " << (i + 1) << "/" << num_test_frames << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Mat output;
        if (pipeline.processFrame(test_image, output)) {
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            processing_times.push_back(elapsed);
            
            std::cout << "[Test] Frame processed successfully!" << std::endl;
            std::cout << "[Test] Output size: " << output.cols << "x" << output.rows << std::endl;
            std::cout << "[Test] Total time: " << elapsed << " ms" << std::endl;
            
            // 最初のフレームのみ保存
            if (i == 0) {
                std::string output_path = "output_integrated_pipeline_test.png";
                cv::imwrite(output_path, output);
                std::cout << "[Test] Output saved to: " << output_path << std::endl;
            }
        } else {
            std::cerr << "[Test] ERROR: Frame processing failed!" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // ========================================================================
    // 統計情報表示
    // ========================================================================
    std::cout << "========================================" << std::endl;
    std::cout << "Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto stats = pipeline.getStatistics();
    std::cout << "Total Frames: " << stats.total_frames << std::endl;
    std::cout << "Dropped Frames: " << stats.dropped_frames << std::endl;
    std::cout << "Error Frames: " << stats.error_frames << std::endl;
    std::cout << std::endl;
    
    std::cout << "Frame Time Statistics:" << std::endl;
    std::cout << "  Average: " << stats.avg_frame_time_ms << " ms" << std::endl;
    std::cout << "  Min: " << stats.min_frame_time_ms << " ms" << std::endl;
    std::cout << "  Max: " << stats.max_frame_time_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "FPS Statistics:" << std::endl;
    std::cout << "  Current: " << stats.current_fps << " fps" << std::endl;
    std::cout << "  Average: " << stats.avg_fps << " fps" << std::endl;
    std::cout << std::endl;
    
    // 処理時間の平均計算
    if (!processing_times.empty()) {
        double avg_time = 0.0;
        for (double t : processing_times) {
            avg_time += t;
        }
        avg_time /= processing_times.size();
        
        std::cout << "Measured Processing Time:" << std::endl;
        std::cout << "  Average: " << avg_time << " ms" << std::endl;
        std::cout << "  Theoretical FPS: " << (1000.0 / avg_time) << " fps" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
