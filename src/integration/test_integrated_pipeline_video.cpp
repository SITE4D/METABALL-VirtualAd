/**
 * @file test_integrated_pipeline_video.cpp
 * @brief Video file processing test for IntegratedPipeline
 */

#include "IntegratedPipeline.h"
#include "PipelineConfig.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace VirtualAd::Integration;

// 関数プロトタイプ宣言
int runDummyMode();

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "IntegratedPipeline Video Test Program" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // コマンドライン引数チェック
    // ========================================================================
    std::string input_video = "data/samples/test_video.mp4";  // デフォルト
    std::string output_video = "output_integrated_pipeline_video.mp4";
    
    if (argc >= 2) {
        input_video = argv[1];
    }
    if (argc >= 3) {
        output_video = argv[2];
    }
    
    std::cout << "[Test] Input Video: " << input_video << std::endl;
    std::cout << "[Test] Output Video: " << output_video << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // 動画ファイル読み込み
    // ========================================================================
    std::cout << "[Test] Opening video file..." << std::endl;
    cv::VideoCapture cap(input_video);
    
    if (!cap.isOpened()) {
        std::cerr << "[Test] ERROR: Could not open video file: " << input_video << std::endl;
        std::cout << std::endl;
        std::cout << "Usage: " << argv[0] << " [input_video.mp4] [output_video.mp4]" << std::endl;
        std::cout << std::endl;
        std::cout << "Note: If video file not found, processing dummy frames..." << std::endl;
        std::cout << std::endl;
        
        // ダミーフレーム処理モード
        return runDummyMode();
    }
    
    // 動画情報取得
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "[Test] Video Information:" << std::endl;
    std::cout << "  Size: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Total Frames: " << total_frames << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // 動画ライター初期化
    // ========================================================================
    std::cout << "[Test] Initializing video writer..." << std::endl;
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(output_video, fourcc, fps, cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        std::cerr << "[Test] ERROR: Could not open video writer: " << output_video << std::endl;
        return 1;
    }
    std::cout << "[Test] Video writer initialized" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // IntegratedPipeline初期化
    // ========================================================================
    std::cout << "[Test] Initializing IntegratedPipeline..." << std::endl;
    PipelineConfig config;
    config.single_thread_mode = true;
    config.camera_pose_model_path = "models/camera_pose.onnx";
    config.segmentation_model_path = "models/segmentation.onnx";
    config.depth_model_path = "models/depth.onnx";
    
    IntegratedPipeline pipeline;
    if (!pipeline.initialize(config)) {
        std::cerr << "[Test] ERROR: Failed to initialize pipeline" << std::endl;
        return 1;
    }
    std::cout << "[Test] Pipeline initialized successfully!" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // フレーム処理ループ
    // ========================================================================
    std::cout << "[Test] Processing video frames..." << std::endl;
    std::cout << std::endl;
    
    cv::Mat frame, output;
    int processed_frames = 0;
    int failed_frames = 0;
    std::vector<double> processing_times;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (cap.read(frame)) {
        processed_frames++;
        
        // 進捗表示（10フレームごと）
        if (processed_frames % 10 == 0 || processed_frames == 1) {
            std::cout << "[Test] Processing frame " << processed_frames 
                      << "/" << total_frames << "..." << std::endl;
        }
        
        // フレーム処理
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        if (pipeline.processFrame(frame, output)) {
            auto frame_end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(
                frame_end - frame_start).count();
            processing_times.push_back(elapsed);
            
            // 出力動画に書き込み
            writer.write(output);
        } else {
            std::cerr << "[Test] WARNING: Frame " << processed_frames 
                      << " processing failed!" << std::endl;
            failed_frames++;
            
            // 処理失敗時は元フレームを書き込み
            writer.write(frame);
        }
        
        // 100フレームで停止（テスト用）
        if (processed_frames >= 100) {
            std::cout << "[Test] Reached 100 frames limit (test mode)" << std::endl;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    cap.release();
    writer.release();
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processing Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processed Frames: " << processed_frames << std::endl;
    std::cout << "Failed Frames: " << failed_frames << std::endl;
    std::cout << "Success Rate: " << (100.0 * (processed_frames - failed_frames) / processed_frames) 
              << "%" << std::endl;
    std::cout << std::endl;
    
    // 処理時間統計
    if (!processing_times.empty()) {
        double avg_time = 0.0;
        double min_time = processing_times[0];
        double max_time = processing_times[0];
        
        for (double t : processing_times) {
            avg_time += t;
            if (t < min_time) min_time = t;
            if (t > max_time) max_time = t;
        }
        avg_time /= processing_times.size();
        
        std::cout << "Processing Time Statistics:" << std::endl;
        std::cout << "  Average: " << avg_time << " ms" << std::endl;
        std::cout << "  Min: " << min_time << " ms" << std::endl;
        std::cout << "  Max: " << max_time << " ms" << std::endl;
        std::cout << "  Theoretical FPS: " << (1000.0 / avg_time) << " fps" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Total Processing Time: " << total_time << " seconds" << std::endl;
    std::cout << "Actual FPS: " << (processed_frames / total_time) << " fps" << std::endl;
    std::cout << std::endl;
    
    // パイプライン統計
    auto stats = pipeline.getStatistics();
    std::cout << "Pipeline Statistics:" << std::endl;
    std::cout << "  Total Frames: " << stats.total_frames << std::endl;
    std::cout << "  Dropped Frames: " << stats.dropped_frames << std::endl;
    std::cout << "  Error Frames: " << stats.error_frames << std::endl;
    std::cout << "  Average Frame Time: " << stats.avg_total_time_ms << " ms" << std::endl;
    std::cout << "  Average FPS: " << stats.avg_fps << " fps" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Output saved to: " << output_video << std::endl;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

// ダミーフレーム処理モード（動画ファイルがない場合）
int runDummyMode() {
    std::cout << "[Test] Running in DUMMY MODE (no video file)" << std::endl;
    std::cout << "[Test] Processing 30 dummy frames..." << std::endl;
    std::cout << std::endl;
    
    // IntegratedPipeline初期化
    VirtualAd::Integration::PipelineConfig config;
    config.single_thread_mode = true;
    
    VirtualAd::Integration::IntegratedPipeline pipeline;
    if (!pipeline.initialize(config)) {
        std::cerr << "[Test] ERROR: Failed to initialize pipeline" << std::endl;
        return 1;
    }
    
    // ダミー画像作成
    cv::Mat dummy_frame(1080, 1920, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::circle(dummy_frame, cv::Point(960, 540), 200, cv::Scalar(0, 255, 0), 5);
    cv::putText(dummy_frame, "DUMMY FRAME", cv::Point(700, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
    
    // 30フレーム処理
    for (int i = 0; i < 30; ++i) {
        cv::Mat output;
        if (pipeline.processFrame(dummy_frame, output)) {
            if (i % 10 == 0) {
                std::cout << "[Test] Processed frame " << (i+1) << "/30" << std::endl;
            }
        }
    }
    
    auto stats = pipeline.getStatistics();
    std::cout << std::endl;
    std::cout << "Dummy Mode Statistics:" << std::endl;
    std::cout << "  Total Frames: " << stats.total_frames << std::endl;
    std::cout << "  Average Frame Time: " << stats.avg_total_time_ms << " ms" << std::endl;
    std::cout << "  Average FPS: " << stats.avg_fps << " fps" << std::endl;
    std::cout << std::endl;
    std::cout << "Test completed!" << std::endl;
    
    return 0;
}
