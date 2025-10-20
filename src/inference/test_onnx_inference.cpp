#include "ONNXInference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace metaball;

int main(int argc, char** argv) {
    std::cout << "=== ONNX Inference Test ===" << std::endl;
    
    // コマンドライン引数チェック
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/camera_pose_net.onnx data/test_image.jpg" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    
    // 画像読み込み
    std::cout << "\n1. Loading image: " << image_path << std::endl;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return 1;
    }
    std::cout << "   Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // ONNX推論オブジェクト作成
    std::cout << "\n2. Creating ONNX inference object..." << std::endl;
    ONNXInference inference;
    
    // モデル読み込み
    std::cout << "\n3. Loading ONNX model: " << model_path << std::endl;
    if (!inference.loadModel(model_path)) {
        std::cerr << "Error: Failed to load model: " << inference.getLastError() << std::endl;
        return 1;
    }
    
    // ウォームアップ推論（初回は遅い）
    std::cout << "\n4. Warm-up inference..." << std::endl;
    std::vector<float> warmup_pose;
    if (!inference.infer(image, warmup_pose)) {
        std::cerr << "Error: Warm-up inference failed: " << inference.getLastError() << std::endl;
        return 1;
    }
    
    // 推論実行（時間測定）
    std::cout << "\n5. Running inference (10 iterations)..." << std::endl;
    const int num_iterations = 10;
    std::vector<double> inference_times;
    
    for (int i = 0; i < num_iterations; ++i) {
        std::vector<float> pose;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = inference.infer(image, pose);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!success) {
            std::cerr << "Error: Inference failed at iteration " << i << ": " 
                      << inference.getLastError() << std::endl;
            return 1;
        }
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        inference_times.push_back(elapsed_ms);
        
        if (i == 0) {
            std::cout << "\n   Iteration 0 result:" << std::endl;
            std::cout << "     rvec: [" << pose[0] << ", " << pose[1] << ", " << pose[2] << "]" << std::endl;
            std::cout << "     tvec: [" << pose[3] << ", " << pose[4] << ", " << pose[5] << "]" << std::endl;
            std::cout << "     Time: " << elapsed_ms << " ms" << std::endl;
        }
    }
    
    // 統計計算
    double total_time = 0.0;
    double min_time = inference_times[0];
    double max_time = inference_times[0];
    
    for (double time : inference_times) {
        total_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    double avg_time = total_time / num_iterations;
    
    std::cout << "\n6. Performance Statistics:" << std::endl;
    std::cout << "   Average time: " << avg_time << " ms" << std::endl;
    std::cout << "   Min time:     " << min_time << " ms" << std::endl;
    std::cout << "   Max time:     " << max_time << " ms" << std::endl;
    std::cout << "   Throughput:   " << (1000.0 / avg_time) << " fps" << std::endl;
    
    // 目標パフォーマンスチェック（5ms以内）
    if (avg_time <= 5.0) {
        std::cout << "\n✓ PASS: Inference time meets target (<= 5ms)" << std::endl;
    } else {
        std::cout << "\n✗ WARNING: Inference time exceeds target (> 5ms)" << std::endl;
    }
    
    // 可視化（オプション）
    if (argc >= 4 && std::string(argv[3]) == "--visualize") {
        std::cout << "\n7. Visualizing result..." << std::endl;
        
        // 最終推論結果取得
        std::vector<float> final_pose;
        inference.infer(image, final_pose);
        
        // 画像にテキスト描画
        cv::Mat vis_image = image.clone();
        
        std::string rvec_text = "rvec: [" + std::to_string(final_pose[0]) + ", " + 
                                std::to_string(final_pose[1]) + ", " + 
                                std::to_string(final_pose[2]) + "]";
        std::string tvec_text = "tvec: [" + std::to_string(final_pose[3]) + ", " + 
                                std::to_string(final_pose[4]) + ", " + 
                                std::to_string(final_pose[5]) + "]";
        
        cv::putText(vis_image, rvec_text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(vis_image, tvec_text, cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("ONNX Inference Result", vis_image);
        std::cout << "   Press any key to close..." << std::endl;
        cv::waitKey(0);
    }
    
    std::cout << "\n=== Test completed successfully ===" << std::endl;
    
    return 0;
}
