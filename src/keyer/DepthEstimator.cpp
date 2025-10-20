/**
 * @file DepthEstimator.cpp
 * @brief Implementation of DepthEstimator class
 */

#include "DepthEstimator.h"
#include <iostream>
#include <chrono>

namespace VirtualAd {
namespace Keyer {

// コンストラクタ
DepthEstimator::DepthEstimator(int input_size)
    : is_loaded_(false),
      input_size_(input_size),
      inference_time_(0.0)
{
    // ONNX Runtime環境初期化
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DepthEstimator");
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        std::cout << "DepthEstimator initialized (input_size=" << input_size_ << ")" << std::endl;
    }
    catch (const std::exception& e) {
        last_error_ = std::string("Failed to initialize ONNX Runtime: ") + e.what();
        std::cerr << last_error_ << std::endl;
    }
}

// デストラクタ
DepthEstimator::~DepthEstimator()
{
    // ONNX Runtimeリソース解放（unique_ptrが自動的に解放）
    if (is_loaded_) {
        std::cout << "DepthEstimator destroyed (model was loaded)" << std::endl;
    }
}

// モデルがロード済みか確認
bool DepthEstimator::isLoaded() const
{
    return is_loaded_;
}

// 最後のエラーメッセージを取得
std::string DepthEstimator::getLastError() const
{
    return last_error_;
}

// 推論時間を取得
double DepthEstimator::getInferenceTime() const
{
    return inference_time_;
}

// デプスマップを可視化（静的メソッド）
void DepthEstimator::visualizeDepth(const cv::Mat& depth_map, 
                                   cv::Mat& color_depth,
                                   int colormap)
{
    // 入力チェック
    if (depth_map.empty() || depth_map.type() != CV_32FC1) {
        std::cerr << "ERROR: Invalid depth map (must be CV_32FC1)" << std::endl;
        return;
    }
    
    // 0.0-1.0 → 0-255に変換
    cv::Mat depth_8u;
    depth_map.convertTo(depth_8u, CV_8U, 255.0);
    
    // カラーマップ適用
    cv::applyColorMap(depth_8u, color_depth, colormap);
}

} // namespace Keyer
} // namespace VirtualAd
