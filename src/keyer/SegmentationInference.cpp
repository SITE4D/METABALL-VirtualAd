/**
 * @file SegmentationInference.cpp
 * @brief Implementation of SegmentationInference class
 */

#include "SegmentationInference.h"
#include <iostream>
#include <chrono>

namespace VirtualAd {
namespace Keyer {

// コンストラクタ
SegmentationInference::SegmentationInference(int input_size)
    : is_loaded_(false),
      input_size_(input_size),
      inference_time_(0.0)
{
    // ONNX Runtime環境初期化
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SegmentationInference");
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        std::cout << "SegmentationInference initialized (input_size=" << input_size_ << ")" << std::endl;
    }
    catch (const std::exception& e) {
        last_error_ = std::string("Failed to initialize ONNX Runtime: ") + e.what();
        std::cerr << last_error_ << std::endl;
    }
}

// デストラクタ
SegmentationInference::~SegmentationInference()
{
    // ONNX Runtimeリソース解放（unique_ptrが自動的に解放）
    if (is_loaded_) {
        std::cout << "SegmentationInference destroyed (model was loaded)" << std::endl;
    }
}

// モデルがロード済みか確認
bool SegmentationInference::isLoaded() const
{
    return is_loaded_;
}

// 最後のエラーメッセージを取得
std::string SegmentationInference::getLastError() const
{
    return last_error_;
}

// 推論時間を取得
double SegmentationInference::getInferenceTime() const
{
    return inference_time_;
}

// マスクをカラー画像に変換（静的メソッド）
void SegmentationInference::maskToColor(const cv::Mat& mask, cv::Mat& color_mask)
{
    // 出力画像作成
    color_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
    
    // クラスごとに色を割り当て
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            uchar class_id = mask.at<uchar>(y, x);
            
            if (class_id < 4) {
                cv::Scalar color = CLASS_COLORS[class_id];
                color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(color[0]),
                    static_cast<uchar>(color[1]),
                    static_cast<uchar>(color[2])
                );
            }
        }
    }
}

// マスクをオーバーレイ表示（静的メソッド）
void SegmentationInference::overlayMask(const cv::Mat& image, const cv::Mat& mask,
                                       cv::Mat& output, float alpha)
{
    // 入力チェック
    if (image.size() != mask.size()) {
        std::cerr << "ERROR: Image and mask size mismatch" << std::endl;
        output = image.clone();
        return;
    }
    
    if (alpha < 0.0f || alpha > 1.0f) {
        std::cerr << "WARNING: Alpha should be in [0.0, 1.0], clamping" << std::endl;
        alpha = std::max(0.0f, std::min(1.0f, alpha));
    }
    
    // カラーマスク作成
    cv::Mat color_mask;
    maskToColor(mask, color_mask);
    
    // オーバーレイ合成
    cv::addWeighted(image, 1.0f - alpha, color_mask, alpha, 0.0, output);
}

} // namespace Keyer
} // namespace VirtualAd
