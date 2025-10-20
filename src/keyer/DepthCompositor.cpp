/**
 * @file DepthCompositor.cpp
 * @brief Implementation of DepthCompositor class
 */

#include "DepthCompositor.h"
#include <iostream>
#include <chrono>

namespace VirtualAd {
namespace Keyer {

// コンストラクタ
DepthCompositor::DepthCompositor()
    : depth_threshold_(0.1f),
      processing_time_(0.0)
{
    std::cout << "DepthCompositor initialized" << std::endl;
}

// デストラクタ
DepthCompositor::~DepthCompositor()
{
    std::cout << "DepthCompositor destroyed" << std::endl;
}

// デプス閾値を設定
void DepthCompositor::setDepthThreshold(float threshold)
{
    if (threshold < 0.0f || threshold > 1.0f) {
        std::cerr << "WARNING: Depth threshold should be in [0.0, 1.0], clamping" << std::endl;
        threshold = std::max(0.0f, std::min(1.0f, threshold));
    }
    depth_threshold_ = threshold;
}

// デプス閾値を取得
float DepthCompositor::getDepthThreshold() const
{
    return depth_threshold_;
}

// 最後のエラーメッセージを取得
std::string DepthCompositor::getLastError() const
{
    return last_error_;
}

// 処理時間を取得
double DepthCompositor::getProcessingTime() const
{
    return processing_time_;
}

// 入力画像を検証
bool DepthCompositor::validateInputs(const cv::Mat& image,
                                    const cv::Mat& segmentation_mask,
                                    const cv::Mat& depth_map,
                                    const cv::Mat& ad_texture)
{
    // 入力画像チェック
    if (image.empty()) {
        last_error_ = "Empty input image";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (image.type() != CV_8UC3) {
        last_error_ = "Input image must be CV_8UC3 (BGR)";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // セグメンテーションマスクチェック
    if (segmentation_mask.empty()) {
        last_error_ = "Empty segmentation mask";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (segmentation_mask.type() != CV_8UC1) {
        last_error_ = "Segmentation mask must be CV_8UC1";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (segmentation_mask.size() != image.size()) {
        last_error_ = "Segmentation mask size mismatch";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // デプスマップチェック（オプション）
    if (!depth_map.empty()) {
        if (depth_map.type() != CV_32FC1) {
            last_error_ = "Depth map must be CV_32FC1";
            std::cerr << "ERROR: " << last_error_ << std::endl;
            return false;
        }
        
        if (depth_map.size() != image.size()) {
            last_error_ = "Depth map size mismatch";
            std::cerr << "ERROR: " << last_error_ << std::endl;
            return false;
        }
    }
    
    // 広告テクスチャチェック
    if (ad_texture.empty()) {
        last_error_ = "Empty ad texture";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (ad_texture.type() != CV_8UC3) {
        last_error_ = "Ad texture must be CV_8UC3 (BGR)";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    return true;
}

// シンプル合成を実行（デプス情報なし）
bool DepthCompositor::compositeSimple(const cv::Mat& image,
                                     const cv::Mat& segmentation_mask,
                                     const cv::Mat& ad_texture,
                                     cv::Mat& output)
{
    // 入力検証（デプスマップなし）
    cv::Mat empty_depth;
    if (!validateInputs(image, segmentation_mask, empty_depth, ad_texture)) {
        return false;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 広告テクスチャをリサイズ
        cv::Mat resized_ad;
        resizeAdTexture(ad_texture, image.size(), resized_ad);
        
        // 出力画像作成
        output = image.clone();
        
        // ピクセル単位合成
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                uchar class_id = segmentation_mask.at<uchar>(y, x);
                
                // バックネットクラスの場合のみ広告で置き換え
                if (class_id == static_cast<uchar>(SegmentationClass::BACKNET)) {
                    output.at<cv::Vec3b>(y, x) = resized_ad.at<cv::Vec3b>(y, x);
                }
                // その他（選手、審判、背景）は元画像のまま
            }
        }
        
        // 処理時間計測
        auto end_time = std::chrono::high_resolution_clock::now();
        processing_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        last_error_.clear();
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = std::string("Simple compositing failed: ") + e.what();
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
}

// 広告テクスチャをリサイズ（静的メソッド）
void DepthCompositor::resizeAdTexture(const cv::Mat& ad_texture,
                                     const cv::Size& target_size,
                                     cv::Mat& resized_ad)
{
    if (ad_texture.empty()) {
        std::cerr << "ERROR: Empty ad texture" << std::endl;
        return;
    }
    
    if (ad_texture.size() == target_size) {
        resized_ad = ad_texture.clone();
        return;
    }
    
    // アスペクト比を維持してリサイズ
    cv::resize(ad_texture, resized_ad, target_size, 0, 0, cv::INTER_LINEAR);
}

} // namespace Keyer
} // namespace VirtualAd
