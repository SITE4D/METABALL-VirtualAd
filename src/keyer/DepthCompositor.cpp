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
