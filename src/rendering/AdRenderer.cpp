/**
 * @file AdRenderer.cpp
 * @brief Implementation of AdRenderer class
 */

#include "AdRenderer.h"
#include <iostream>
#include <chrono>

namespace VirtualAd {
namespace Rendering {

// コンストラクタ
AdRenderer::AdRenderer()
    : blend_mode_(BlendMode::ALPHA_BLEND),
      alpha_(0.8f),
      processing_time_(0.0),
      is_initialized_(false)
{
    std::cout << "AdRenderer initialized" << std::endl;
}

// デストラクタ
AdRenderer::~AdRenderer()
{
    std::cout << "AdRenderer destroyed" << std::endl;
}

// 初期化
bool AdRenderer::initialize(const cv::Mat& camera_matrix, 
                           const cv::Mat& dist_coeffs)
{
    // カメラ行列検証
    if (camera_matrix.empty() || camera_matrix.type() != CV_64FC1) {
        last_error_ = "Invalid camera matrix (must be CV_64FC1)";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        last_error_ = "Invalid camera matrix size (must be 3x3)";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // 歪み係数検証（オプション）
    if (!dist_coeffs.empty()) {
        if (dist_coeffs.type() != CV_64FC1) {
            last_error_ = "Invalid distortion coefficients (must be CV_64FC1)";
            std::cerr << "ERROR: " << last_error_ << std::endl;
            return false;
        }
        
        int num_coeffs = dist_coeffs.rows * dist_coeffs.cols;
        if (num_coeffs != 4 && num_coeffs != 5 && num_coeffs != 8 && 
            num_coeffs != 12 && num_coeffs != 14) {
            last_error_ = "Invalid distortion coefficients count";
            std::cerr << "ERROR: " << last_error_ << std::endl;
            return false;
        }
    }
    
    // コピー
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
    
    is_initialized_ = true;
    last_error_.clear();
    
    std::cout << "AdRenderer initialized successfully" << std::endl;
    return true;
}

// バックネット3D平面設定
void AdRenderer::setBacknetPlane(const std::vector<cv::Point3f>& corners_3d)
{
    if (corners_3d.size() != 4) {
        std::cerr << "WARNING: Expected 4 corners, got " << corners_3d.size() << std::endl;
    }
    
    backnet_corners_3d_ = corners_3d;
    std::cout << "Backnet plane set with " << corners_3d.size() << " corners" << std::endl;
}

// 広告テクスチャ設定
bool AdRenderer::setAdTexture(const cv::Mat& ad_texture)
{
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
    
    ad_texture_ = ad_texture.clone();
    last_error_.clear();
    
    std::cout << "Ad texture set: " << ad_texture_.size() << std::endl;
    return true;
}

// ブレンディングモード設定
void AdRenderer::setBlendMode(BlendMode mode)
{
    blend_mode_ = mode;
}

// アルファ値設定
void AdRenderer::setAlpha(float alpha)
{
    if (alpha < 0.0f || alpha > 1.0f) {
        std::cerr << "WARNING: Alpha should be in [0.0, 1.0], clamping" << std::endl;
        alpha = std::max(0.0f, std::min(1.0f, alpha));
    }
    alpha_ = alpha;
}

// 処理時間取得
double AdRenderer::getProcessingTime() const
{
    return processing_time_;
}

// 最後のエラーメッセージ取得
std::string AdRenderer::getLastError() const
{
    return last_error_;
}

// 初期化状態確認
bool AdRenderer::isInitialized() const
{
    return is_initialized_;
}

// 入力検証
bool AdRenderer::validateInputs(const cv::Mat& image,
                               const cv::Mat& rvec,
                               const cv::Mat& tvec)
{
    // 初期化確認
    if (!is_initialized_) {
        last_error_ = "AdRenderer not initialized";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // 入力画像確認
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
    
    // rvec確認
    if (rvec.empty()) {
        last_error_ = "Empty rotation vector";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (rvec.rows * rvec.cols != 3) {
        last_error_ = "Rotation vector must be 3x1 or 1x3";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // tvec確認
    if (tvec.empty()) {
        last_error_ = "Empty translation vector";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    if (tvec.rows * tvec.cols != 3) {
        last_error_ = "Translation vector must be 3x1 or 1x3";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // バックネット3D座標確認
    if (backnet_corners_3d_.size() != 4) {
        last_error_ = "Backnet plane not set (need 4 corners)";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    // 広告テクスチャ確認
    if (ad_texture_.empty()) {
        last_error_ = "Ad texture not set";
        std::cerr << "ERROR: " << last_error_ << std::endl;
        return false;
    }
    
    return true;
}

// 3D点を2Dに投影
void AdRenderer::projectPoints(const cv::Mat& rvec,
                              const cv::Mat& tvec,
                              std::vector<cv::Point2f>& projected_points)
{
    // OpenCV projectPoints使用
    cv::projectPoints(backnet_corners_3d_, rvec, tvec,
                     camera_matrix_, dist_coeffs_,
                     projected_points);
}

// 透視変換行列計算
void AdRenderer::computePerspectiveTransform(
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points,
    cv::Mat& transform_matrix)
{
    // 4点対応から透視変換行列を計算
    transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
}

// 次のパートで実装予定:
// - applyTexture()
// - render()

} // namespace Rendering
} // namespace VirtualAd
