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

// ONNXモデルをロード
bool DepthEstimator::loadModel(const std::string& model_path)
{
    try {
        // セッションオプション作成
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Windows用: ワイド文字列に変換
#ifdef _WIN32
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), 
                                             static_cast<int>(model_path.length()), nullptr, 0);
        std::wstring wmodel_path(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), 
                           static_cast<int>(model_path.length()), &wmodel_path[0], size_needed);
        
        // セッション作成（ワイド文字列）
        session_ = std::make_unique<Ort::Session>(*env_, wmodel_path.c_str(), *session_options_);
#else
        // Linux/Mac用
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
#endif
        
        // 入力情報取得
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes != 1) {
            last_error_ = "Expected 1 input node, got " + std::to_string(num_input_nodes);
            return false;
        }
        
        // 入力名取得
        auto input_name_alloc = session_->GetInputNameAllocated(0, *allocator_);
        std::string input_name(input_name_alloc.get());
        
        // 入力形状確認
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        
        std::cout << "Model loaded: " << model_path << std::endl;
        std::cout << "  Input: " << input_name << std::endl;
        std::cout << "  Input shape: [";
        for (size_t i = 0; i < input_dims.size(); i++) {
            std::cout << input_dims[i];
            if (i < input_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 出力情報取得
        size_t num_output_nodes = session_->GetOutputCount();
        if (num_output_nodes != 1) {
            last_error_ = "Expected 1 output node, got " + std::to_string(num_output_nodes);
            return false;
        }
        
        // 出力名取得
        auto output_name_alloc = session_->GetOutputNameAllocated(0, *allocator_);
        std::string output_name(output_name_alloc.get());
        
        std::cout << "  Output: " << output_name << std::endl;
        
        is_loaded_ = true;
        last_error_.clear();
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = std::string("Failed to load model: ") + e.what();
        std::cerr << last_error_ << std::endl;
        is_loaded_ = false;
        return false;
    }
}

// 画像を前処理
void DepthEstimator::preprocessImage(const cv::Mat& image, std::vector<float>& input_tensor)
{
    // リサイズ
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_size_, input_size_));
    
    // BGR → RGB変換
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // float変換と正規化
    cv::Mat float_image;
    rgb.convertTo(float_image, CV_32F, 1.0 / 255.0);
    
    // テンソルサイズ確保（[1, 3, H, W]）
    size_t tensor_size = 1 * 3 * input_size_ * input_size_;
    input_tensor.resize(tensor_size);
    
    // HWC → CHW変換 + ImageNet正規化
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < input_size_; y++) {
            for (int x = 0; x < input_size_; x++) {
                float pixel_value = float_image.at<cv::Vec3f>(y, x)[c];
                
                // ImageNet正規化
                pixel_value = (pixel_value - MEAN[c]) / STD[c];
                
                // テンソル格納（CHW順）
                size_t index = c * input_size_ * input_size_ + y * input_size_ + x;
                input_tensor[index] = pixel_value;
            }
        }
    }
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
