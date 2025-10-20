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

// デプス推定を実行
bool DepthEstimator::estimate(const cv::Mat& image, cv::Mat& depth_map)
{
    if (!is_loaded_) {
        last_error_ = "Model not loaded";
        return false;
    }
    
    if (image.empty()) {
        last_error_ = "Empty input image";
        return false;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 前処理
        std::vector<float> input_tensor;
        preprocessImage(image, input_tensor);
        
        // 入力テンソル作成
        std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor.data(), 
            input_tensor.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        // 入力名・出力名取得
        auto input_name_alloc = session_->GetInputNameAllocated(0, *allocator_);
        auto output_name_alloc = session_->GetOutputNameAllocated(0, *allocator_);
        
        const char* input_names[] = {input_name_alloc.get()};
        const char* output_names[] = {output_name_alloc.get()};
        
        // 推論実行
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor_ort,
            1,
            output_names,
            1
        );
        
        // 出力テンソル取得
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // 出力サイズ計算
        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        // 出力テンソルをvectorにコピー
        std::vector<float> output_tensor(output_data, output_data + output_size);
        
        // 後処理
        postprocessOutput(output_tensor, image.size(), depth_map);
        
        // 推論時間計測
        auto end_time = std::chrono::high_resolution_clock::now();
        inference_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        last_error_.clear();
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = std::string("Depth estimation failed: ") + e.what();
        std::cerr << last_error_ << std::endl;
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

// 出力テンソルを後処理
void DepthEstimator::postprocessOutput(const std::vector<float>& output_tensor,
                                      const cv::Size& original_size,
                                      cv::Mat& depth_map)
{
    // 出力形状: [1, 1, H, W]
    int output_h = input_size_;
    int output_w = input_size_;
    
    // 一時デプスマップ作成（モデル出力サイズ）
    cv::Mat temp_depth(output_h, output_w, CV_32FC1);
    
    // テンソルからMat形式に変換
    for (int y = 0; y < output_h; y++) {
        for (int x = 0; x < output_w; x++) {
            size_t index = y * output_w + x;
            temp_depth.at<float>(y, x) = output_tensor[index];
        }
    }
    
    // デプス値を0.0-1.0に正規化（小さいほど手前）
    double min_val, max_val;
    cv::minMaxLoc(temp_depth, &min_val, &max_val);
    
    cv::Mat normalized_depth;
    if (max_val - min_val > 1e-6) {
        // 反転: MiDaSは大きいほど手前なので、小さいほど手前に変換
        normalized_depth = (max_val - temp_depth) / (max_val - min_val);
    } else {
        // デプス差がない場合は全て0.5
        normalized_depth = cv::Mat(output_h, output_w, CV_32FC1, cv::Scalar(0.5f));
    }
    
    // 元画像サイズにリサイズ
    if (original_size.width != input_size_ || original_size.height != input_size_) {
        cv::resize(normalized_depth, depth_map, original_size, 0, 0, cv::INTER_LINEAR);
    } else {
        depth_map = normalized_depth.clone();
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
