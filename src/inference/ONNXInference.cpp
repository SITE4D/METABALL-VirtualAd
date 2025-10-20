#include "ONNXInference.h"
#include <iostream>

namespace metaball {

ONNXInference::ONNXInference()
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault))
    , model_loaded_(false)
    , last_error_("")
{
}

ONNXInference::~ONNXInference() {
}

bool ONNXInference::loadModel(const std::string& model_path) {
    try {
        // ONNX Runtime環境初期化
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        
        // セッションオプション設定
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // モデル読み込み（Windowsの場合はワイド文字列に変換）
        #ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), *session_options_);
        #else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        #endif
        
        // 入力情報取得
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 入力ノード名
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes != 1) {
            setError("Model should have exactly 1 input node");
            return false;
        }
        
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_node_names_.push_back(input_name.get());
        
        // 入力shape
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();
        
        // 動的バッチサイズの場合、-1を1に設定
        if (input_shape_[0] == -1) {
            input_shape_[0] = 1;
        }
        
        // 入力shapeの検証
        if (input_shape_.size() != 4 || 
            input_shape_[1] != INPUT_CHANNELS ||
            input_shape_[2] != INPUT_HEIGHT ||
            input_shape_[3] != INPUT_WIDTH) {
            setError("Invalid input shape. Expected [1, 3, 224, 224]");
            return false;
        }
        
        // 出力情報取得
        size_t num_output_nodes = session_->GetOutputCount();
        if (num_output_nodes != 1) {
            setError("Model should have exactly 1 output node");
            return false;
        }
        
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_node_names_.push_back(output_name.get());
        
        // 出力shape
        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor_info.GetShape();
        
        // 動的バッチサイズの場合、-1を1に設定
        if (output_shape_[0] == -1) {
            output_shape_[0] = 1;
        }
        
        // 出力shapeの検証
        if (output_shape_.size() != 2 || output_shape_[1] != OUTPUT_SIZE) {
            setError("Invalid output shape. Expected [1, 6]");
            return false;
        }
        
        model_loaded_ = true;
        last_error_ = "";
        
        std::cout << "Model loaded successfully: " << model_path << std::endl;
        std::cout << "  Input shape: [" << input_shape_[0] << ", " << input_shape_[1] 
                  << ", " << input_shape_[2] << ", " << input_shape_[3] << "]" << std::endl;
        std::cout << "  Output shape: [" << output_shape_[0] << ", " << output_shape_[1] << "]" << std::endl;
        
        return true;
    }
    catch (const Ort::Exception& e) {
        setError(std::string("ONNX Runtime error: ") + e.what());
        return false;
    }
    catch (const std::exception& e) {
        setError(std::string("Error: ") + e.what());
        return false;
    }
}

bool ONNXInference::infer(const cv::Mat& image, std::vector<float>& pose) {
    if (!model_loaded_) {
        setError("Model not loaded");
        return false;
    }
    
    if (image.empty()) {
        setError("Input image is empty");
        return false;
    }
    
    try {
        // 前処理
        std::vector<float> input_tensor;
        if (!preprocessImage(image, input_tensor)) {
            return false;
        }
        
        // 入力テンソル作成
        auto input_tensor_obj = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_tensor.data(),
            input_tensor.size(),
            input_shape_.data(),
            input_shape_.size()
        );
        
        // 推論実行
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(),
            &input_tensor_obj,
            1,
            output_node_names_.data(),
            1
        );
        
        // 出力取得
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        pose.assign(output_data, output_data + OUTPUT_SIZE);
        
        return true;
    }
    catch (const Ort::Exception& e) {
        setError(std::string("Inference error: ") + e.what());
        return false;
    }
    catch (const std::exception& e) {
        setError(std::string("Error: ") + e.what());
        return false;
    }
}

bool ONNXInference::preprocessImage(const cv::Mat& image, std::vector<float>& input_tensor) {
    try {
        // BGR → RGB変換
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        
        // リサイズ
        cv::Mat resized_image;
        cv::resize(rgb_image, resized_image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        
        // float変換 [0, 255] → [0.0, 1.0]
        cv::Mat float_image;
        resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
        
        // テンソルサイズ確保
        input_tensor.resize(INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
        
        // HWC → CHW変換 + 正規化（ImageNet mean/std）
        for (int c = 0; c < INPUT_CHANNELS; ++c) {
            float mean = (c == 0) ? MEAN_R : (c == 1) ? MEAN_G : MEAN_B;
            float std = (c == 0) ? STD_R : (c == 1) ? STD_G : STD_B;
            
            for (int h = 0; h < INPUT_HEIGHT; ++h) {
                for (int w = 0; w < INPUT_WIDTH; ++w) {
                    int hwc_idx = h * INPUT_WIDTH * INPUT_CHANNELS + w * INPUT_CHANNELS + c;
                    int chw_idx = c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w;
                    
                    float pixel = float_image.data[hwc_idx * sizeof(float)];
                    float* pixel_ptr = reinterpret_cast<float*>(&float_image.data[hwc_idx * sizeof(float)]);
                    input_tensor[chw_idx] = (*pixel_ptr - mean) / std;
                }
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        setError(std::string("Preprocessing error: ") + e.what());
        return false;
    }
}

void ONNXInference::setError(const std::string& error) {
    last_error_ = error;
    std::cerr << "ONNXInference Error: " << error << std::endl;
}

} // namespace metaball
