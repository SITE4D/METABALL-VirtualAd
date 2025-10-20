#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace metaball {

/**
 * @brief ONNX Runtime C++ APIを使用した推論ラッパークラス
 * 
 * カメラポーズ推定モデル（ResNet-18ベース）のONNX推論を実行
 * 入力: RGB画像 (224x224)
 * 出力: カメラポーズ [rvec(3), tvec(3)]
 */
class ONNXInference {
public:
    /**
     * @brief コンストラクタ
     */
    ONNXInference();

    /**
     * @brief デストラクタ
     */
    ~ONNXInference();

    /**
     * @brief ONNXモデルを読み込む
     * 
     * @param model_path ONNXモデルファイルのパス
     * @return true 成功
     * @return false 失敗
     */
    bool loadModel(const std::string& model_path);

    /**
     * @brief 推論を実行
     * 
     * @param image 入力画像（任意サイズのBGR画像、内部で224x224にリサイズ）
     * @param pose 出力カメラポーズ [rvec(3), tvec(3)]
     * @return true 成功
     * @return false 失敗
     */
    bool infer(const cv::Mat& image, std::vector<float>& pose);

    /**
     * @brief モデルがロード済みかチェック
     * 
     * @return true ロード済み
     * @return false 未ロード
     */
    bool isLoaded() const { return model_loaded_; }

    /**
     * @brief 最後のエラーメッセージを取得
     * 
     * @return std::string エラーメッセージ
     */
    std::string getLastError() const { return last_error_; }

private:
    /**
     * @brief 画像の前処理（リサイズ、正規化）
     * 
     * @param image 入力画像（BGR）
     * @param input_tensor 出力テンソル
     * @return true 成功
     * @return false 失敗
     */
    bool preprocessImage(const cv::Mat& image, std::vector<float>& input_tensor);

    /**
     * @brief エラーメッセージを設定
     * 
     * @param error エラーメッセージ
     */
    void setError(const std::string& error);

private:
    // ONNX Runtime関連
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::MemoryInfo memory_info_;

    // モデル情報
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    // モデルパラメータ
    static constexpr int INPUT_WIDTH = 224;
    static constexpr int INPUT_HEIGHT = 224;
    static constexpr int INPUT_CHANNELS = 3;
    static constexpr int OUTPUT_SIZE = 6;  // rvec(3) + tvec(3)

    // ImageNet正規化パラメータ
    static constexpr float MEAN_R = 0.485f;
    static constexpr float MEAN_G = 0.456f;
    static constexpr float MEAN_B = 0.406f;
    static constexpr float STD_R = 0.229f;
    static constexpr float STD_G = 0.224f;
    static constexpr float STD_B = 0.225f;

    // 状態
    bool model_loaded_;
    std::string last_error_;
};

} // namespace metaball
