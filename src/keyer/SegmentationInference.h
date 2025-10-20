/**
 * @file SegmentationInference.h
 * @brief Segmentation inference using ONNX Runtime
 * 
 * DeepLabV3+モデルを使用したセグメンテーション推論。
 * 選手、審判、バックネット、背景の4クラス分類を行います。
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace VirtualAd {
namespace Keyer {

/**
 * @brief セグメンテーションクラス
 */
enum class SegmentationClass {
    BACKGROUND = 0,
    PLAYER = 1,
    UMPIRE = 2,
    BACKNET = 3
};

/**
 * @brief ONNX Runtime を使用したセグメンテーション推論
 */
class SegmentationInference {
public:
    /**
     * @brief コンストラクタ
     * @param input_size 入力画像サイズ（正方形、デフォルト512x512）
     */
    explicit SegmentationInference(int input_size = 512);
    
    /**
     * @brief デストラクタ
     */
    ~SegmentationInference();
    
    /**
     * @brief ONNXモデルをロード
     * @param model_path モデルファイルパス
     * @return 成功したらtrue
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief セグメンテーション推論実行
     * @param image 入力画像（BGR、任意サイズ）
     * @param mask 出力マスク（クラスインデックス、入力画像と同じサイズ）
     * @return 成功したらtrue
     */
    bool infer(const cv::Mat& image, cv::Mat& mask);
    
    /**
     * @brief モデルがロード済みか確認
     * @return ロード済みならtrue
     */
    bool isLoaded() const;
    
    /**
     * @brief 最後のエラーメッセージを取得
     * @return エラーメッセージ
     */
    std::string getLastError() const;
    
    /**
     * @brief 推論時間を取得（ミリ秒）
     * @return 最後の推論にかかった時間
     */
    double getInferenceTime() const;
    
    /**
     * @brief マスクをカラー画像に変換
     * @param mask クラスインデックスマスク
     * @param color_mask 出力カラーマスク（BGR）
     */
    static void maskToColor(const cv::Mat& mask, cv::Mat& color_mask);
    
    /**
     * @brief マスクをオーバーレイ表示
     * @param image 元画像
     * @param mask クラスインデックスマスク
     * @param output 出力画像（マスクをオーバーレイ）
     * @param alpha オーバーレイの透明度（0.0-1.0）
     */
    static void overlayMask(const cv::Mat& image, const cv::Mat& mask, 
                           cv::Mat& output, float alpha = 0.5f);

private:
    /**
     * @brief 画像を前処理
     * @param image 入力画像（BGR）
     * @param input_tensor 出力テンソル（[1, 3, H, W]、正規化済み）
     */
    void preprocessImage(const cv::Mat& image, std::vector<float>& input_tensor);
    
    /**
     * @brief 出力テンソルを後処理
     * @param output_tensor 出力テンソル（[1, 4, H, W]）
     * @param original_size 元画像サイズ
     * @param mask 出力マスク（クラスインデックス）
     */
    void postprocessOutput(const std::vector<float>& output_tensor,
                          const cv::Size& original_size,
                          cv::Mat& mask);
    
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    
    // モデル情報
    bool is_loaded_;
    int input_size_;            // 入力サイズ（正方形）
    std::string last_error_;
    double inference_time_;     // 最後の推論時間（ms）
    
    // 正規化パラメータ（ImageNet）
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3] = {0.229f, 0.224f, 0.225f};
    
    // クラスカラー（BGR）
    static constexpr cv::Scalar CLASS_COLORS[4] = {
        cv::Scalar(128, 128, 128),  // BACKGROUND: グレー
        cv::Scalar(0, 0, 255),      // PLAYER: 赤
        cv::Scalar(0, 255, 255),    // UMPIRE: 黄
        cv::Scalar(255, 0, 0)       // BACKNET: 青
    };
};

} // namespace Keyer
} // namespace VirtualAd
