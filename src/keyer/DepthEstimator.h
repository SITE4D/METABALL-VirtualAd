/**
 * @file DepthEstimator.h
 * @brief Depth estimation inference using ONNX Runtime
 * 
 * MiDaS Smallモデルを使用した相対デプス推定。
 */

#ifndef VIRTUALAD_KEYER_DEPTHESTIMATOR_H
#define VIRTUALAD_KEYER_DEPTHESTIMATOR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace VirtualAd {
namespace Keyer {

/**
 * @class DepthEstimator
 * @brief MiDaS Smallによる相対デプス推定
 * 
 * ONNX Runtimeを使用してデプスマップを推定します。
 * 入力: RGB画像（任意サイズ）
 * 出力: 相対デプスマップ（0.0-1.0、小さいほど手前）
 */
class DepthEstimator {
public:
    /**
     * @brief コンストラクタ
     * @param input_size モデル入力サイズ（デフォルト384x384、MiDaS Small）
     */
    explicit DepthEstimator(int input_size = 384);
    
    /**
     * @brief デストラクタ
     */
    ~DepthEstimator();
    
    /**
     * @brief ONNXモデルをロード
     * @param model_path ONNXモデルファイルパス
     * @return 成功した場合true
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief デプス推定を実行
     * @param image 入力画像（BGR、任意サイズ）
     * @param depth_map 出力デプスマップ（CV_32FC1、元画像サイズ、0.0-1.0）
     * @return 成功した場合true
     */
    bool estimate(const cv::Mat& image, cv::Mat& depth_map);
    
    /**
     * @brief モデルがロード済みか確認
     * @return ロード済みの場合true
     */
    bool isLoaded() const;
    
    /**
     * @brief 最後のエラーメッセージを取得
     * @return エラーメッセージ
     */
    std::string getLastError() const;
    
    /**
     * @brief 推論時間を取得（ミリ秒）
     * @return 推論時間
     */
    double getInferenceTime() const;
    
    /**
     * @brief デプスマップを可視化
     * @param depth_map 入力デプスマップ（CV_32FC1、0.0-1.0）
     * @param color_depth 出力カラー化デプスマップ（CV_8UC3）
     * @param colormap OpenCVカラーマップ（デフォルト: COLORMAP_MAGMA）
     */
    static void visualizeDepth(const cv::Mat& depth_map, 
                              cv::Mat& color_depth,
                              int colormap = cv::COLORMAP_MAGMA);

private:
    /**
     * @brief 画像を前処理
     * @param image 入力画像（BGR）
     * @param input_tensor 出力テンソル（[1, 3, H, W]、ImageNet正規化）
     */
    void preprocessImage(const cv::Mat& image, std::vector<float>& input_tensor);
    
    /**
     * @brief 出力テンソルを後処理
     * @param output_tensor 出力テンソル（[1, 1, H, W]）
     * @param original_size 元画像サイズ
     * @param depth_map 出力デプスマップ（元画像サイズ、正規化済み）
     */
    void postprocessOutput(const std::vector<float>& output_tensor,
                          const cv::Size& original_size,
                          cv::Mat& depth_map);

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    
    // モデル設定
    bool is_loaded_;
    int input_size_;
    
    // ImageNet正規化パラメータ
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3] = {0.229f, 0.224f, 0.225f};
    
    // エラー・パフォーマンス情報
    std::string last_error_;
    double inference_time_;
};

} // namespace Keyer
} // namespace VirtualAd

#endif // VIRTUALAD_KEYER_DEPTHESTIMATOR_H
