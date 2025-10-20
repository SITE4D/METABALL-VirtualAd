#pragma once

#include "ONNXInference.h"
#include "../tracking/PnPSolver.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace metaball {

/**
 * @brief カメラポーズ補正クラス
 * 
 * PnPソルバーによる従来のポーズ推定とAI推論を統合し、
 * より高精度なカメラポーズ推定を実現
 * 
 * 動作モード:
 * - PNP_ONLY: PnPソルバーのみ使用
 * - AI_ONLY: AI推論のみ使用
 * - BLENDED: PnPとAI推論をブレンディング（推奨）
 */
class CameraPoseRefiner {
public:
    /**
     * @brief 補正モード
     */
    enum class Mode {
        PNP_ONLY,   // PnPソルバーのみ
        AI_ONLY,    // AI推論のみ
        BLENDED     // ブレンディング（デフォルト）
    };

    /**
     * @brief コンストラクタ
     * @param pnp_solver PnPソルバー（shared_ptr）
     */
    explicit CameraPoseRefiner(std::shared_ptr<VirtualAd::Tracking::PnPSolver> pnp_solver);

    /**
     * @brief デストラクタ
     */
    ~CameraPoseRefiner();

    /**
     * @brief AIモデルを読み込む
     * @param model_path ONNXモデルファイルのパス
     * @return true 成功
     * @return false 失敗
     */
    bool loadModel(const std::string& model_path);

    /**
     * @brief カメラポーズを推定・補正
     * 
     * @param image 入力画像
     * @param object_points 3D点群（ワールド座標）
     * @param image_points 2D点群（画像座標）
     * @param pose 出力カメラポーズ
     * @param inlier_mask 出力インライアマスク（PnP使用時）
     * @return true 成功
     * @return false 失敗
     */
    bool refinePose(const cv::Mat& image,
                   const std::vector<cv::Point3f>& object_points,
                   const std::vector<cv::Point2f>& image_points,
                   VirtualAd::Tracking::CameraPose& pose,
                   std::vector<uchar>& inlier_mask);

    /**
     * @brief 補正モードを設定
     * @param mode 補正モード
     */
    void setMode(Mode mode) { mode_ = mode; }

    /**
     * @brief 現在の補正モードを取得
     * @return Mode 補正モード
     */
    Mode getMode() const { return mode_; }

    /**
     * @brief ブレンディング係数を設定
     * 
     * alpha = 0.0: PnPのみ
     * alpha = 1.0: AIのみ
     * alpha = 0.5: 50/50ブレンド（デフォルト）
     * 
     * @param alpha ブレンディング係数 [0.0, 1.0]
     */
    void setBlendAlpha(float alpha);

    /**
     * @brief ブレンディング係数を取得
     * @return float ブレンディング係数
     */
    float getBlendAlpha() const { return blend_alpha_; }

    /**
     * @brief AIモデルがロード済みかチェック
     * @return true ロード済み
     * @return false 未ロード
     */
    bool isModelLoaded() const { return ai_inference_ && ai_inference_->isLoaded(); }

    /**
     * @brief 最後のPnPエラーを取得
     * @return double 再投影誤差（ピクセル）
     */
    double getLastPnPError() const { return last_pnp_error_; }

    /**
     * @brief 最後の処理時間を取得
     * @return double 処理時間（ミリ秒）
     */
    double getLastProcessingTime() const { return last_processing_time_; }

    /**
     * @brief 最後のエラーメッセージを取得
     * @return std::string エラーメッセージ
     */
    std::string getLastError() const { return last_error_; }

private:
    /**
     * @brief PnPのみでポーズ推定
     * @param object_points 3D点群
     * @param image_points 2D点群
     * @param pose 出力ポーズ
     * @param inlier_mask 出力インライアマスク
     * @return true 成功
     * @return false 失敗
     */
    bool estimateWithPnP(const std::vector<cv::Point3f>& object_points,
                        const std::vector<cv::Point2f>& image_points,
                        VirtualAd::Tracking::CameraPose& pose,
                        std::vector<uchar>& inlier_mask);

    /**
     * @brief AIのみでポーズ推定
     * @param image 入力画像
     * @param pose 出力ポーズ
     * @return true 成功
     * @return false 失敗
     */
    bool estimateWithAI(const cv::Mat& image,
                       VirtualAd::Tracking::CameraPose& pose);

    /**
     * @brief PnPとAIをブレンディング
     * @param pnp_pose PnPポーズ
     * @param ai_pose AIポーズ
     * @param pose 出力ブレンドポーズ
     * @return true 成功
     * @return false 失敗
     */
    bool blendPoses(const VirtualAd::Tracking::CameraPose& pnp_pose,
                   const VirtualAd::Tracking::CameraPose& ai_pose,
                   VirtualAd::Tracking::CameraPose& pose);

    /**
     * @brief エラーメッセージを設定
     * @param error エラーメッセージ
     */
    void setError(const std::string& error);

private:
    // PnPソルバー
    std::shared_ptr<VirtualAd::Tracking::PnPSolver> pnp_solver_;
    
    // AI推論
    std::unique_ptr<ONNXInference> ai_inference_;
    
    // 補正モード
    Mode mode_;
    
    // ブレンディング係数（0.0=PnPのみ、1.0=AIのみ）
    float blend_alpha_;
    
    // 統計情報
    double last_pnp_error_;
    double last_processing_time_;
    std::string last_error_;
};

} // namespace metaball
