#pragma once

#include "FrameData.h"
#include "PipelineConfig.h"
#include "PipelineStatistics.h"
#include "ThreadSafeQueue.h"
#include "../tracking/FeatureTracker.h"
#include "../tracking/PnPSolver.h"
#include "../inference/CameraPoseRefiner.h"
#include "../keyer/SegmentationInference.h"
#include "../keyer/DepthEstimator.h"
#include "../keyer/DepthCompositor.h"
#include "../rendering/AdRenderer.h"

#include <memory>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

namespace VirtualAd {
namespace Integration {

/**
 * @brief 統合パイプラインクラス
 * 
 * すべてのコンポーネント（Tracking, Keyer, Rendering）を統合し、
 * シングルスレッドまたはマルチスレッドで処理を実行します。
 */
class IntegratedPipeline {
public:
    /**
     * @brief 動作モード
     */
    enum class Mode {
        SINGLE_THREAD,  ///< シングルスレッドモード（デバッグ用）
        MULTI_THREAD    ///< マルチスレッドモード（本番用）
    };
    
    /**
     * @brief コンストラクタ
     */
    IntegratedPipeline();
    
    /**
     * @brief デストラクタ
     */
    ~IntegratedPipeline();
    
    // コピー・ムーブ禁止
    IntegratedPipeline(const IntegratedPipeline&) = delete;
    IntegratedPipeline& operator=(const IntegratedPipeline&) = delete;
    IntegratedPipeline(IntegratedPipeline&&) = delete;
    IntegratedPipeline& operator=(IntegratedPipeline&&) = delete;
    
    /**
     * @brief パイプライン初期化
     * 
     * @param config 設定パラメータ
     * @return 成功時true、失敗時false
     */
    bool initialize(const PipelineConfig& config);
    
    /**
     * @brief パイプライン開始（マルチスレッドモードのみ）
     * 
     * @return 成功時true、失敗時false
     */
    bool start();
    
    /**
     * @brief パイプライン停止（マルチスレッドモードのみ）
     */
    void stop();
    
    /**
     * @brief 1フレーム処理
     * 
     * @param frame 入力フレーム
     * @param output 出力フレーム
     * @return 成功時true、失敗時false
     * 
     * シングルスレッドモードでは、この関数内ですべての処理を実行します。
     * マルチスレッドモードでは、フレームをキューに追加し、別スレッドで処理します。
     */
    bool processFrame(const cv::Mat& frame, cv::Mat& output);
    
    /**
     * @brief 統計情報取得
     * 
     * @return 現在の統計情報
     */
    PipelineStatistics getStatistics() const;
    
private:
    // ========================================================================
    // コンポーネント
    // ========================================================================
    
    std::unique_ptr<Tracking::FeatureTracker> tracker_;
    std::unique_ptr<Tracking::PnPSolver> pnp_solver_;
    std::unique_ptr<Inference::CameraPoseRefiner> pose_refiner_;
    std::unique_ptr<Keyer::SegmentationInference> segmentation_;
    std::unique_ptr<Keyer::DepthEstimator> depth_estimator_;
    std::unique_ptr<Keyer::DepthCompositor> compositor_;
    std::unique_ptr<Rendering::AdRenderer> renderer_;
    
    // ========================================================================
    // スレッド管理
    // ========================================================================
    
    Mode mode_;
    std::atomic<bool> running_;
    std::thread tracking_thread_;
    std::thread keyer_thread_;
    std::thread render_thread_;
    
    // ========================================================================
    // キュー（マルチスレッドモード用）
    // ========================================================================
    
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<FrameData>>> input_queue_;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<FrameData>>> tracking_queue_;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<FrameData>>> keyer_queue_;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<FrameData>>> render_queue_;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<FrameData>>> output_queue_;
    
    // ========================================================================
    // 設定・統計情報
    // ========================================================================
    
    PipelineConfig config_;
    mutable std::mutex stats_mutex_;
    PipelineStatistics statistics_;
    
    // ========================================================================
    // プライベートメソッド（実装は次のステップ）
    // ========================================================================
    
    bool processFrameSingleThread(std::shared_ptr<FrameData> frame_data);
    void trackingThreadFunc();
    void keyerThreadFunc();
    void renderThreadFunc();
};

} // namespace Integration
} // namespace VirtualAd
