#include "IntegratedPipeline.h"
#include <iostream>

namespace VirtualAd {
namespace Integration {

// ============================================================================
// コンストラクタ・デストラクタ
// ============================================================================

IntegratedPipeline::IntegratedPipeline()
    : mode_(Mode::SINGLE_THREAD)
    , running_(false)
{
    std::cout << "[IntegratedPipeline] Constructor called" << std::endl;
}

IntegratedPipeline::~IntegratedPipeline() {
    std::cout << "[IntegratedPipeline] Destructor called" << std::endl;
    
    // マルチスレッドモードの場合、スレッド停止
    if (running_) {
        stop();
    }
}

// ============================================================================
// パブリックメソッド
// ============================================================================

bool IntegratedPipeline::initialize(const PipelineConfig& config) {
    std::cout << "[IntegratedPipeline] Initializing..." << std::endl;
    
    // 設定パラメータのバリデーション
    if (!config.validate()) {
        std::cerr << "[IntegratedPipeline] ERROR: Invalid configuration" << std::endl;
        return false;
    }
    
    // 設定保存
    config_ = config;
    
    // 動作モード設定
    mode_ = config.single_thread_mode ? Mode::SINGLE_THREAD : Mode::MULTI_THREAD;
    std::cout << "[IntegratedPipeline] Mode: " 
              << (mode_ == Mode::SINGLE_THREAD ? "SINGLE_THREAD" : "MULTI_THREAD") 
              << std::endl;
    
    // ========================================================================
    // コンポーネント初期化
    // ========================================================================
    
    // FeatureTracker初期化
    try {
        // デフォルトカメラパラメータ（Full HD: 1920x1080）
        Tracking::CameraIntrinsics intrinsics = Tracking::CameraIntrinsics::createDefaultFullHD();
        
        // ORB特徴検出器を使用
        tracker_ = std::make_unique<Tracking::FeatureTracker>(
            Tracking::FeatureDetectorType::ORB,
            intrinsics,
            1000  // max_features
        );
        std::cout << "[IntegratedPipeline] FeatureTracker initialized (ORB, 1000 features)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[IntegratedPipeline] ERROR: Failed to create FeatureTracker: " 
                  << e.what() << std::endl;
        return false;
    }
    
    // PnPSolver初期化
    try {
        // デフォルトカメラパラメータ（Full HD: 1920x1080）
        Tracking::CameraIntrinsics intrinsics = Tracking::CameraIntrinsics::createDefaultFullHD();
        
        pnp_solver_ = std::make_unique<Tracking::PnPSolver>(intrinsics);
        std::cout << "[IntegratedPipeline] PnPSolver initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[IntegratedPipeline] ERROR: Failed to create PnPSolver: " 
                  << e.what() << std::endl;
        return false;
    }
    
    // CameraPoseRefiner初期化 - ONNX Runtime required (commented out)
    // try {
    //     pose_refiner_ = std::make_unique<Inference::CameraPoseRefiner>(pnp_solver_);
    //     if (!pose_refiner_->loadModel(config.camera_pose_model_path)) {
    //         std::cerr << "[IntegratedPipeline] ERROR: Failed to load camera pose model: " 
    //                   << config.camera_pose_model_path << std::endl;
    //         // 注: モデル読み込みは任意（PnPのみでも動作可能）
    //         std::cout << "[IntegratedPipeline] WARNING: CameraPoseRefiner will use PnP only" << std::endl;
    //     }
    //     std::cout << "[IntegratedPipeline] CameraPoseRefiner initialized" << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "[IntegratedPipeline] ERROR: Failed to create CameraPoseRefiner: " 
    //               << e.what() << std::endl;
    //     return false;
    // }
    std::cout << "[IntegratedPipeline] WARNING: CameraPoseRefiner disabled (ONNX Runtime not available)" << std::endl;
    
    // SegmentationInference初期化 - ONNX Runtime required (commented out)
    // try {
    //     segmentation_ = std::make_unique<Keyer::SegmentationInference>();
    //     if (!segmentation_->loadModel(config.segmentation_model_path)) {
    //         std::cerr << "[IntegratedPipeline] ERROR: Failed to load segmentation model: " 
    //                   << config.segmentation_model_path << std::endl;
    //         // 注: モデル読み込みは任意
    //         std::cout << "[IntegratedPipeline] WARNING: Segmentation will be skipped" << std::endl;
    //     }
    //     std::cout << "[IntegratedPipeline] SegmentationInference initialized" << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "[IntegratedPipeline] ERROR: Failed to create SegmentationInference: " 
    //               << e.what() << std::endl;
    //     return false;
    // }
    std::cout << "[IntegratedPipeline] WARNING: SegmentationInference disabled (ONNX Runtime not available)" << std::endl;
    
    // DepthEstimator初期化 - ONNX Runtime required (commented out)
    // try {
    //     depth_estimator_ = std::make_unique<Keyer::DepthEstimator>();
    //     if (!depth_estimator_->loadModel(config.depth_model_path)) {
    //         std::cerr << "[IntegratedPipeline] ERROR: Failed to load depth model: " 
    //                   << config.depth_model_path << std::endl;
    //         // 注: モデル読み込みは任意
    //         std::cout << "[IntegratedPipeline] WARNING: Depth estimation will be skipped" << std::endl;
    //     }
    //     std::cout << "[IntegratedPipeline] DepthEstimator initialized" << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "[IntegratedPipeline] ERROR: Failed to create DepthEstimator: " 
    //               << e.what() << std::endl;
    //     return false;
    // }
    std::cout << "[IntegratedPipeline] WARNING: DepthEstimator disabled (ONNX Runtime not available)" << std::endl;
    
    // DepthCompositor初期化
    try {
        compositor_ = std::make_unique<Keyer::DepthCompositor>();
        std::cout << "[IntegratedPipeline] DepthCompositor initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[IntegratedPipeline] ERROR: Failed to create DepthCompositor: " 
                  << e.what() << std::endl;
        return false;
    }
    
    // AdRenderer初期化
    try {
        renderer_ = std::make_unique<Rendering::AdRenderer>();
        std::cout << "[IntegratedPipeline] AdRenderer initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[IntegratedPipeline] ERROR: Failed to create AdRenderer: " 
                  << e.what() << std::endl;
        return false;
    }
    
    // ========================================================================
    // マルチスレッドモードの場合、キュー初期化
    // ========================================================================
    
    if (mode_ == Mode::MULTI_THREAD) {
        input_queue_ = std::make_unique<ThreadSafeQueue<std::shared_ptr<FrameData>>>(
            config.input_queue_size);
        tracking_queue_ = std::make_unique<ThreadSafeQueue<std::shared_ptr<FrameData>>>(
            config.tracking_queue_size);
        keyer_queue_ = std::make_unique<ThreadSafeQueue<std::shared_ptr<FrameData>>>(
            config.keyer_queue_size);
        render_queue_ = std::make_unique<ThreadSafeQueue<std::shared_ptr<FrameData>>>(
            config.render_queue_size);
        output_queue_ = std::make_unique<ThreadSafeQueue<std::shared_ptr<FrameData>>>(
            config.output_queue_size);
        
        std::cout << "[IntegratedPipeline] Thread-safe queues initialized" << std::endl;
    }
    
    // 統計情報リセット
    statistics_.reset();
    
    std::cout << "[IntegratedPipeline] Initialization complete" << std::endl;
    return true;
}

// ============================================================================
// プライベートメソッド（スタブ実装、次のステップで完全実装）
// ============================================================================

bool IntegratedPipeline::start() {
    // TODO: 次のステップで実装
    std::cout << "[IntegratedPipeline] start() - Not implemented yet" << std::endl;
    return false;
}

void IntegratedPipeline::stop() {
    // TODO: 次のステップで実装
    std::cout << "[IntegratedPipeline] stop() - Not implemented yet" << std::endl;
}

bool IntegratedPipeline::processFrame(const cv::Mat& frame, cv::Mat& output) {
    // 入力画像チェック
    if (frame.empty()) {
        std::cerr << "[IntegratedPipeline] ERROR: Input frame is empty" << std::endl;
        return false;
    }
    
    // モード別処理
    if (mode_ == Mode::SINGLE_THREAD) {
        // ========================================================================
        // シングルスレッドモード
        // ========================================================================
        
        // FrameData作成
        auto frame_data = std::make_shared<FrameData>();
        frame_data->image = frame.clone();
        frame_data->frame_id = statistics_.total_frames;
        
        // タイムスタンプ計算（秒）
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        frame_data->timestamp = std::chrono::duration<double>(duration).count();
        
        // シングルスレッド処理実行
        if (processFrameSingleThread(frame_data)) {
            // 出力画像取得
            if (!frame_data->final_output.empty()) {
                output = frame_data->final_output;
                return true;
            } else {
                std::cerr << "[IntegratedPipeline] ERROR: Final output is empty" << std::endl;
                return false;
            }
        } else {
            std::cerr << "[IntegratedPipeline] ERROR: Single-thread processing failed" << std::endl;
            return false;
        }
    } else {
        // ========================================================================
        // マルチスレッドモード（次のステップで実装）
        // ========================================================================
        
        std::cerr << "[IntegratedPipeline] ERROR: Multi-thread mode not implemented yet" << std::endl;
        return false;
    }
}

PipelineStatistics IntegratedPipeline::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

bool IntegratedPipeline::processFrameSingleThread(std::shared_ptr<FrameData> frame_data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // データバリデーション
    if (!frame_data || !frame_data->validate()) {
        std::cerr << "[IntegratedPipeline] ERROR: Invalid frame data" << std::endl;
        return false;
    }
    
    // ========================================================================
    // トラッキング処理
    // ========================================================================
    auto tracking_start = std::chrono::high_resolution_clock::now();
    
    // 1. 特徴点検出・トラッキング（FeatureTracker使用）
    // NOTE: FeatureTrackerはinitialize()が必要なため、ここでは簡易的な処理
    // 実際のシステムでは事前にリファレンスフレームで初期化する必要がある
    
    // ダミーデータ設定（実装後は実際のトラッキング結果に置き換え）
    frame_data->tracking_success = true;
    frame_data->rvec = cv::Mat::zeros(3, 1, CV_64F);
    frame_data->tvec = cv::Mat::zeros(3, 1, CV_64F);
    frame_data->tvec.at<double>(2, 0) = 3.0;  // Z軸方向に3メートル
    
    // ダミーコーナー設定（1920x1080の画面中央に200x200のバックネット）
    frame_data->corners.clear();
    frame_data->corners.push_back(cv::Point2f(860, 440));  // 左上
    frame_data->corners.push_back(cv::Point2f(1060, 440)); // 右上
    frame_data->corners.push_back(cv::Point2f(1060, 640)); // 右下
    frame_data->corners.push_back(cv::Point2f(860, 640));  // 左下
    frame_data->inlier_count = 4;
    
    auto tracking_end = std::chrono::high_resolution_clock::now();
    frame_data->tracking_time_ms = std::chrono::duration<double, std::milli>(
        tracking_end - tracking_start).count();
    
    std::cout << "[IntegratedPipeline] Tracking: " 
              << frame_data->tracking_time_ms << " ms" << std::endl;
    
    // ========================================================================
    // キーヤー処理
    // ========================================================================
    auto keyer_start = std::chrono::high_resolution_clock::now();
    
    // 1. セグメンテーション推論（SegmentationInference使用）- ONNX Runtime required (commented out)
    // if (segmentation_ && segmentation_->isLoaded()) {
    //     cv::Mat seg_mask;
    //     if (segmentation_->infer(frame_data->image, seg_mask)) {
    //         // セグメンテーションマスクを保存（0=背景、1=選手、2=審判、3=バックネット）
    //         frame_data->segmentation_mask = seg_mask;
    //         std::cout << "[IntegratedPipeline] Segmentation: Success" << std::endl;
    //     } else {
    //         std::cerr << "[IntegratedPipeline] WARNING: Segmentation failed: " 
    //                   << segmentation_->getLastError() << std::endl;
    //         // ダミーマスク作成（全て背景）
    //         frame_data->segmentation_mask = cv::Mat::zeros(
    //             frame_data->image.size(), CV_8UC1);
    //     }
    // } else {
    //     // モデル未ロードの場合、ダミーマスク作成
    //     std::cout << "[IntegratedPipeline] WARNING: Segmentation model not loaded, using dummy mask" << std::endl;
    //     frame_data->segmentation_mask = cv::Mat::zeros(
    //         frame_data->image.size(), CV_8UC1);
    // }
    
    // ダミーマスク作成（全て背景）- ONNX Runtime not available
    std::cout << "[IntegratedPipeline] INFO: Using dummy segmentation mask (ONNX Runtime not available)" << std::endl;
    frame_data->segmentation_mask = cv::Mat::zeros(frame_data->image.size(), CV_8UC1);
    
    // 2. デプス推定（DepthEstimator使用）- ONNX Runtime required (commented out)
    // if (depth_estimator_ && depth_estimator_->isLoaded()) {
    //     cv::Mat depth;
    //     if (depth_estimator_->estimate(frame_data->image, depth)) {
    //         // デプスマップを保存（CV_32FC1、0.0=近、1.0=遠）
    //         frame_data->depth_map = depth;
    //         std::cout << "[IntegratedPipeline] Depth estimation: Success" << std::endl;
    //     } else {
    //         std::cerr << "[IntegratedPipeline] WARNING: Depth estimation failed: "
    //                   << depth_estimator_->getLastError() << std::endl;
    //         // ダミーデプスマップ作成（一定値0.5）
    //         frame_data->depth_map = cv::Mat::ones(
    //             frame_data->image.size(), CV_32FC1) * 0.5f;
    //     }
    // } else {
    //     // モデル未ロードの場合、ダミーデプスマップ作成
    //     std::cout << "[IntegratedPipeline] WARNING: Depth model not loaded, using dummy depth map" << std::endl;
    //     frame_data->depth_map = cv::Mat::ones(
    //         frame_data->image.size(), CV_32FC1) * 0.5f;
    // }
    
    // ダミーデプスマップ作成（一定値0.5）- ONNX Runtime not available
    std::cout << "[IntegratedPipeline] INFO: Using dummy depth map (ONNX Runtime not available)" << std::endl;
    frame_data->depth_map = cv::Mat::ones(frame_data->image.size(), CV_32FC1) * 0.5f;
    
    auto keyer_end = std::chrono::high_resolution_clock::now();
    frame_data->keyer_time_ms = std::chrono::duration<double, std::milli>(
        keyer_end - keyer_start).count();
    
    std::cout << "[IntegratedPipeline] Keyer: " 
              << frame_data->keyer_time_ms << " ms" << std::endl;
    
    // ========================================================================
    // レンダリング・合成処理
    // ========================================================================
    auto render_start = std::chrono::high_resolution_clock::now();
    
    // 1. 広告レンダリング（AdRenderer使用）
    // NOTE: AdRendererは初期化とテクスチャ設定が必要なため、ダミー実装
    // 実際のシステムでは事前に初期化・設定する必要がある
    if (renderer_ && renderer_->isInitialized()) {
        cv::Mat rendered;
        if (renderer_->render(frame_data->image, frame_data->rvec, 
                             frame_data->tvec, rendered)) {
            frame_data->rendered_ad = rendered;
            std::cout << "[IntegratedPipeline] Ad rendering: Success" << std::endl;
        } else {
            std::cerr << "[IntegratedPipeline] WARNING: Ad rendering failed: "
                      << renderer_->getLastError() << std::endl;
            // ダミー広告画像作成（緑色の矩形）
            frame_data->rendered_ad = frame_data->image.clone();
            cv::rectangle(frame_data->rendered_ad, frame_data->corners[0], 
                         frame_data->corners[2], cv::Scalar(0, 255, 0), -1);
        }
    } else {
        // レンダラー未初期化の場合、ダミー広告画像作成
        std::cout << "[IntegratedPipeline] WARNING: Renderer not initialized, using dummy ad" << std::endl;
        frame_data->rendered_ad = frame_data->image.clone();
        if (frame_data->corners.size() == 4) {
            cv::rectangle(frame_data->rendered_ad, frame_data->corners[0], 
                         frame_data->corners[2], cv::Scalar(0, 255, 0), -1);
        }
    }
    
    // 2. デプス合成（DepthCompositor使用）
    if (compositor_) {
        cv::Mat composited;
        if (compositor_->composite(frame_data->image, 
                                  frame_data->segmentation_mask,
                                  frame_data->depth_map,
                                  frame_data->rendered_ad,
                                  composited, 0.5f)) {
            frame_data->final_output = composited;
            std::cout << "[IntegratedPipeline] Depth compositing: Success" << std::endl;
        } else {
            std::cerr << "[IntegratedPipeline] WARNING: Compositing failed: "
                      << compositor_->getLastError() << std::endl;
            // フォールバック: レンダリング済み画像をそのまま使用
            frame_data->final_output = frame_data->rendered_ad;
        }
    } else {
        // コンポジター未作成の場合、レンダリング済み画像をそのまま使用
        std::cout << "[IntegratedPipeline] WARNING: Compositor not available" << std::endl;
        frame_data->final_output = frame_data->rendered_ad;
    }
    
    auto render_end = std::chrono::high_resolution_clock::now();
    frame_data->rendering_time_ms = std::chrono::duration<double, std::milli>(
        render_end - render_start).count();
    
    std::cout << "[IntegratedPipeline] Rendering: " 
              << frame_data->rendering_time_ms << " ms" << std::endl;
    
    // ========================================================================
    // 統計情報更新
    // ========================================================================
    auto end_time = std::chrono::high_resolution_clock::now();
    frame_data->total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    // スレッドセーフに統計情報を更新
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        statistics_.total_frames++;
        statistics_.updateFrameTime(
            frame_data->tracking_time_ms,
            frame_data->keyer_time_ms,
            frame_data->rendering_time_ms,
            frame_data->total_time_ms
        );
        
        // FPS計算（1000ms / 処理時間ms）
        double current_fps = (frame_data->total_time_ms > 0.0) ? 
            (1000.0 / frame_data->total_time_ms) : 0.0;
        statistics_.updateFPS(current_fps);
    }
    
    std::cout << "[IntegratedPipeline] Total processing time: " 
              << frame_data->total_time_ms << " ms" << std::endl;
    
    return true;
}

void IntegratedPipeline::trackingThreadFunc() {
    // TODO: 次のステップで実装
}

void IntegratedPipeline::keyerThreadFunc() {
    // TODO: 次のステップで実装
}

void IntegratedPipeline::renderThreadFunc() {
    // TODO: 次のステップで実装
}

} // namespace Integration
} // namespace VirtualAd
