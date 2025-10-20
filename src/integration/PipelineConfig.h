#pragma once

#include <string>
#include <cstddef>

namespace VirtualAd {
namespace Integration {

/**
 * @brief 統合パイプラインの設定パラメータ
 * 
 * この構造体は、IntegratedPipelineクラスの初期化に必要な
 * すべての設定パラメータを保持します。
 */
struct PipelineConfig {
    // ========================================================================
    // 動作モード
    // ========================================================================
    
    /** シングルスレッドモード（true）またはマルチスレッドモード（false） */
    bool single_thread_mode = false;
    
    // ========================================================================
    // ONNXモデルパス
    // ========================================================================
    
    /** カメラポーズ推定ONNXモデルパス */
    std::string camera_pose_model_path = "models/camera_pose_net.onnx";
    
    /** セグメンテーションONNXモデルパス */
    std::string segmentation_model_path = "models/segmentation_model.onnx";
    
    /** デプス推定ONNXモデルパス */
    std::string depth_model_path = "models/depth_model.onnx";
    
    // ========================================================================
    // キュー設定
    // ========================================================================
    
    /** 入力キューの最大サイズ */
    size_t input_queue_size = 5;
    
    /** トラッキングキューの最大サイズ */
    size_t tracking_queue_size = 5;
    
    /** キーヤーキューの最大サイズ */
    size_t keyer_queue_size = 5;
    
    /** レンダリングキューの最大サイズ */
    size_t render_queue_size = 5;
    
    /** 出力キューの最大サイズ */
    size_t output_queue_size = 5;
    
    // ========================================================================
    // タイムアウト設定
    // ========================================================================
    
    /** push操作のタイムアウト（ミリ秒） */
    int push_timeout_ms = 1000;
    
    /** pop操作のタイムアウト（ミリ秒） */
    int pop_timeout_ms = 1000;
    
    // ========================================================================
    // バリデーション
    // ========================================================================
    
    /**
     * @brief 設定パラメータの妥当性をチェック
     * 
     * @return 妥当な場合true、そうでない場合false
     * 
     * 妥当性チェック項目:
     * - ONNXモデルパスが空でない
     * - キューサイズが1以上
     * - タイムアウト時間が正の値
     */
    bool validate() const {
        // モデルパスチェック
        if (camera_pose_model_path.empty()) {
            return false;
        }
        if (segmentation_model_path.empty()) {
            return false;
        }
        if (depth_model_path.empty()) {
            return false;
        }
        
        // キューサイズチェック
        if (input_queue_size < 1) {
            return false;
        }
        if (tracking_queue_size < 1) {
            return false;
        }
        if (keyer_queue_size < 1) {
            return false;
        }
        if (render_queue_size < 1) {
            return false;
        }
        if (output_queue_size < 1) {
            return false;
        }
        
        // タイムアウトチェック
        if (push_timeout_ms <= 0) {
            return false;
        }
        if (pop_timeout_ms <= 0) {
            return false;
        }
        
        return true;
    }
};

} // namespace Integration
} // namespace VirtualAd
