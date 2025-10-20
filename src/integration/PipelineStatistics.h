#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace VirtualAd {
namespace Integration {

/**
 * @brief 統合パイプラインの統計情報
 * 
 * この構造体は、パイプライン実行中のパフォーマンス統計情報を保持します。
 * フレームレート、処理時間、キューサイズなどの情報を収集・計算します。
 */
struct PipelineStatistics {
    // ========================================================================
    // フレーム統計
    // ========================================================================
    
    /** 処理された総フレーム数 */
    uint64_t total_frames = 0;
    
    /** ドロップされたフレーム数 */
    uint64_t dropped_frames = 0;
    
    /** エラーフレーム数 */
    uint64_t error_frames = 0;
    
    // ========================================================================
    // 処理時間統計（ミリ秒）
    // ========================================================================
    
    /** トラッキング平均処理時間 */
    double avg_tracking_time_ms = 0.0;
    
    /** キーヤー平均処理時間 */
    double avg_keyer_time_ms = 0.0;
    
    /** レンダリング平均処理時間 */
    double avg_rendering_time_ms = 0.0;
    
    /** 総平均処理時間 */
    double avg_total_time_ms = 0.0;
    
    /** 最小処理時間 */
    double min_total_time_ms = 0.0;
    
    /** 最大処理時間 */
    double max_total_time_ms = 0.0;
    
    // ========================================================================
    // フレームレート統計（fps）
    // ========================================================================
    
    /** 現在のフレームレート */
    double current_fps = 0.0;
    
    /** 平均フレームレート */
    double avg_fps = 0.0;
    
    // ========================================================================
    // キュー統計
    // ========================================================================
    
    /** 入力キューサイズ */
    size_t input_queue_size = 0;
    
    /** トラッキングキューサイズ */
    size_t tracking_queue_size = 0;
    
    /** キーヤーキューサイズ */
    size_t keyer_queue_size = 0;
    
    /** レンダリングキューサイズ */
    size_t render_queue_size = 0;
    
    /** 出力キューサイズ */
    size_t output_queue_size = 0;
    
    // ========================================================================
    // ヘルパーメソッド
    // ========================================================================
    
    /**
     * @brief すべての統計情報をリセット
     */
    void reset() {
        total_frames = 0;
        dropped_frames = 0;
        error_frames = 0;
        
        avg_tracking_time_ms = 0.0;
        avg_keyer_time_ms = 0.0;
        avg_rendering_time_ms = 0.0;
        avg_total_time_ms = 0.0;
        min_total_time_ms = 0.0;
        max_total_time_ms = 0.0;
        
        current_fps = 0.0;
        avg_fps = 0.0;
        
        input_queue_size = 0;
        tracking_queue_size = 0;
        keyer_queue_size = 0;
        render_queue_size = 0;
        output_queue_size = 0;
    }
    
    /**
     * @brief フレーム処理時間を更新
     * 
     * @param tracking_time トラッキング処理時間（ミリ秒）
     * @param keyer_time キーヤー処理時間（ミリ秒）
     * @param rendering_time レンダリング処理時間（ミリ秒）
     * @param total_time 総処理時間（ミリ秒）
     */
    void updateFrameTime(double tracking_time, double keyer_time, 
                        double rendering_time, double total_time) {
        // 平均処理時間更新（移動平均）
        const double alpha = 0.1;  // 平滑化係数
        avg_tracking_time_ms = alpha * tracking_time + (1.0 - alpha) * avg_tracking_time_ms;
        avg_keyer_time_ms = alpha * keyer_time + (1.0 - alpha) * avg_keyer_time_ms;
        avg_rendering_time_ms = alpha * rendering_time + (1.0 - alpha) * avg_rendering_time_ms;
        avg_total_time_ms = alpha * total_time + (1.0 - alpha) * avg_total_time_ms;
        
        // 最小・最大処理時間更新
        if (total_frames == 0 || total_time < min_total_time_ms) {
            min_total_time_ms = total_time;
        }
        if (total_frames == 0 || total_time > max_total_time_ms) {
            max_total_time_ms = total_time;
        }
    }
    
    /**
     * @brief フレームレートを更新
     * 
     * @param fps 現在のフレームレート
     */
    void updateFPS(double fps) {
        current_fps = fps;
        
        // 平均FPS更新（移動平均）
        const double alpha = 0.1;  // 平滑化係数
        avg_fps = alpha * fps + (1.0 - alpha) * avg_fps;
    }
    
    /**
     * @brief ドロップ率を計算
     * 
     * @return ドロップ率（0.0-1.0）
     */
    double getDropRate() const {
        if (total_frames == 0) {
            return 0.0;
        }
        return static_cast<double>(dropped_frames) / static_cast<double>(total_frames);
    }
    
    /**
     * @brief エラー率を計算
     * 
     * @return エラー率（0.0-1.0）
     */
    double getErrorRate() const {
        if (total_frames == 0) {
            return 0.0;
        }
        return static_cast<double>(error_frames) / static_cast<double>(total_frames);
    }
};

} // namespace Integration
} // namespace VirtualAd
