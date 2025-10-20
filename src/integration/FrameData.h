#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

namespace VirtualAd {
namespace Integration {

/**
 * @brief フレーム処理に必要なすべてのデータを保持する構造体
 * 
 * この構造体は、統合パイプライン内でフレームとその処理結果を
 * 管理するために使用されます。各処理段階（トラッキング、キーヤー、
 * レンダリング）の結果を保持し、スレッド間でデータを受け渡します。
 */
struct FrameData {
    // ========================================================================
    // 基本情報
    // ========================================================================
    
    /** フレームID（連番） */
    int64_t frame_id;
    
    /** タイムスタンプ（秒） */
    double timestamp;
    
    // ========================================================================
    // 画像データ
    // ========================================================================
    
    /** 入力画像（BGR、1920x1080） */
    cv::Mat image;
    
    // ========================================================================
    // トラッキング結果
    // ========================================================================
    
    /** トラッキング成功フラグ */
    bool tracking_success;
    
    /** 回転ベクトル（3x1、Rodrigues形式） */
    cv::Mat rvec;
    
    /** 並進ベクトル（3x1） */
    cv::Mat tvec;
    
    /** 検出されたバックネットコーナー（4点） */
    std::vector<cv::Point2f> corners;
    
    /** インライア数（RANSAC後） */
    int inlier_count;
    
    // ========================================================================
    // キーヤー結果
    // ========================================================================
    
    /** セグメンテーションマスク（CV_8UC1、0=背景、255=前景） */
    cv::Mat segmentation_mask;
    
    /** デプスマップ（CV_32FC1、正規化済み、0.0=近、1.0=遠） */
    cv::Mat depth_map;
    
    // ========================================================================
    // レンダリング結果
    // ========================================================================
    
    /** レンダリング済みバーチャル広告（BGR） */
    cv::Mat rendered_ad;
    
    /** 最終出力画像（BGR、1920x1080） */
    cv::Mat final_output;
    
    // ========================================================================
    // パフォーマンス統計
    // ========================================================================
    
    /** トラッキング処理時間（ミリ秒） */
    double tracking_time_ms;
    
    /** キーヤー処理時間（ミリ秒） */
    double keyer_time_ms;
    
    /** レンダリング処理時間（ミリ秒） */
    double rendering_time_ms;
    
    /** 総処理時間（ミリ秒） */
    double total_time_ms;
    
    // ========================================================================
    // コンストラクタ
    // ========================================================================
    
    /**
     * @brief デフォルトコンストラクタ
     * 
     * すべてのメンバー変数をデフォルト値で初期化します。
     */
    FrameData()
        : frame_id(0)
        , timestamp(0.0)
        , tracking_success(false)
        , inlier_count(0)
        , tracking_time_ms(0.0)
        , keyer_time_ms(0.0)
        , rendering_time_ms(0.0)
        , total_time_ms(0.0)
    {
    }
    
    // ========================================================================
    // ヘルパーメソッド
    // ========================================================================
    
    /**
     * @brief すべてのデータをクリア
     * 
     * この関数は、FrameDataのすべてのメンバー変数を初期状態に戻します。
     * メモリ効率のため、cv::Matはreleaseされます。
     */
    void clear() {
        frame_id = 0;
        timestamp = 0.0;
        
        // 画像データ解放
        image.release();
        
        // トラッキング結果クリア
        tracking_success = false;
        rvec.release();
        tvec.release();
        corners.clear();
        inlier_count = 0;
        
        // キーヤー結果クリア
        segmentation_mask.release();
        depth_map.release();
        
        // レンダリング結果クリア
        rendered_ad.release();
        final_output.release();
        
        // 統計情報クリア
        tracking_time_ms = 0.0;
        keyer_time_ms = 0.0;
        rendering_time_ms = 0.0;
        total_time_ms = 0.0;
    }
    
    /**
     * @brief FrameDataの妥当性をチェック
     * 
     * @return 妥当な場合true、そうでない場合false
     * 
     * 妥当性チェック項目:
     * - frame_idが0以上
     * - timestampが0以上
     * - imageが空でない
     * - imageのサイズが妥当（width > 0 && height > 0）
     */
    bool validate() const {
        // 基本情報チェック
        if (frame_id < 0) {
            return false;
        }
        if (timestamp < 0.0) {
            return false;
        }
        
        // 画像データチェック
        if (image.empty()) {
            return false;
        }
        if (image.cols <= 0 || image.rows <= 0) {
            return false;
        }
        
        return true;
    }
};

} // namespace Integration
} // namespace VirtualAd
