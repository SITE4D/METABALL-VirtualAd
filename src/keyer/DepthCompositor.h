/**
 * @file DepthCompositor.h
 * @brief Depth-based compositing for virtual advertisement
 * 
 * セグメンテーションマスクとデプスマップを使用して、
 * バーチャル広告と元画像を合成します。
 */

#ifndef VIRTUALAD_KEYER_DEPTHCOMPOSITOR_H
#define VIRTUALAD_KEYER_DEPTHCOMPOSITOR_H

#include <opencv2/opencv.hpp>
#include <string>

namespace VirtualAd {
namespace Keyer {

/**
 * @class DepthCompositor
 * @brief デプスベース合成クラス
 * 
 * セグメンテーションマスクとデプスマップを使用して、
 * 選手がバーチャル広告の前に自然に表示されるように合成します。
 */
class DepthCompositor {
public:
    /**
     * @brief セグメンテーションクラスID
     */
    enum class SegmentationClass : int {
        BACKGROUND = 0,  ///< 背景
        PLAYER = 1,      ///< 選手
        UMPIRE = 2,      ///< 審判
        BACKNET = 3      ///< バックネット
    };
    
    /**
     * @brief コンストラクタ
     */
    DepthCompositor();
    
    /**
     * @brief デストラクタ
     */
    ~DepthCompositor();
    
    /**
     * @brief デプスベース合成を実行
     * 
     * @param image 入力画像（BGR、8UC3）
     * @param segmentation_mask セグメンテーションマスク（8UC1、クラスID）
     * @param depth_map デプスマップ（32FC1、0.0-1.0、小さいほど手前）
     * @param ad_texture 広告テクスチャ（BGR、8UC3）
     * @param output 出力合成画像（BGR、8UC3）
     * @param backnet_depth バックネットの推定デプス値（デフォルト: 0.5）
     * @return 成功した場合true
     */
    bool composite(const cv::Mat& image,
                  const cv::Mat& segmentation_mask,
                  const cv::Mat& depth_map,
                  const cv::Mat& ad_texture,
                  cv::Mat& output,
                  float backnet_depth = 0.5f);
    
    /**
     * @brief シンプル合成を実行（デプス情報なし）
     * 
     * セグメンテーションマスクのみを使用した簡易合成。
     * バックネット領域を広告で置き換えます。
     * 
     * @param image 入力画像（BGR、8UC3）
     * @param segmentation_mask セグメンテーションマスク（8UC1、クラスID）
     * @param ad_texture 広告テクスチャ（BGR、8UC3）
     * @param output 出力合成画像（BGR、8UC3）
     * @return 成功した場合true
     */
    bool compositeSimple(const cv::Mat& image,
                        const cv::Mat& segmentation_mask,
                        const cv::Mat& ad_texture,
                        cv::Mat& output);
    
    /**
     * @brief 広告テクスチャをリサイズして画像サイズに合わせる
     * 
     * @param ad_texture 元の広告テクスチャ
     * @param target_size ターゲットサイズ
     * @param resized_ad リサイズされた広告テクスチャ
     */
    static void resizeAdTexture(const cv::Mat& ad_texture,
                               const cv::Size& target_size,
                               cv::Mat& resized_ad);
    
    /**
     * @brief デプス閾値を設定
     * 
     * @param threshold デプス閾値（0.0-1.0）
     */
    void setDepthThreshold(float threshold);
    
    /**
     * @brief デプス閾値を取得
     * 
     * @return デプス閾値
     */
    float getDepthThreshold() const;
    
    /**
     * @brief 最後のエラーメッセージを取得
     * 
     * @return エラーメッセージ
     */
    std::string getLastError() const;
    
    /**
     * @brief 処理時間を取得（ミリ秒）
     * 
     * @return 処理時間
     */
    double getProcessingTime() const;

private:
    /**
     * @brief 入力画像を検証
     * 
     * @param image 入力画像
     * @param segmentation_mask セグメンテーションマスク
     * @param depth_map デプスマップ
     * @param ad_texture 広告テクスチャ
     * @return すべて有効な場合true
     */
    bool validateInputs(const cv::Mat& image,
                       const cv::Mat& segmentation_mask,
                       const cv::Mat& depth_map,
                       const cv::Mat& ad_texture);
    
    /**
     * @brief デプスベースのピクセル単位合成
     * 
     * @param image 入力画像
     * @param segmentation_mask セグメンテーションマスク
     * @param depth_map デプスマップ
     * @param ad_texture 広告テクスチャ（リサイズ済み）
     * @param output 出力画像
     * @param backnet_depth バックネットデプス値
     */
    void compositePixelwise(const cv::Mat& image,
                           const cv::Mat& segmentation_mask,
                           const cv::Mat& depth_map,
                           const cv::Mat& ad_texture,
                           cv::Mat& output,
                           float backnet_depth);

private:
    float depth_threshold_;      ///< デプス閾値
    std::string last_error_;     ///< 最後のエラーメッセージ
    double processing_time_;     ///< 処理時間（ミリ秒）
};

} // namespace Keyer
} // namespace VirtualAd

#endif // VIRTUALAD_KEYER_DEPTHCOMPOSITOR_H
