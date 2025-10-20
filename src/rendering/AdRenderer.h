/**
 * @file AdRenderer.h
 * @brief Virtual advertisement renderer with perspective transformation
 */

#ifndef VIRTUALAD_RENDERING_ADRENDERER_H
#define VIRTUALAD_RENDERING_ADRENDERER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace VirtualAd {
namespace Rendering {

/**
 * @class AdRenderer
 * @brief Renders virtual advertisements with perspective transformation
 * 
 * AdRendererは、カメラポーズ（rvec, tvec）を使用して、
 * 3D空間に定義されたバックネット平面に広告テクスチャを透視変換して配置します。
 * 
 * 主な機能:
 * - 3D平面定義（バックネット4隅）
 * - カメラポーズからの2D投影計算
 * - 透視変換によるテクスチャマッピング
 * - 複数のブレンディングモード（REPLACE, ALPHA_BLEND, ADDITIVE）
 * 
 * 使用例:
 * @code
 * AdRenderer renderer;
 * renderer.initialize(camera_matrix, dist_coeffs);
 * 
 * std::vector<cv::Point3f> backnet_3d = {...};
 * renderer.setBacknetPlane(backnet_3d);
 * 
 * cv::Mat ad_texture = cv::imread("ad.png");
 * renderer.setAdTexture(ad_texture);
 * 
 * cv::Mat output;
 * renderer.render(image, rvec, tvec, output);
 * @endcode
 */
class AdRenderer {
public:
    /**
     * @brief ブレンディングモード
     */
    enum class BlendMode {
        REPLACE,      ///< 完全置き換え（広告領域を元画像と置き換え）
        ALPHA_BLEND,  ///< アルファブレンディング（透明度合成）
        ADDITIVE      ///< 加算合成（明るさを加算）
    };

    /**
     * @brief コンストラクタ
     */
    AdRenderer();

    /**
     * @brief デストラクタ
     */
    ~AdRenderer();

    /**
     * @brief 初期化
     * 
     * @param camera_matrix カメラ行列（3x3）
     * @param dist_coeffs 歪み係数（4, 5, 8, 12, 14要素）
     * @return 成功時true、失敗時false
     */
    bool initialize(const cv::Mat& camera_matrix, 
                   const cv::Mat& dist_coeffs);

    /**
     * @brief バックネット3D平面設定
     * 
     * @param corners_3d バックネット4隅の3D座標（左上、右上、右下、左下の順）
     * 
     * 例: 10m × 5mのバックネット、カメラから10m離れた位置
     * @code
     * std::vector<cv::Point3f> corners = {
     *     cv::Point3f(-5.0f, 2.5f, 10.0f),  // 左上
     *     cv::Point3f( 5.0f, 2.5f, 10.0f),  // 右上
     *     cv::Point3f( 5.0f, -2.5f, 10.0f), // 右下
     *     cv::Point3f(-5.0f, -2.5f, 10.0f)  // 左下
     * };
     * @endcode
     */
    void setBacknetPlane(const std::vector<cv::Point3f>& corners_3d);

    /**
     * @brief 広告テクスチャ設定
     * 
     * @param ad_texture 広告画像（CV_8UC3）
     * @return 成功時true、失敗時false
     */
    bool setAdTexture(const cv::Mat& ad_texture);

    /**
     * @brief レンダリング実行
     * 
     * @param image 入力画像（CV_8UC3）
     * @param rvec 回転ベクトル（3x1）
     * @param tvec 並進ベクトル（3x1）
     * @param output 出力画像（CV_8UC3）
     * @return 成功時true、失敗時false
     */
    bool render(const cv::Mat& image,
               const cv::Mat& rvec,
               const cv::Mat& tvec,
               cv::Mat& output);

    /**
     * @brief ブレンディングモード設定
     * 
     * @param mode ブレンディングモード
     */
    void setBlendMode(BlendMode mode);

    /**
     * @brief アルファ値設定（ALPHA_BLENDモード用）
     * 
     * @param alpha 透明度（0.0=完全透明、1.0=完全不透明）
     */
    void setAlpha(float alpha);

    /**
     * @brief 処理時間取得
     * 
     * @return 最後のrender()呼び出しの処理時間（ミリ秒）
     */
    double getProcessingTime() const;

    /**
     * @brief 最後のエラーメッセージ取得
     * 
     * @return エラーメッセージ（エラーがない場合は空文字列）
     */
    std::string getLastError() const;

    /**
     * @brief 初期化状態確認
     * 
     * @return 初期化済みの場合true
     */
    bool isInitialized() const;

private:
    /**
     * @brief 3D点を2Dに投影
     * 
     * @param rvec 回転ベクトル
     * @param tvec 並進ベクトル
     * @param projected_points 投影された2D点
     */
    void projectPoints(const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      std::vector<cv::Point2f>& projected_points);

    /**
     * @brief 透視変換行列計算
     * 
     * @param src_points ソース画像の4隅
     * @param dst_points 投影先の4隅
     * @param transform_matrix 透視変換行列（出力）
     */
    void computePerspectiveTransform(
        const std::vector<cv::Point2f>& src_points,
        const std::vector<cv::Point2f>& dst_points,
        cv::Mat& transform_matrix);

    /**
     * @brief テクスチャ適用
     * 
     * @param image 入力画像
     * @param transform_matrix 透視変換行列
     * @param ad_texture 広告テクスチャ
     * @param output 出力画像
     */
    void applyTexture(const cv::Mat& image,
                     const cv::Mat& transform_matrix,
                     const cv::Mat& ad_texture,
                     cv::Mat& output);

    /**
     * @brief 入力検証
     * 
     * @param image 入力画像
     * @param rvec 回転ベクトル
     * @param tvec 並進ベクトル
     * @return 入力が有効な場合true
     */
    bool validateInputs(const cv::Mat& image,
                       const cv::Mat& rvec,
                       const cv::Mat& tvec);

    // メンバ変数
    cv::Mat camera_matrix_;                    ///< カメラ行列
    cv::Mat dist_coeffs_;                      ///< 歪み係数
    std::vector<cv::Point3f> backnet_corners_3d_; ///< バックネット3D座標
    cv::Mat ad_texture_;                       ///< 広告テクスチャ
    BlendMode blend_mode_;                     ///< ブレンディングモード
    float alpha_;                              ///< アルファ値（0.0-1.0）
    double processing_time_;                   ///< 処理時間（ミリ秒）
    std::string last_error_;                   ///< エラーメッセージ
    bool is_initialized_;                      ///< 初期化済みフラグ
};

} // namespace Rendering
} // namespace VirtualAd

#endif // VIRTUALAD_RENDERING_ADRENDERER_H
