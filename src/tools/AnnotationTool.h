// AnnotationTool.h
// バックネット座標アノテーションツール

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace VirtualAd {
namespace Tools {

/**
 * @brief バックネット4隅のアノテーション点
 */
struct BacknetCorners {
    cv::Point2f topLeft;
    cv::Point2f topRight;
    cv::Point2f bottomLeft;
    cv::Point2f bottomRight;
    int frameNumber;
    bool valid;

    BacknetCorners() : frameNumber(-1), valid(false) {}
};

/**
 * @brief アノテーションツールクラス
 * 
 * インタラクティブに映像フレームのバックネット4隅を指定し、
 * アノテーションデータを保存します。
 */
class AnnotationTool {
public:
    /**
     * @brief コンストラクタ
     * @param windowName ウィンドウ名
     */
    explicit AnnotationTool(const std::string& windowName = "Annotation Tool");

    /**
     * @brief デストラクタ
     */
    ~AnnotationTool();

    /**
     * @brief フレームをアノテーションする
     * @param frame アノテーション対象のフレーム
     * @param frameNumber フレーム番号
     * @return アノテーション結果
     */
    BacknetCorners annotateFrame(const cv::Mat& frame, int frameNumber);

    /**
     * @brief アノテーションデータをJSON形式で保存
     * @param outputPath 出力パス
     * @param annotations アノテーションデータのリスト
     * @return 成功時true
     */
    static bool saveAnnotations(const std::string& outputPath, 
                                const std::vector<BacknetCorners>& annotations);

    /**
     * @brief アノテーションデータをJSON形式から読み込み
     * @param inputPath 入力パス
     * @return アノテーションデータのリスト
     */
    static std::vector<BacknetCorners> loadAnnotations(const std::string& inputPath);

    /**
     * @brief 現在のアノテーション状態をリセット
     */
    void reset();

    /**
     * @brief 最後のポイントを削除（Undo）
     */
    void undo();

private:
    /**
     * @brief マウスコールバック
     */
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);

    /**
     * @brief 画面を描画
     */
    void draw();

    /**
     * @brief ヘルプテキストを描画
     */
    void drawHelp();

private:
    std::string windowName_;
    cv::Mat currentFrame_;
    cv::Mat displayFrame_;
    int currentFrameNumber_;

    std::vector<cv::Point2f> points_;
    bool annotationComplete_;

    // 描画設定
    static const int POINT_RADIUS = 5;
    static const cv::Scalar POINT_COLOR;
    static const cv::Scalar LINE_COLOR;
    static const cv::Scalar TEXT_COLOR;
};

} // namespace Tools
} // namespace VirtualAd
