// WindowPreviewOutput.h
// OpenCV imshow()ベースのシンプルなウィンドウプレビュー出力

#pragma once

#include "IVideoOutput.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace VirtualAd {
namespace Core {

/**
 * @brief OpenCVのimshow()を使用したウィンドウプレビュー出力クラス
 * 
 * シンプルなウィンドウにフレームを表示します。
 * デバッグやテスト目的に最適です。
 */
class WindowPreviewOutput : public IVideoOutput {
public:
    /**
     * @brief コンストラクタ
     * @param windowName ウィンドウ名（デフォルト: "Preview"）
     * @param waitKeyDelay cv::waitKey()の遅延（ms）。0=無限待機、1=1ms待機（デフォルト）
     */
    explicit WindowPreviewOutput(const std::string& windowName = "Preview", int waitKeyDelay = 1);
    
    /**
     * @brief デストラクタ
     */
    virtual ~WindowPreviewOutput();

    // IVideoOutputインターフェース実装
    bool open(int width, int height, double fps) override;
    void close() override;
    bool writeFrame(const cv::Mat& frame) override;
    bool isOpened() const override;
    std::string getName() const override;

    /**
     * @brief 最後に押されたキーを取得
     * @return キーコード（何も押されていない場合は-1）
     */
    int getLastKey() const { return lastKey_; }

    /**
     * @brief 統計情報を取得
     */
    int getDisplayedFrameCount() const { return displayedFrameCount_; }

private:
    std::string windowName_;    // ウィンドウ名
    int waitKeyDelay_;          // cv::waitKey()の遅延
    bool opened_;               // ウィンドウが開いているか
    int width_;                 // ウィンドウ幅
    int height_;                // ウィンドウ高さ
    int lastKey_;               // 最後に押されたキー
    int displayedFrameCount_;   // 表示したフレーム数
};

} // namespace Core
} // namespace VirtualAd
