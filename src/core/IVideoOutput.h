#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace VirtualAd {

/**
 * @brief 映像出力の抽象インターフェース
 * 
 * ファイル録画、HDMI出力、プレビューウィンドウなど、
 * あらゆる映像出力先の共通インターフェースを定義します。
 */
class IVideoOutput {
public:
    virtual ~IVideoOutput() = default;

    /**
     * @brief 映像出力を開始する
     * @param width フレーム幅（ピクセル）
     * @param height フレーム高さ（ピクセル）
     * @param fps フレームレート
     * @return 成功時true、失敗時false
     */
    virtual bool open(int width, int height, double fps) = 0;

    /**
     * @brief 映像出力を停止する
     */
    virtual void close() = 0;

    /**
     * @brief フレームを出力する
     * @param frame 出力するフレーム画像
     * @return 成功時true、失敗時false
     */
    virtual bool writeFrame(const cv::Mat& frame) = 0;

    /**
     * @brief 映像出力が開いているか確認する
     * @return 開いている場合true、そうでない場合false
     */
    virtual bool isOpened() const = 0;

    /**
     * @brief 出力先の名前を取得する
     * @return 出力先名（ファイルパス、デバイス名など）
     */
    virtual std::string getName() const = 0;
};

} // namespace VirtualAd
