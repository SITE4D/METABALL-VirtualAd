#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace VirtualAd {

/**
 * @brief 映像ソースの抽象インターフェース
 * 
 * ライブキャプチャ、ファイル再生など、あらゆる映像入力ソースの
 * 共通インターフェースを定義します。
 */
class IVideoSource {
public:
    virtual ~IVideoSource() = default;

    /**
     * @brief 映像ソースを開始する
     * @return 成功時true、失敗時false
     */
    virtual bool start() = 0;

    /**
     * @brief 映像ソースを停止する
     */
    virtual void stop() = 0;

    /**
     * @brief 次のフレームを取得する
     * @return フレーム画像（取得失敗時は空のMat）
     */
    virtual cv::Mat getFrame() = 0;

    /**
     * @brief 映像ソースが開いているか確認する
     * @return 開いている場合true、そうでない場合false
     */
    virtual bool isOpened() const = 0;

    /**
     * @brief フレームレートを取得する
     * @return FPS値（取得できない場合は0.0）
     */
    virtual double getFrameRate() const = 0;

    /**
     * @brief フレーム幅を取得する
     * @return フレーム幅（ピクセル）
     */
    virtual int getWidth() const = 0;

    /**
     * @brief フレーム高さを取得する
     * @return フレーム高さ（ピクセル）
     */
    virtual int getHeight() const = 0;

    /**
     * @brief 映像ソースの名前を取得する
     * @return ソース名（デバイス名、ファイルパスなど）
     */
    virtual std::string getName() const = 0;
};

} // namespace VirtualAd
