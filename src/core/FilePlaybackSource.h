#pragma once

#include "IVideoSource.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

namespace VirtualAd {

/**
 * @brief ファイル再生ソースの実装
 * 
 * 映像ファイル（MP4など）を読み込み、指定されたフレームレートで
 * 再生するビデオソースです。
 */
class FilePlaybackSource : public IVideoSource {
public:
    /**
     * @brief コンストラクタ
     * @param filePath 映像ファイルのパス
     */
    explicit FilePlaybackSource(const std::string& filePath);

    /**
     * @brief デストラクタ
     */
    ~FilePlaybackSource() override;

    // IVideoSource インターフェース実装
    bool start() override;
    void stop() override;
    cv::Mat getFrame() override;
    bool isOpened() const override;
    double getFrameRate() const override;
    int getWidth() const override;
    int getHeight() const override;
    std::string getName() const override;

    /**
     * @brief 指定したフレーム位置にシークする
     * @param frameNumber フレーム番号（0始まり）
     * @return 成功時true
     */
    bool seekToFrame(int frameNumber);

    /**
     * @brief 現在のフレーム位置を取得する
     * @return 現在のフレーム番号
     */
    int getCurrentFrameNumber() const;

    /**
     * @brief 総フレーム数を取得する
     * @return 総フレーム数
     */
    int getTotalFrames() const;

    /**
     * @brief 一時停止/再開を切り替える
     */
    void togglePause();

    /**
     * @brief 一時停止状態かどうかを取得する
     * @return 一時停止中の場合true
     */
    bool isPaused() const;

private:
    std::string m_filePath;
    cv::VideoCapture m_capture;
    
    double m_fps;
    int m_width;
    int m_height;
    int m_totalFrames;
    
    bool m_isOpened;
    bool m_isPaused;
    
    // フレームレート制御用
    std::chrono::steady_clock::time_point m_lastFrameTime;
    std::chrono::microseconds m_frameDuration;
};

} // namespace VirtualAd
