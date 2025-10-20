// DirectShowCaptureSource.h
// DirectShow APIを使用したライブキャプチャソース

#pragma once

#include "IVideoSource.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace VirtualAd {

/**
 * @brief DirectShowを使用したライブキャプチャソースの実装
 * 
 * Windowsネイティブのキャプチャデバイスから映像を取得します。
 * 60fps対応を目指します。
 */
class DirectShowCaptureSource : public IVideoSource {
public:
    /**
     * @brief デバイス情報
     */
    struct DeviceInfo {
        std::string name;           // デバイス名
        int deviceIndex;            // デバイスインデックス
        std::string description;    // 説明
    };

    /**
     * @brief コンストラクタ
     * @param deviceIndex キャプチャデバイスのインデックス（デフォルト: 0）
     */
    explicit DirectShowCaptureSource(int deviceIndex = 0);

    /**
     * @brief デストラクタ
     */
    ~DirectShowCaptureSource() override;

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
     * @brief 利用可能なキャプチャデバイスを列挙する
     * @return デバイス情報のリスト
     */
    static std::vector<DeviceInfo> enumerateDevices();

    /**
     * @brief キャプチャ解像度を設定する
     * @param width 幅
     * @param height 高さ
     * @return 成功時true
     */
    bool setResolution(int width, int height);

    /**
     * @brief キャプチャフレームレートを設定する
     * @param fps フレームレート
     * @return 成功時true
     */
    bool setFrameRate(double fps);

private:
    int m_deviceIndex;
    cv::VideoCapture m_capture;
    
    int m_width;
    int m_height;
    double m_fps;
    
    bool m_isOpened;
    bool m_isRunning;
    
    // フレーム統計
    uint64_t m_frameCount;
    std::chrono::steady_clock::time_point m_startTime;
};

} // namespace VirtualAd
