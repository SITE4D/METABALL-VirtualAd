// FileWriterOutput.h
// OpenCV VideoWriterベースのファイル録画出力

#pragma once

#include "IVideoOutput.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace VirtualAd {
namespace Core {

/**
 * @brief OpenCVのVideoWriterを使用したファイル録画出力クラス
 * 
 * 映像フレームをファイルに書き込みます。
 * H.264/H.265コーデックをサポートします。
 */
class FileWriterOutput : public IVideoOutput {
public:
    /**
     * @brief コーデック種別
     */
    enum class Codec {
        H264,       // H.264 (AVC) - 'avc1' FourCC
        H265,       // H.265 (HEVC) - 'hev1' FourCC
        MJPEG,      // Motion JPEG - 'MJPG' FourCC
        MP4V        // MPEG-4 Part 2 - 'mp4v' FourCC
    };

    /**
     * @brief コンストラクタ
     * @param outputPath 出力ファイルパス
     * @param codec コーデック種別（デフォルト: H264）
     */
    explicit FileWriterOutput(const std::string& outputPath, Codec codec = Codec::H264);
    
    /**
     * @brief デストラクタ
     */
    virtual ~FileWriterOutput();

    // IVideoOutputインターフェース実装
    bool open(int width, int height, double fps) override;
    void close() override;
    bool writeFrame(const cv::Mat& frame) override;
    bool isOpened() const override;
    std::string getName() const override;

    /**
     * @brief 統計情報を取得
     */
    int getWrittenFrameCount() const { return writtenFrameCount_; }
    std::string getOutputPath() const { return outputPath_; }

private:
    /**
     * @brief コーデック種別からFourCCコードを取得
     */
    int getCodecFourCC() const;

    /**
     * @brief コーデック名を取得
     */
    std::string getCodecName() const;

private:
    std::string outputPath_;        // 出力ファイルパス
    Codec codec_;                   // コーデック種別
    cv::VideoWriter writer_;        // OpenCV VideoWriter
    int width_;                     // ビデオ幅
    int height_;                    // ビデオ高さ
    double fps_;                    // フレームレート
    int writtenFrameCount_;         // 書き込んだフレーム数
};

} // namespace Core
} // namespace VirtualAd
