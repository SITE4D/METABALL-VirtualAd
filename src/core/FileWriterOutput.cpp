// FileWriterOutput.cpp
// OpenCV VideoWriterベースのファイル録画出力の実装

#include "FileWriterOutput.h"
#include <iostream>

namespace VirtualAd {
namespace Core {

FileWriterOutput::FileWriterOutput(const std::string& outputPath, Codec codec)
    : outputPath_(outputPath)
    , codec_(codec)
    , width_(0)
    , height_(0)
    , fps_(0.0)
    , writtenFrameCount_(0)
{
}

FileWriterOutput::~FileWriterOutput()
{
    close();
}

int FileWriterOutput::getCodecFourCC() const
{
    switch (codec_) {
        case Codec::H264:
            return cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        case Codec::H265:
            return cv::VideoWriter::fourcc('h', 'e', 'v', '1');
        case Codec::MJPEG:
            return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        case Codec::MP4V:
            return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        default:
            return cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // デフォルトはH.264
    }
}

std::string FileWriterOutput::getCodecName() const
{
    switch (codec_) {
        case Codec::H264:  return "H.264 (AVC)";
        case Codec::H265:  return "H.265 (HEVC)";
        case Codec::MJPEG: return "Motion JPEG";
        case Codec::MP4V:  return "MPEG-4 Part 2";
        default:           return "Unknown";
    }
}

bool FileWriterOutput::open(int width, int height, double fps)
{
    if (writer_.isOpened()) {
        std::cerr << "[FileWriterOutput] 既にファイルが開いています: " << outputPath_ << std::endl;
        return false;
    }

    width_ = width;
    height_ = height;
    fps_ = fps;

    // VideoWriterを開く
    int fourcc = getCodecFourCC();
    writer_.open(outputPath_, fourcc, fps, cv::Size(width, height), true);

    if (!writer_.isOpened()) {
        std::cerr << "[FileWriterOutput] ファイルを開けませんでした: " << outputPath_ << std::endl;
        std::cerr << "  コーデック: " << getCodecName() << std::endl;
        std::cerr << "  解像度: " << width << "x" << height << std::endl;
        std::cerr << "  FPS: " << fps << std::endl;
        return false;
    }

    writtenFrameCount_ = 0;

    std::cout << "[FileWriterOutput] ファイルを開きました: " << outputPath_ << std::endl;
    std::cout << "  コーデック: " << getCodecName() << std::endl;
    std::cout << "  解像度: " << width_ << "x" << height_ << std::endl;
    std::cout << "  FPS: " << fps_ << std::endl;

    return true;
}

void FileWriterOutput::close()
{
    if (!writer_.isOpened()) {
        return;
    }

    writer_.release();

    std::cout << "[FileWriterOutput] ファイルを閉じました: " << outputPath_ << std::endl;
    std::cout << "  書き込んだフレーム数: " << writtenFrameCount_ << std::endl;
    std::cout << "  総時間: " << (writtenFrameCount_ / fps_) << " sec" << std::endl;
}

bool FileWriterOutput::writeFrame(const cv::Mat& frame)
{
    if (!writer_.isOpened()) {
        std::cerr << "[FileWriterOutput] ファイルが開いていません" << std::endl;
        return false;
    }

    if (frame.empty()) {
        std::cerr << "[FileWriterOutput] 空のフレームを受け取りました" << std::endl;
        return false;
    }

    // フレームサイズチェック
    if (frame.cols != width_ || frame.rows != height_) {
        std::cerr << "[FileWriterOutput] フレームサイズが一致しません: " 
                  << frame.cols << "x" << frame.rows 
                  << " (期待: " << width_ << "x" << height_ << ")" << std::endl;
        return false;
    }

    // フレームを書き込み
    writer_.write(frame);

    writtenFrameCount_++;

    return true;
}

bool FileWriterOutput::isOpened() const
{
    return writer_.isOpened();
}

std::string FileWriterOutput::getName() const
{
    return "FileWriterOutput[" + outputPath_ + "]";
}

} // namespace Core
} // namespace VirtualAd
