// WindowPreviewOutput.cpp
// OpenCV imshow()ベースのシンプルなウィンドウプレビュー出力の実装

#include "WindowPreviewOutput.h"
#include <iostream>

namespace VirtualAd {
namespace Core {

WindowPreviewOutput::WindowPreviewOutput(const std::string& windowName, int waitKeyDelay)
    : windowName_(windowName)
    , waitKeyDelay_(waitKeyDelay)
    , opened_(false)
    , width_(0)
    , height_(0)
    , lastKey_(-1)
    , displayedFrameCount_(0)
{
}

WindowPreviewOutput::~WindowPreviewOutput()
{
    close();
}

bool WindowPreviewOutput::open(int width, int height, double fps)
{
    if (opened_) {
        std::cerr << "[WindowPreviewOutput] 既にウィンドウが開いています: " << windowName_ << std::endl;
        return false;
    }

    width_ = width;
    height_ = height;

    // OpenCVウィンドウを作成
    cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);
    opened_ = true;
    displayedFrameCount_ = 0;

    std::cout << "[WindowPreviewOutput] ウィンドウを開きました: " << windowName_ 
              << " (" << width_ << "x" << height_ << " @ " << fps << " fps)" << std::endl;

    return true;
}

void WindowPreviewOutput::close()
{
    if (!opened_) {
        return;
    }

    cv::destroyWindow(windowName_);
    opened_ = false;

    std::cout << "[WindowPreviewOutput] ウィンドウを閉じました: " << windowName_ 
              << " (表示フレーム数: " << displayedFrameCount_ << ")" << std::endl;
}

bool WindowPreviewOutput::writeFrame(const cv::Mat& frame)
{
    if (!opened_) {
        std::cerr << "[WindowPreviewOutput] ウィンドウが開いていません" << std::endl;
        return false;
    }

    if (frame.empty()) {
        std::cerr << "[WindowPreviewOutput] 空のフレームを受け取りました" << std::endl;
        return false;
    }

    // フレームを表示
    cv::imshow(windowName_, frame);

    // キー入力を処理（waitKeyDelay_ミリ秒待機）
    lastKey_ = cv::waitKey(waitKeyDelay_);

    displayedFrameCount_++;

    return true;
}

bool WindowPreviewOutput::isOpened() const
{
    return opened_;
}

std::string WindowPreviewOutput::getName() const
{
    return "WindowPreviewOutput[" + windowName_ + "]";
}

} // namespace Core
} // namespace VirtualAd
