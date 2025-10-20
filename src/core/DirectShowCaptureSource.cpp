// DirectShowCaptureSource.cpp
// DirectShow APIを使用したライブキャプチャソースの実装

#include "DirectShowCaptureSource.h"
#include <iostream>
#include <chrono>

namespace VirtualAd {

DirectShowCaptureSource::DirectShowCaptureSource(int deviceIndex)
    : m_deviceIndex(deviceIndex)
    , m_width(0)
    , m_height(0)
    , m_fps(0.0)
    , m_isOpened(false)
    , m_isRunning(false)
    , m_frameCount(0)
{
}

DirectShowCaptureSource::~DirectShowCaptureSource()
{
    stop();
}

bool DirectShowCaptureSource::start()
{
    if (m_isOpened) {
        std::cerr << "[DirectShowCaptureSource] Already opened" << std::endl;
        return false;
    }

    // OpenCV VideoCapture with DirectShow backend (CAP_DSHOW)
    m_capture.open(m_deviceIndex, cv::CAP_DSHOW);

    if (!m_capture.isOpened()) {
        std::cerr << "[DirectShowCaptureSource] Failed to open device: " << m_deviceIndex << std::endl;
        return false;
    }

    // デバイス情報を取得
    m_width = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    m_height = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_fps = m_capture.get(cv::CAP_PROP_FPS);

    m_isOpened = true;
    m_isRunning = true;
    m_frameCount = 0;
    m_startTime = std::chrono::steady_clock::now();

    std::cout << "[DirectShowCaptureSource] Device opened: " << m_deviceIndex << std::endl;
    std::cout << "  Resolution: " << m_width << "x" << m_height << std::endl;
    std::cout << "  FPS: " << m_fps << std::endl;

    return true;
}

void DirectShowCaptureSource::stop()
{
    if (!m_isOpened) {
        return;
    }

    m_capture.release();
    m_isOpened = false;
    m_isRunning = false;

    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(endTime - m_startTime).count();

    std::cout << "[DirectShowCaptureSource] Device closed" << std::endl;
    std::cout << "  Captured frames: " << m_frameCount << std::endl;
    std::cout << "  Duration: " << elapsed << " sec" << std::endl;
    if (elapsed > 0) {
        std::cout << "  Average FPS: " << (m_frameCount / static_cast<double>(elapsed)) << std::endl;
    }
}

cv::Mat DirectShowCaptureSource::getFrame()
{
    if (!m_isOpened || !m_isRunning) {
        return cv::Mat();
    }

    cv::Mat frame;
    if (!m_capture.read(frame)) {
        std::cerr << "[DirectShowCaptureSource] Failed to read frame" << std::endl;
        return cv::Mat();
    }

    m_frameCount++;

    return frame;
}

bool DirectShowCaptureSource::isOpened() const
{
    return m_isOpened;
}

double DirectShowCaptureSource::getFrameRate() const
{
    return m_fps;
}

int DirectShowCaptureSource::getWidth() const
{
    return m_width;
}

int DirectShowCaptureSource::getHeight() const
{
    return m_height;
}

std::string DirectShowCaptureSource::getName() const
{
    return "DirectShowCaptureSource[Device " + std::to_string(m_deviceIndex) + "]";
}

std::vector<DirectShowCaptureSource::DeviceInfo> DirectShowCaptureSource::enumerateDevices()
{
    std::vector<DeviceInfo> devices;

    // OpenCVを使用してデバイスを列挙
    // 最大10デバイスを試行
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture cap(i, cv::CAP_DSHOW);
        if (cap.isOpened()) {
            DeviceInfo info;
            info.deviceIndex = i;
            info.name = "Camera " + std::to_string(i);
            info.description = "DirectShow Device " + std::to_string(i);

            devices.push_back(info);
            cap.release();
        }
    }

    std::cout << "[DirectShowCaptureSource] Found " << devices.size() << " devices" << std::endl;

    return devices;
}

bool DirectShowCaptureSource::setResolution(int width, int height)
{
    if (!m_isOpened) {
        std::cerr << "[DirectShowCaptureSource] Device not opened" << std::endl;
        return false;
    }

    bool success = m_capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    success &= m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    if (success) {
        m_width = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
        m_height = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));

        std::cout << "[DirectShowCaptureSource] Resolution set to: " << m_width << "x" << m_height << std::endl;
    } else {
        std::cerr << "[DirectShowCaptureSource] Failed to set resolution" << std::endl;
    }

    return success;
}

bool DirectShowCaptureSource::setFrameRate(double fps)
{
    if (!m_isOpened) {
        std::cerr << "[DirectShowCaptureSource] Device not opened" << std::endl;
        return false;
    }

    bool success = m_capture.set(cv::CAP_PROP_FPS, fps);

    if (success) {
        m_fps = m_capture.get(cv::CAP_PROP_FPS);
        std::cout << "[DirectShowCaptureSource] FPS set to: " << m_fps << std::endl;
    } else {
        std::cerr << "[DirectShowCaptureSource] Failed to set FPS" << std::endl;
    }

    return success;
}

} // namespace VirtualAd
