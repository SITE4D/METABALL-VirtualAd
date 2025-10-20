#include "FilePlaybackSource.h"
#include <iostream>
#include <thread>

namespace VirtualAd {

FilePlaybackSource::FilePlaybackSource(const std::string& filePath)
    : m_filePath(filePath)
    , m_fps(0.0)
    , m_width(0)
    , m_height(0)
    , m_totalFrames(0)
    , m_isOpened(false)
    , m_isPaused(false)
{
    // フレームレートは後で設定される
    m_frameDuration = std::chrono::microseconds(0);
}

FilePlaybackSource::~FilePlaybackSource()
{
    stop();
}

bool FilePlaybackSource::start()
{
    if (m_isOpened) {
        std::cout << "FilePlaybackSource: Already opened" << std::endl;
        return true;
    }

    // ファイルを開く
    m_capture.open(m_filePath);
    if (!m_capture.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << m_filePath << std::endl;
        return false;
    }

    // ビデオプロパティを取得
    m_fps = m_capture.get(cv::CAP_PROP_FPS);
    m_width = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    m_height = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_totalFrames = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_COUNT));

    if (m_fps <= 0.0) {
        std::cerr << "Error: Invalid FPS: " << m_fps << std::endl;
        m_capture.release();
        return false;
    }

    // フレーム間隔を計算（マイクロ秒）
    m_frameDuration = std::chrono::microseconds(static_cast<int64_t>(1000000.0 / m_fps));

    m_isOpened = true;
    m_isPaused = false;
    m_lastFrameTime = std::chrono::steady_clock::now();

    std::cout << "FilePlaybackSource opened: " << m_filePath << std::endl;
    std::cout << "  Resolution: " << m_width << "x" << m_height << std::endl;
    std::cout << "  FPS: " << m_fps << std::endl;
    std::cout << "  Total frames: " << m_totalFrames << std::endl;

    return true;
}

void FilePlaybackSource::stop()
{
    if (m_capture.isOpened()) {
        m_capture.release();
    }
    m_isOpened = false;
    m_isPaused = false;
}

cv::Mat FilePlaybackSource::getFrame()
{
    if (!m_isOpened) {
        std::cerr << "Error: Video source is not opened" << std::endl;
        return cv::Mat();
    }

    if (m_isPaused) {
        // 一時停止中は前回のフレームを返す
        // （実際には前回のフレームを保持していないので空のMatを返す）
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 60fps相当の待機
        return cv::Mat();
    }

    // フレームレート制御：前回のフレームからの経過時間を確認
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastFrameTime);
    
    if (elapsed < m_frameDuration) {
        // まだ次のフレームを出すタイミングではない
        auto waitTime = m_frameDuration - elapsed;
        std::this_thread::sleep_for(waitTime);
    }

    // フレーム読み込み
    cv::Mat frame;
    if (!m_capture.read(frame)) {
        // ファイル終端に達した場合、先頭に戻る（ループ再生）
        m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!m_capture.read(frame)) {
            std::cerr << "Error: Failed to read frame" << std::endl;
            return cv::Mat();
        }
        std::cout << "Video loop: Restarting from beginning" << std::endl;
    }

    m_lastFrameTime = std::chrono::steady_clock::now();
    return frame;
}

bool FilePlaybackSource::isOpened() const
{
    return m_isOpened && m_capture.isOpened();
}

double FilePlaybackSource::getFrameRate() const
{
    return m_fps;
}

int FilePlaybackSource::getWidth() const
{
    return m_width;
}

int FilePlaybackSource::getHeight() const
{
    return m_height;
}

std::string FilePlaybackSource::getName() const
{
    return m_filePath;
}

bool FilePlaybackSource::seekToFrame(int frameNumber)
{
    if (!m_isOpened) {
        std::cerr << "Error: Cannot seek - video not opened" << std::endl;
        return false;
    }

    if (frameNumber < 0 || frameNumber >= m_totalFrames) {
        std::cerr << "Error: Frame number out of range: " << frameNumber << std::endl;
        return false;
    }

    bool success = m_capture.set(cv::CAP_PROP_POS_FRAMES, frameNumber);
    if (success) {
        m_lastFrameTime = std::chrono::steady_clock::now();
    }
    return success;
}

int FilePlaybackSource::getCurrentFrameNumber() const
{
    if (!m_isOpened) {
        return -1;
    }
    return static_cast<int>(m_capture.get(cv::CAP_PROP_POS_FRAMES));
}

int FilePlaybackSource::getTotalFrames() const
{
    return m_totalFrames;
}

void FilePlaybackSource::togglePause()
{
    m_isPaused = !m_isPaused;
    std::cout << "Playback " << (m_isPaused ? "paused" : "resumed") << std::endl;
}

bool FilePlaybackSource::isPaused() const
{
    return m_isPaused;
}

} // namespace VirtualAd
