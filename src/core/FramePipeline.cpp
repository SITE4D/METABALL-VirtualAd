#include "FramePipeline.h"
#include <iostream>

namespace VirtualAd {

FramePipeline::FramePipeline()
    : m_source(nullptr)
    , m_running(false)
    , m_frameCount(0)
    , m_lastFrameCount(0)
    , m_currentFPS(0.0)
{
    m_lastFpsUpdate = std::chrono::steady_clock::now();
}

FramePipeline::~FramePipeline()
{
    stop();
}

void FramePipeline::setSource(std::shared_ptr<IVideoSource> source)
{
    if (m_running) {
        std::cerr << "Error: Cannot change source while pipeline is running" << std::endl;
        return;
    }
    m_source = source;
}

void FramePipeline::addOutput(std::shared_ptr<IVideoOutput> output)
{
    if (output) {
        m_outputs.push_back(output);
    }
}

void FramePipeline::clearOutputs()
{
    if (m_running) {
        std::cerr << "Error: Cannot clear outputs while pipeline is running" << std::endl;
        return;
    }
    m_outputs.clear();
}

bool FramePipeline::start()
{
    if (m_running) {
        std::cerr << "Warning: Pipeline is already running" << std::endl;
        return true;
    }

    if (!m_source) {
        std::cerr << "Error: No video source set" << std::endl;
        return false;
    }

    if (!m_source->start()) {
        std::cerr << "Error: Failed to start video source" << std::endl;
        return false;
    }

    // 出力を開く
    int width = m_source->getWidth();
    int height = m_source->getHeight();
    double fps = m_source->getFrameRate();

    for (auto& output : m_outputs) {
        if (!output->open(width, height, fps)) {
            std::cerr << "Warning: Failed to open output: " << output->getName() << std::endl;
        }
    }

    m_running = true;
    m_frameCount = 0;
    m_lastFrameCount = 0;
    m_currentFPS = 0.0;
    m_lastFpsUpdate = std::chrono::steady_clock::now();

    std::cout << "Pipeline started: " << width << "x" << height << " @ " << fps << " fps" << std::endl;
    return true;
}

void FramePipeline::stop()
{
    if (!m_running) {
        return;
    }

    m_running = false;

    if (m_source) {
        m_source->stop();
    }

    for (auto& output : m_outputs) {
        output->close();
    }

    std::cout << "Pipeline stopped. Total frames processed: " << m_frameCount << std::endl;
}

bool FramePipeline::isRunning() const
{
    return m_running;
}

bool FramePipeline::processFrame()
{
    if (!m_running) {
        std::cerr << "Error: Pipeline is not running" << std::endl;
        return false;
    }

    if (!m_source || !m_source->isOpened()) {
        std::cerr << "Error: Video source is not opened" << std::endl;
        return false;
    }

    // フレーム取得
    cv::Mat frame = m_source->getFrame();
    if (frame.empty()) {
        return false;
    }

    // 全ての出力に書き込み
    for (auto& output : m_outputs) {
        if (output->isOpened()) {
            output->writeFrame(frame);
        }
    }

    m_frameCount++;

    // FPS計測（1秒ごとに更新）
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastFpsUpdate).count();
    if (elapsed >= 1000) {
        uint64_t framesDiff = m_frameCount - m_lastFrameCount;
        m_currentFPS = static_cast<double>(framesDiff) / (elapsed / 1000.0);
        m_lastFrameCount = m_frameCount;
        m_lastFpsUpdate = now;
    }

    return true;
}

double FramePipeline::getCurrentFPS() const
{
    return m_currentFPS;
}

uint64_t FramePipeline::getFrameCount() const
{
    return m_frameCount;
}

} // namespace VirtualAd
