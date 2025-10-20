#pragma once

#include "IVideoSource.h"
#include "IVideoOutput.h"
#include <memory>
#include <vector>
#include <chrono>

namespace VirtualAd {

/**
 * @brief フレーム処理パイプラインの統合クラス
 * 
 * 映像ソースからフレームを取得し、処理を行い、出力先に送る
 * パイプライン全体を管理します。
 */
class FramePipeline {
public:
    FramePipeline();
    ~FramePipeline();

    /**
     * @brief 映像ソースを設定する
     * @param source 映像ソース（nullptr可）
     */
    void setSource(std::shared_ptr<IVideoSource> source);

    /**
     * @brief 映像出力を追加する
     * @param output 映像出力
     */
    void addOutput(std::shared_ptr<IVideoOutput> output);

    /**
     * @brief すべての映像出力をクリアする
     */
    void clearOutputs();

    /**
     * @brief パイプラインを開始する
     * @return 成功時true、失敗時false
     */
    bool start();

    /**
     * @brief パイプラインを停止する
     */
    void stop();

    /**
     * @brief パイプラインが実行中か確認する
     * @return 実行中の場合true
     */
    bool isRunning() const;

    /**
     * @brief 1フレーム処理を実行する
     * @return 成功時true、失敗時false（フレーム取得失敗等）
     */
    bool processFrame();

    /**
     * @brief 現在のフレームレートを取得する
     * @return FPS値
     */
    double getCurrentFPS() const;

    /**
     * @brief 処理したフレーム総数を取得する
     * @return フレーム数
     */
    uint64_t getFrameCount() const;

private:
    std::shared_ptr<IVideoSource> m_source;
    std::vector<std::shared_ptr<IVideoOutput>> m_outputs;
    
    bool m_running;
    uint64_t m_frameCount;
    
    // FPS計測用
    std::chrono::steady_clock::time_point m_lastFpsUpdate;
    uint64_t m_lastFrameCount;
    double m_currentFPS;
};

} // namespace VirtualAd
