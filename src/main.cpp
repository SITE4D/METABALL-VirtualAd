#include <iostream>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "core/FilePlaybackSource.h"
#include "core/WindowPreviewOutput.h"
#include "core/FileWriterOutput.h"
#include "core/FramePipeline.h"

/**
 * @file main.cpp
 * @brief METABALL Virtual Ad - メインエントリーポイント
 *
 * Phase 1 Step 3: 映像出力実装（WindowPreviewOutput, FileWriterOutput）
 * FramePipelineを使用した統合テスト
 */

void printBanner() {
    std::cout << "=============================================" << std::endl;
    std::cout << "   METABALL Virtual Ad System v1.0.0" << std::endl;
    std::cout << "   Phase 1: 映像I/Oパイプライン" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
}

void printSystemInfo() {
    std::cout << "[システム情報]" << std::endl;
    std::cout << "  OpenCV バージョン: " << CV_VERSION << std::endl;
    std::cout << "  C++ Standard: C++" << (__cplusplus / 100 % 100) << std::endl;
    std::cout << std::endl;
}

/**
 * @brief テスト1: FilePlaybackSource → WindowPreviewOutput
 * プレビューウィンドウに映像を表示
 */
void testPreviewOutput() {
    std::cout << "=============================================" << std::endl;
    std::cout << "[テスト1: WindowPreviewOutput]" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // サンプル映像のパス
    std::string videoPath = "C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\2025-10-08_13-47-52.mp4";
    
    // ソースと出力を作成
    auto source = std::make_shared<VirtualAd::FilePlaybackSource>(videoPath);
    auto preview = std::make_shared<VirtualAd::Core::WindowPreviewOutput>("VirtualAd Preview", 1);
    
    // パイプラインを作成
    VirtualAd::FramePipeline pipeline;
    pipeline.setSource(source);
    pipeline.addOutput(preview);
    
    // パイプライン起動
    if (!pipeline.start()) {
        std::cerr << "エラー: パイプラインを起動できませんでした" << std::endl;
        return;
    }
    
    std::cout << "✓ パイプライン起動成功" << std::endl;
    std::cout << "  解像度: " << source->getWidth() << "x" << source->getHeight() << std::endl;
    std::cout << "  フレームレート: " << source->getFrameRate() << " fps" << std::endl;
    std::cout << std::endl;
    
    std::cout << "100フレーム再生します（ESCキーで中断）..." << std::endl;
    
    // 100フレーム処理
    int maxFrames = 100;
    auto startTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < maxFrames; ++i) {
        if (!pipeline.processFrame()) {
            std::cerr << "警告: フレーム処理失敗（フレーム " << i << "）" << std::endl;
            break;
        }
        
        // 10フレームごとに進捗表示
        if ((i + 1) % 10 == 0) {
            std::cout << "  処理済み: " << (i + 1) << "/" << maxFrames 
                      << " フレーム (パイプラインFPS: " << pipeline.getCurrentFPS() << ")" << std::endl;
        }
        
        // ESCキーで中断
        if (preview->getLastKey() == 27) {
            std::cout << "  ユーザーによる中断" << std::endl;
            break;
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // 統計表示
    std::cout << std::endl;
    std::cout << "[統計]" << std::endl;
    std::cout << "  処理フレーム数: " << pipeline.getFrameCount() << std::endl;
    std::cout << "  表示フレーム数: " << preview->getDisplayedFrameCount() << std::endl;
    std::cout << "  処理時間: " << elapsed << " ms" << std::endl;
    std::cout << "  平均FPS: " << pipeline.getCurrentFPS() << std::endl;
    
    if (pipeline.getCurrentFPS() >= source->getFrameRate() * 0.95) {
        std::cout << "  ✓ 60fps目標達成！" << std::endl;
    }
    
    pipeline.stop();
    std::cout << "✓ テスト1完了" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief テスト2: FilePlaybackSource → FileWriterOutput
 * 映像ファイルに録画
 */
void testFileWriterOutput() {
    std::cout << "=============================================" << std::endl;
    std::cout << "[テスト2: FileWriterOutput]" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // サンプル映像のパス
    std::string videoPath = "C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\2025-10-08_13-47-52.mp4";
    std::string outputPath = "output_test.mp4";
    
    // ソースと出力を作成
    auto source = std::make_shared<VirtualAd::FilePlaybackSource>(videoPath);
    auto writer = std::make_shared<VirtualAd::Core::FileWriterOutput>(
        outputPath, 
        VirtualAd::Core::FileWriterOutput::Codec::H264
    );
    
    // パイプラインを作成
    VirtualAd::FramePipeline pipeline;
    pipeline.setSource(source);
    pipeline.addOutput(writer);
    
    // パイプライン起動
    if (!pipeline.start()) {
        std::cerr << "エラー: パイプラインを起動できませんでした" << std::endl;
        return;
    }
    
    std::cout << "✓ パイプライン起動成功" << std::endl;
    std::cout << "  入力: " << videoPath << std::endl;
    std::cout << "  出力: " << outputPath << std::endl;
    std::cout << "  解像度: " << source->getWidth() << "x" << source->getHeight() << std::endl;
    std::cout << "  フレームレート: " << source->getFrameRate() << " fps" << std::endl;
    std::cout << std::endl;
    
    std::cout << "50フレーム録画します..." << std::endl;
    
    // 50フレーム処理
    int maxFrames = 50;
    auto startTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < maxFrames; ++i) {
        if (!pipeline.processFrame()) {
            std::cerr << "警告: フレーム処理失敗（フレーム " << i << "）" << std::endl;
            break;
        }
        
        // 10フレームごとに進捗表示
        if ((i + 1) % 10 == 0) {
            std::cout << "  録画済み: " << (i + 1) << "/" << maxFrames << " フレーム" << std::endl;
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // 統計表示
    std::cout << std::endl;
    std::cout << "[統計]" << std::endl;
    std::cout << "  処理フレーム数: " << pipeline.getFrameCount() << std::endl;
    std::cout << "  録画フレーム数: " << writer->getWrittenFrameCount() << std::endl;
    std::cout << "  処理時間: " << elapsed << " ms" << std::endl;
    std::cout << "  平均FPS: " << pipeline.getCurrentFPS() << std::endl;
    
    pipeline.stop();
    std::cout << "✓ テスト2完了" << std::endl;
    std::cout << "  出力ファイル: " << outputPath << std::endl;
    std::cout << std::endl;
}

/**
 * @brief テスト3: 複数出力同時処理
 * FilePlaybackSource → WindowPreviewOutput + FileWriterOutput
 */
void testMultipleOutputs() {
    std::cout << "=============================================" << std::endl;
    std::cout << "[テスト3: 複数出力同時処理]" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // サンプル映像のパス
    std::string videoPath = "C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\2025-10-08_13-47-52.mp4";
    std::string outputPath = "output_multi_test.mp4";
    
    // ソースと出力を作成
    auto source = std::make_shared<VirtualAd::FilePlaybackSource>(videoPath);
    auto preview = std::make_shared<VirtualAd::Core::WindowPreviewOutput>("Multi Output Test", 1);
    auto writer = std::make_shared<VirtualAd::Core::FileWriterOutput>(
        outputPath, 
        VirtualAd::Core::FileWriterOutput::Codec::H264
    );
    
    // パイプラインを作成
    VirtualAd::FramePipeline pipeline;
    pipeline.setSource(source);
    pipeline.addOutput(preview);
    pipeline.addOutput(writer);
    
    // パイプライン起動
    if (!pipeline.start()) {
        std::cerr << "エラー: パイプラインを起動できませんでした" << std::endl;
        return;
    }
    
    std::cout << "✓ パイプライン起動成功（2つの出力）" << std::endl;
    std::cout << "  入力: " << videoPath << std::endl;
    std::cout << "  出力1: プレビューウィンドウ" << std::endl;
    std::cout << "  出力2: " << outputPath << std::endl;
    std::cout << std::endl;
    
    std::cout << "50フレーム処理します（ESCキーで中断）..." << std::endl;
    
    // 50フレーム処理
    int maxFrames = 50;
    auto startTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < maxFrames; ++i) {
        if (!pipeline.processFrame()) {
            std::cerr << "警告: フレーム処理失敗（フレーム " << i << "）" << std::endl;
            break;
        }
        
        // 10フレームごとに進捗表示
        if ((i + 1) % 10 == 0) {
            std::cout << "  処理済み: " << (i + 1) << "/" << maxFrames 
                      << " フレーム (FPS: " << pipeline.getCurrentFPS() << ")" << std::endl;
        }
        
        // ESCキーで中断
        if (preview->getLastKey() == 27) {
            std::cout << "  ユーザーによる中断" << std::endl;
            break;
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // 統計表示
    std::cout << std::endl;
    std::cout << "[統計]" << std::endl;
    std::cout << "  処理フレーム数: " << pipeline.getFrameCount() << std::endl;
    std::cout << "  表示フレーム数: " << preview->getDisplayedFrameCount() << std::endl;
    std::cout << "  録画フレーム数: " << writer->getWrittenFrameCount() << std::endl;
    std::cout << "  処理時間: " << elapsed << " ms" << std::endl;
    std::cout << "  平均FPS: " << pipeline.getCurrentFPS() << std::endl;
    
    pipeline.stop();
    std::cout << "✓ テスト3完了" << std::endl;
    std::cout << std::endl;
}

void printNextSteps() {
    std::cout << "=============================================" << std::endl;
    std::cout << "[次のステップ]" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "  ✓ Phase 1 Step 1: IVideoSource/IVideoOutput インターフェース実装完了" << std::endl;
    std::cout << "  ✓ Phase 1 Step 2: FilePlaybackSource 実装完了" << std::endl;
    std::cout << "  ✓ Phase 1 Step 3: WindowPreviewOutput, FileWriterOutput 実装完了" << std::endl;
    std::cout << "  - Phase 1 Step 4: Qt6 GUI実装" << std::endl;
    std::cout << "  - Phase 1 Step 5: DirectShowライブキャプチャ実装" << std::endl;
    std::cout << "  - Phase 2: カメラトラッキング" << std::endl;
    std::cout << "  - Phase 3: AIキーヤー" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    printBanner();
    printSystemInfo();
    
    // テスト1: プレビュー出力
    testPreviewOutput();
    
    // テスト2: ファイル録画
    testFileWriterOutput();
    
    // テスト3: 複数出力同時処理
    testMultipleOutputs();
    
    printNextSteps();

    std::cout << "=============================================" << std::endl;
    std::cout << "Phase 1 Step 3 完了: 映像出力実装" << std::endl;
    std::cout << "=============================================" << std::endl;

    return 0;
}
