#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "core/FilePlaybackSource.h"
#include "core/FramePipeline.h"

/**
 * @file main.cpp
 * @brief METABALL Virtual Ad - メインエントリーポイント
 *
 * Phase 1: 映像I/Oパイプライン実装
 * FilePlaybackSourceの動作確認
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

void testFilePlayback() {
    std::cout << "[FilePlaybackSource テスト]" << std::endl;
    
    // サンプル映像のパス
    std::string videoPath = "C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\2025-10-08_13-47-52.mp4";
    
    std::cout << "映像ファイル: " << videoPath << std::endl;
    std::cout << std::endl;
    
    // FilePlaybackSourceを作成
    auto source = std::make_shared<VirtualAd::FilePlaybackSource>(videoPath);
    
    // 映像を開く
    if (!source->start()) {
        std::cerr << "エラー: 映像ファイルを開けませんでした" << std::endl;
        return;
    }
    
    std::cout << "✓ 映像ソース起動成功" << std::endl;
    std::cout << std::endl;
    
    // 映像情報を表示
    std::cout << "[映像情報]" << std::endl;
    std::cout << "  解像度: " << source->getWidth() << "x" << source->getHeight() << std::endl;
    std::cout << "  フレームレート: " << source->getFrameRate() << " fps" << std::endl;
    std::cout << "  総フレーム数: " << source->getTotalFrames() << std::endl;
    std::cout << std::endl;
    
    // フレーム取得テスト（100フレーム再生）
    std::cout << "[フレーム取得テスト - 100フレーム再生]" << std::endl;
    
    int frameCount = 0;
    int maxFrames = 100;
    
    auto startTime = std::chrono::steady_clock::now();
    
    while (frameCount < maxFrames) {
        cv::Mat frame = source->getFrame();
        
        if (frame.empty()) {
            std::cerr << "警告: 空のフレーム取得" << std::endl;
            break;
        }
        
        frameCount++;
        
        // 10フレームごとに進捗表示
        if (frameCount % 10 == 0) {
            std::cout << "  フレーム " << frameCount << "/" << maxFrames 
                      << " (現在位置: " << source->getCurrentFrameNumber() << ")" << std::endl;
        }
        
        // ESCキーまたは 'q' キーでプレビューを閉じる（OpenCV GUIがある場合）
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            std::cout << "  ユーザーによる中断" << std::endl;
            break;
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << std::endl;
    std::cout << "[パフォーマンス]" << std::endl;
    std::cout << "  処理フレーム数: " << frameCount << std::endl;
    std::cout << "  処理時間: " << elapsed << " ms" << std::endl;
    
    if (elapsed > 0) {
        double actualFPS = (frameCount * 1000.0) / elapsed;
        std::cout << "  実フレームレート: " << actualFPS << " fps" << std::endl;
        
        double targetFPS = source->getFrameRate();
        std::cout << "  目標フレームレート: " << targetFPS << " fps" << std::endl;
        
        if (actualFPS >= targetFPS * 0.95) {
            std::cout << "  ✓ フレームレート目標達成！" << std::endl;
        } else {
            std::cout << "  ⚠ フレームレートが目標未達" << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // シーク機能テスト
    std::cout << "[シーク機能テスト]" << std::endl;
    
    int seekFrame = source->getTotalFrames() / 2;
    std::cout << "  フレーム " << seekFrame << " にシーク..." << std::endl;
    
    if (source->seekToFrame(seekFrame)) {
        std::cout << "  ✓ シーク成功" << std::endl;
        std::cout << "  現在位置: " << source->getCurrentFrameNumber() << std::endl;
        
        // シーク後のフレーム取得確認
        cv::Mat frame = source->getFrame();
        if (!frame.empty()) {
            std::cout << "  ✓ シーク後のフレーム取得成功" << std::endl;
        }
    } else {
        std::cout << "  ❌ シーク失敗" << std::endl;
    }
    
    std::cout << std::endl;
    
    // 停止
    source->stop();
    std::cout << "✓ 映像ソース停止" << std::endl;
    std::cout << std::endl;
}

void printNextSteps() {
    std::cout << "[次のステップ]" << std::endl;
    std::cout << "  ✓ Phase 1 Step 1: IVideoSource/IVideoOutput インターフェース実装完了" << std::endl;
    std::cout << "  ✓ Phase 1 Step 2: FilePlaybackSource 実装完了" << std::endl;
    std::cout << "  - Phase 1 Step 3: 簡易GUI実装 (Qt)" << std::endl;
    std::cout << "  - Phase 1 Step 4: ライブキャプチャ実装 (DirectShow)" << std::endl;
    std::cout << "  - Phase 2: カメラトラッキング" << std::endl;
    std::cout << "  - Phase 3: AIキーヤー" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    printBanner();
    printSystemInfo();
    testFilePlayback();
    printNextSteps();

    std::cout << "=============================================" << std::endl;
    std::cout << "Phase 1 Step 2 完了：FilePlaybackSource実装" << std::endl;
    std::cout << "=============================================" << std::endl;

    return 0;
}
