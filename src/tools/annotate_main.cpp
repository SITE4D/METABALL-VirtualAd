// annotate_main.cpp
// アノテーションツールのメインエントリーポイント

#include "tools/AnnotationTool.h"
#include "core/FilePlaybackSource.h"
#include <iostream>
#include <string>

void printUsage() {
    std::cout << "Usage: AnnotationTool <video_path> <output_json>" << std::endl;
    std::cout << "  video_path: Path to video file" << std::endl;
    std::cout << "  output_json: Path to output JSON file" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  AnnotationTool input.mp4 annotations.json" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "   METABALL Virtual Ad - Annotation Tool" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;

    // コマンドライン引数チェック
    if (argc != 3) {
        printUsage();
        return 1;
    }

    std::string videoPath = argv[1];
    std::string outputPath = argv[2];

    std::cout << "Video: " << videoPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << std::endl;

    // 映像ソースを開く
    auto source = std::make_shared<VirtualAd::FilePlaybackSource>(videoPath);
    if (!source->start()) {
        std::cerr << "Error: Failed to open video file" << std::endl;
        return 1;
    }

    std::cout << "Video info:" << std::endl;
    std::cout << "  Resolution: " << source->getWidth() << "x" << source->getHeight() << std::endl;
    std::cout << "  FPS: " << source->getFrameRate() << std::endl;
    std::cout << "  Total frames: " << source->getTotalFrames() << std::endl;
    std::cout << std::endl;

    // アノテーションツールを作成
    VirtualAd::Tools::AnnotationTool tool("Backnet Annotation");

    std::vector<VirtualAd::Tools::BacknetCorners> annotations;

    std::cout << "Instructions:" << std::endl;
    std::cout << "  - Click 4 corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right" << std::endl;
    std::cout << "  - Press 'u' to undo, 'r' to reset" << std::endl;
    std::cout << "  - Press Enter when done" << std::endl;
    std::cout << "  - Press Esc to skip frame" << std::endl;
    std::cout << "  - Press 'q' to quit and save" << std::endl;
    std::cout << std::endl;

    // フレームごとにアノテーション
    int frameNumber = 0;
    int totalFrames = source->getTotalFrames();
    int annotatedCount = 0;

    // 10フレームごとにアノテーション（サンプリング）
    const int FRAME_STEP = 10;

    while (frameNumber < totalFrames) {
        // フレームを取得
        cv::Mat frame = source->getFrame();
        if (frame.empty()) {
            break;
        }

        // アノテーション実施
        auto result = tool.annotateFrame(frame, frameNumber);
        
        if (result.valid) {
            annotations.push_back(result);
            annotatedCount++;
        }

        std::cout << "Progress: " << annotatedCount << " frames annotated" << std::endl;

        // 次のフレームへ（10フレームスキップ）
        for (int i = 0; i < FRAME_STEP - 1; ++i) {
            cv::Mat skipFrame = source->getFrame();
            if (skipFrame.empty()) break;
        }
        frameNumber += FRAME_STEP;

        // 'q'キーでアノテーション終了
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            std::cout << "User requested quit" << std::endl;
            break;
        }
    }

    source->stop();

    // アノテーション結果を保存
    std::cout << std::endl;
    std::cout << "Saving annotations..." << std::endl;
    if (VirtualAd::Tools::AnnotationTool::saveAnnotations(outputPath, annotations)) {
        std::cout << "Success! Annotated " << annotatedCount << " frames" << std::endl;
    } else {
        std::cerr << "Error: Failed to save annotations" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Annotation completed" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
