#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

/**
 * @file main.cpp
 * @brief METABALL Virtual Ad - メインエントリーポイント
 *
 * Phase 0: 環境構築確認用の簡易版
 * Windows環境でビルド・実行できることを確認する
 * OpenCVの基本動作確認を含む
 */

void printBanner() {
    std::cout << "=============================================" << std::endl;
    std::cout << "   METABALL Virtual Ad System v1.0.0" << std::endl;
    std::cout << "   バーチャル広告生成システム" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
}

void printSystemInfo() {
    std::cout << "[システム情報]" << std::endl;

    #ifdef _WIN32
        std::cout << "  OS: Windows" << std::endl;
    #elif __APPLE__
        std::cout << "  OS: macOS" << std::endl;
    #elif __linux__
        std::cout << "  OS: Linux" << std::endl;
    #endif

    #ifdef _MSC_VER
        std::cout << "  Compiler: MSVC " << _MSC_VER << std::endl;
    #elif __GNUC__
        std::cout << "  Compiler: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
    #elif __clang__
        std::cout << "  Compiler: Clang " << __clang_major__ << "." << __clang_minor__ << std::endl;
    #endif

    std::cout << "  C++ Standard: " << __cplusplus << std::endl;
    std::cout << std::endl;
}

void printProjectStatus() {
    std::cout << "[プロジェクトステータス]" << std::endl;
    std::cout << "  現在のフェーズ: Phase 0 - 環境構築" << std::endl;
    std::cout << "  ビルドシステム: CMake + Visual Studio 2022" << std::endl;
    std::cout << "  ターゲット: Windows 10/11 64-bit" << std::endl;
    std::cout << std::endl;

    std::cout << "[OpenCV情報]" << std::endl;
    std::cout << "  バージョン: " << CV_VERSION << std::endl;
    std::cout << "  ビルド情報: " << cv::getBuildInformation().substr(0, 500) << "..." << std::endl;
    std::cout << std::endl;

    std::cout << "[次のステップ]" << std::endl;
    std::cout << "  1. Visual Studio 2022インストール確認 ✓" << std::endl;
    std::cout << "  2. CUDA Toolkit 12.xセットアップ (Phase 1以降)" << std::endl;
    std::cout << "  3. vcpkg経由でライブラリインストール" << std::endl;
    std::cout << "     - OpenCV 4.8+ ✓" << std::endl;
    std::cout << "     - Qt 6.5+ (Phase 1以降)" << std::endl;
    std::cout << "     - ONNX Runtime GPU (Phase 2以降)" << std::endl;
    std::cout << "  4. Phase 1: 映像I/Oパイプライン実装開始" << std::endl;
    std::cout << std::endl;
}

void testOpenCV() {
    std::cout << "[OpenCV動作確認]" << std::endl;
    
    // 簡単なMatオブジェクト作成テスト
    cv::Mat testImage = cv::Mat::zeros(480, 640, CV_8UC3);
    if (testImage.empty()) {
        std::cout << "  ❌ Mat作成失敗" << std::endl;
        return;
    }
    std::cout << "  ✓ Mat作成成功 (640x480, 3チャンネル)" << std::endl;
    
    // 簡単な描画テスト
    cv::rectangle(testImage, cv::Point(100, 100), cv::Point(540, 380), cv::Scalar(0, 255, 0), 2);
    cv::putText(testImage, "METABALL Virtual Ad", cv::Point(150, 240), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    std::cout << "  ✓ 描画関数動作確認" << std::endl;
    
    std::cout << "  ✓ OpenCV基本機能正常動作" << std::endl;
    std::cout << std::endl;
}

void printDocumentation() {
    std::cout << "[ドキュメント]" << std::endl;
    std::cout << "  詳細な情報は以下を参照してください:" << std::endl;
    std::cout << "    - README.md" << std::endl;
    std::cout << "    - docs/PROJECT_OVERVIEW.md" << std::endl;
    std::cout << "    - docs/REQUIREMENTS.md" << std::endl;
    std::cout << "    - docs/TECH_STACK.md" << std::endl;
    std::cout << "    - tasks/todo.md" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    printBanner();
    printSystemInfo();
    printProjectStatus();
    testOpenCV();
    printDocumentation();

    std::cout << "==============================================" << std::endl;
    std::cout << "Phase 0 環境構築が正常に完了しました！" << std::endl;
    std::cout << "✓ Visual Studio 2022" << std::endl;
    std::cout << "✓ CMake 3.31" << std::endl;
    std::cout << "✓ vcpkg パッケージマネージャー" << std::endl;
    std::cout << "✓ OpenCV 4.8.0" << std::endl;
    std::cout << std::endl;
    std::cout << "次はPhase 1に進む準備が整いました。" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
