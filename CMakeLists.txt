cmake_minimum_required(VERSION 3.20)
project(VirtualAd VERSION 1.0.0 LANGUAGES CXX)

# C++20を使用
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ビルドタイプのデフォルト設定
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 出力ディレクトリの設定
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# vcpkg統合の確認
if(DEFINED ENV{VCPKG_ROOT})
    message(STATUS "vcpkg root: $ENV{VCPKG_ROOT}")
endif()

# vcpkgマニフェストモード用の設定
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows")
    message(STATUS "Using vcpkg installed packages from: ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows")
endif()

# パッケージの検索

find_package(OpenCV 4.8 REQUIRED)
find_package(onnxruntime QUIET)  # Phase 3で有効化、見つからない場合は警告のみ
# find_package(Qt6 COMPONENTS Widgets REQUIRED)  # Phase 1以降で有効化

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

if(onnxruntime_FOUND)
    message(STATUS "ONNX Runtime found")
else()
    message(WARNING "ONNX Runtime not found - AI inference features will be disabled")
endif()

# インクルードディレクトリ
include_directories(${CMAKE_SOURCE_DIR}/src)

# コアライブラリ（Phase 1）
add_library(Core STATIC
    src/core/FramePipeline.cpp
    src/core/FilePlaybackSource.cpp
    src/core/WindowPreviewOutput.cpp
    src/core/FileWriterOutput.cpp
    src/core/DirectShowCaptureSource.cpp
)

target_include_directories(Core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Core
    PUBLIC
        ${OpenCV_LIBS}
)

# Toolsライブラリ（Phase 1.5）
add_library(Tools STATIC
    src/tools/AnnotationTool.cpp
)

target_include_directories(Tools
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Tools
    PUBLIC
        ${OpenCV_LIBS}
)

# Trackingライブラリ（Phase 2, Phase 2.5）
add_library(Tracking STATIC
    src/tracking/FeatureDetector.cpp
    src/tracking/FeatureMatcher.cpp
    src/tracking/PnPSolver.cpp
    src/tracking/FeatureTracker.cpp
)

target_include_directories(Tracking
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Tracking
    PUBLIC
        ${OpenCV_LIBS}
)

# Inferenceライブラリ（Phase 3）
if(onnxruntime_FOUND)
    add_library(Inference STATIC
        src/inference/ONNXInference.cpp
        src/inference/CameraPoseRefiner.cpp
    )

    target_include_directories(Inference
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(Inference
        PUBLIC
            ${OpenCV_LIBS}
            onnxruntime::onnxruntime
            Tracking
    )
endif()

# Keyerライブラリ（Phase 4）
if(onnxruntime_FOUND)
    add_library(Keyer STATIC
        src/keyer/SegmentationInference.cpp
        src/keyer/DepthEstimator.cpp
        src/keyer/DepthCompositor.cpp
    )

    target_include_directories(Keyer
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(Keyer
        PUBLIC
            ${OpenCV_LIBS}
            onnxruntime::onnxruntime
    )
else()
    # onnxruntime not found - create Keyer with only DepthCompositor
    add_library(Keyer STATIC
        src/keyer/DepthCompositor.cpp
    )

    target_include_directories(Keyer
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(Keyer
        PUBLIC
            ${OpenCV_LIBS}
    )
endif()

# Renderingライブラリ（Phase 5）
add_library(Rendering STATIC
    src/rendering/AdRenderer.cpp
)

target_include_directories(Rendering
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Rendering
    PUBLIC
        ${OpenCV_LIBS}
)

# Integrationライブラリ（Phase 6）
add_library(Integration STATIC
    src/integration/IntegratedPipeline.cpp
)

target_include_directories(Integration
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

if(onnxruntime_FOUND)
    target_link_libraries(Integration
        PUBLIC
            ${OpenCV_LIBS}
            Tracking
            Inference
            Keyer
            Rendering
    )
else()
    target_link_libraries(Integration
        PUBLIC
            ${OpenCV_LIBS}
            Tracking
            Keyer
            Rendering
    )
endif()

# サブディレクトリ（Phase 2以降で追加）
# add_subdirectory(src/ai)
# add_subdirectory(src/rendering)
# add_subdirectory(src/gui)

# メイン実行ファイル
add_executable(VirtualAd
    src/main.cpp
)

# ライブラリリンク
target_link_libraries(VirtualAd
    PRIVATE
        Core
        ${OpenCV_LIBS}
)
# Phase 2以降で追加:
#     Tracking
#     AI
#     Rendering
#     GUI
#     Qt6::Widgets

# コンパイラ警告
if(MSVC)
    target_compile_options(VirtualAd PRIVATE /W4 /utf-8)
else()
    target_compile_options(VirtualAd PRIVATE -Wall -Wextra -Wpedantic)
endif()

# アノテーションツール実行ファイル（Phase 1.5）
add_executable(AnnotationTool
    src/tools/annotate_main.cpp
)

target_link_libraries(AnnotationTool
    PRIVATE
        Core
        Tools
        ${OpenCV_LIBS}
)

# 特徴点検出テストプログラム（Phase 2）
add_executable(TestFeatureDetection
    src/tracking/test_feature_detection.cpp
)

target_link_libraries(TestFeatureDetection
    PRIVATE
        Core
        Tracking
        ${OpenCV_LIBS}
)

# 特徴点マッチングテストプログラム（Phase 2）
add_executable(TestFeatureMatching
    src/tracking/test_feature_matching.cpp
)

target_link_libraries(TestFeatureMatching
    PRIVATE
        Core
        Tracking
        ${OpenCV_LIBS}
)

# 特徴トラッキングテストプログラム（Phase 2.5）
add_executable(TestFeatureTracking
    src/tracking/test_feature_tracking.cpp
)

target_link_libraries(TestFeatureTracking
    PRIVATE
        Core
        Tracking
        ${OpenCV_LIBS}
)

# ONNX推論テストプログラム（Phase 3）
if(onnxruntime_FOUND)
    add_executable(TestONNXInference
        src/inference/test_onnx_inference.cpp
    )

    target_link_libraries(TestONNXInference
        PRIVATE
            Inference
            ${OpenCV_LIBS}
            onnxruntime::onnxruntime
    )
endif()

# ポーズ補正テストプログラム（Phase 3）
if(onnxruntime_FOUND)
    add_executable(TestPoseRefinement
        src/inference/test_pose_refinement.cpp
    )

    target_link_libraries(TestPoseRefinement
        PRIVATE
            Core
            Inference
            ${OpenCV_LIBS}
            onnxruntime::onnxruntime
    )
endif()

# AIトラッキングデモ（Phase 3）
if(onnxruntime_FOUND)
    add_executable(DemoTrackingAI
        src/inference/demo_tracking_ai.cpp
    )

    target_link_libraries(DemoTrackingAI
        PRIVATE
            Core
            Tracking
            Inference
            ${OpenCV_LIBS}
            onnxruntime::onnxruntime
    )
endif()

# セグメンテーションテストプログラム（Phase 4）
if(onnxruntime_FOUND)
    add_executable(TestSegmentation
        src/keyer/test_segmentation.cpp
    )

    target_link_libraries(TestSegmentation
        PRIVATE
            Keyer
            ${OpenCV_LIBS}
            onnxruntime::onnxruntime
    )
endif()

# デプス合成テストプログラム（Phase 4）
add_executable(TestDepthCompositor
    src/keyer/test_depth_compositor.cpp
)

target_link_libraries(TestDepthCompositor
    PRIVATE
        Keyer
        ${OpenCV_LIBS}
)

# 広告レンダリングテストプログラム（Phase 5）
add_executable(TestAdRenderer
    src/rendering/test_ad_renderer.cpp
)

target_link_libraries(TestAdRenderer
    PRIVATE
        Rendering
        ${OpenCV_LIBS}
)

# 統合パイプラインテストプログラム（Phase 6）
add_executable(TestIntegratedPipeline
    src/integration/test_integrated_pipeline.cpp
)

target_link_libraries(TestIntegratedPipeline
    PRIVATE
        Integration
        ${OpenCV_LIBS}
)

# インストールターゲット
install(TARGETS 
    VirtualAd 
    AnnotationTool 
    TestFeatureDetection 
    TestFeatureMatching 
    TestFeatureTracking
    TestDepthCompositor
    TestAdRenderer
    TestIntegratedPipeline
    RUNTIME DESTINATION bin
)

# onnxruntime依存のターゲット（条件付き）
if(onnxruntime_FOUND)
    install(TARGETS
        TestONNXInference
        TestPoseRefinement
        DemoTrackingAI
        TestSegmentation
        RUNTIME DESTINATION bin
    )
endif()

# メッセージ
message(STATUS "==============================================")
message(STATUS "METABALL Virtual Ad - CMake Configuration")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==============================================")
