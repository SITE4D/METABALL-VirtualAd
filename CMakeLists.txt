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
# find_package(Qt6 COMPONENTS Widgets REQUIRED)  # Phase 1以降で有効化
# find_package(ONNX REQUIRED)  # Phase 2以降で有効化

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# インクルードディレクトリ
include_directories(${CMAKE_SOURCE_DIR}/src)

# コアライブラリ（Phase 1）
add_library(Core STATIC
    src/core/FramePipeline.cpp
    src/core/FilePlaybackSource.cpp
    src/core/WindowPreviewOutput.cpp
    src/core/FileWriterOutput.cpp
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

# サブディレクトリ（Phase 2以降で追加）
# add_subdirectory(src/tracking)
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

# インストールターゲット
install(TARGETS VirtualAd
    RUNTIME DESTINATION bin
)

# メッセージ
message(STATUS "==============================================")
message(STATUS "METABALL Virtual Ad - CMake Configuration")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==============================================")
