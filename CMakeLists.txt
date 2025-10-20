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

# パッケージの検索（Phase 0完了後にコメント解除）
# find_package(OpenCV 4.8 REQUIRED)
# find_package(Qt6 COMPONENTS Widgets REQUIRED)
# find_package(ONNX REQUIRED)

# インクルードディレクトリ
include_directories(${CMAKE_SOURCE_DIR}/src)

# サブディレクトリ（Phase 1以降で追加）
# add_subdirectory(src/core)
# add_subdirectory(src/tracking)
# add_subdirectory(src/ai)
# add_subdirectory(src/rendering)
# add_subdirectory(src/gui)

# メイン実行ファイル（Phase 0では簡易版）
add_executable(VirtualAd
    src/main.cpp
)

# ライブラリリンク（Phase 1以降で追加）
# target_link_libraries(VirtualAd
#     PRIVATE
#         Core
#         Tracking
#         AI
#         Rendering
#         GUI
#         ${OpenCV_LIBS}
#         Qt6::Widgets
# )

# コンパイラ警告
if(MSVC)
    target_compile_options(VirtualAd PRIVATE /W4)
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
