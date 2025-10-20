# 技術スタック

## 概要

METABALL Virtual Adは、C++をコア言語とし、AI/ML部分でPythonを活用するハイブリッド構成です。Windows専用に最適化され、NVIDIA GPUの性能を最大限に引き出します。

---

## 開発環境

### IDE・ビルドツール
- **Visual Studio 2022** (Community以上)
  - C++20サポート
  - CMake統合
  - デバッガー
  - プロファイラー

- **CMake 3.20+**
  - クロスプラットフォームビルドシステム
  - 依存関係管理
  - テストフレームワーク統合

- **Git**
  - バージョン管理
  - GitHub連携

### Python環境
- **Python 3.10+**
  - PyTorch 2.0+
  - NumPy, OpenCV-Python
  - ONNX

- **仮想環境**
  - venv または Anaconda
  - requirements.txt管理

---

## コア技術スタック

### プログラミング言語

#### C++20
**用途**: コア処理、リアルタイム映像処理、GPU制御

**選定理由**:
- 高速な実行速度
- メモリ管理の柔軟性
- DirectX/CUDA直接制御
- 成熟したライブラリエコシステム

**主要機能**:
- 映像入出力
- カメラトラッキング
- レンダリング
- GPU計算
- UI

#### Python 3.10+
**用途**: AI/ML学習、データ処理、スクリプト

**選定理由**:
- PyTorch/TensorFlowとの親和性
- 豊富なCV/MLライブラリ
- 学習パイプラインの迅速な開発

**主要機能**:
- AIモデル訓練
- データ前処理
- ONNX変換
- 学習スクリプト

---

## 映像処理

### OpenCV 4.8+
**役割**: 画像処理、カメラキャリブレーション、特徴点検出

**主要機能**:
- VideoCapture (入力抽象化)
- 特徴点検出 (ORB, AKAZE)
- ホモグラフィ推定
- PnPソルバー
- 画像変換

**インストール**:
```bash
# vcpkgを使用
vcpkg install opencv4[contrib,cuda]:x64-windows
```

**使用例**:
```cpp
#include <opencv2/opencv.hpp>

cv::VideoCapture capture(0);  // カメラ0
cv::Mat frame;
capture >> frame;

// 特徴点検出
cv::Ptr<cv::ORB> orb = cv::ORB::create();
std::vector<cv::KeyPoint> keypoints;
orb->detect(frame, keypoints);
```

### FFmpeg 6.0+
**役割**: 動画ファイル入出力、エンコーディング

**主要機能**:
- MP4/MOV/AVI読み込み
- H.264エンコーディング
- リアルタイムストリーミング

**インストール**:
```bash
vcpkg install ffmpeg:x64-windows
```

---

## GPU処理・レンダリング

### DirectX 12
**役割**: Windows専用GPUレンダリング

**主要機能**:
- 高速レンダリング
- コンピュートシェーダー
- 透視変換
- テクスチャマッピング

**SDK**:
- Windows SDK (Visual Studio同梱)
- DirectX 12 Agility SDK

**使用例**:
```cpp
#include <d3d12.h>
#include <dxgi1_6.h>

// デバイス作成
ID3D12Device* device;
D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0,
                  IID_PPV_ARGS(&device));
```

### CUDA 12.x
**役割**: NVIDIA GPU汎用計算

**主要機能**:
- 並列処理
- AI推論高速化
- DirectX連携

**インストール**:
- CUDA Toolkit 12.x (https://developer.nvidia.com/cuda-downloads)
- cuDNN 8.x

**要件**:
- NVIDIA GPU (Compute Capability 7.5+)
- RTX 3060以上推奨

**使用例**:
```cuda
__global__ void compositeKernel(
    uchar3* output,
    const uchar3* input,
    const uchar* mask,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        if (mask[idx] > 128) {
            output[idx] = input[idx];
        }
    }
}
```

---

## AI/ML推論

### ONNX Runtime 1.16+
**役割**: クロスプラットフォームAI推論

**主要機能**:
- PyTorchモデルの実行
- GPU推論
- モデル最適化

**インストール**:
```bash
vcpkg install onnxruntime-gpu:x64-windows
```

**使用例**:
```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VirtualAd");
Ort::SessionOptions session_options;
session_options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_ALL
);

Ort::Session session(env, L"model.onnx", session_options);
```

### TensorRT 8.6+
**役割**: NVIDIA GPU専用の超高速推論

**主要機能**:
- INT8量子化
- 推論速度3-5倍高速化
- CUDA統合

**インストール**:
- TensorRT 8.6 (https://developer.nvidia.com/tensorrt)

**最適化フロー**:
```
PyTorch → ONNX → TensorRT Engine
```

**使用例**:
```cpp
#include <NvInfer.h>

nvinfer1::IBuilder* builder =
    nvinfer1::createInferBuilder(logger);
nvinfer1::INetworkDefinition* network =
    builder->createNetworkV2(0U);

// ONNXパーサーでモデル読み込み
auto parser = nvonnxparser::createParser(*network, logger);
parser->parseFromFile("model.onnx",
                      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

// INT8量子化設定
nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
config->setFlag(nvinfer1::BuilderFlag::kINT8);

// エンジン構築
nvinfer1::ICudaEngine* engine =
    builder->buildEngineWithConfig(*network, *config);
```

---

## AI/ML学習（Python）

### PyTorch 2.0+
**役割**: AIモデル訓練

**主要機能**:
- カメラトラッキング補正モデル
- セグメンテーションモデル
- デプス推定モデル

**インストール**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**学習例**:
```python
import torch
import torch.nn as nn

class CameraParamPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            'pytorch/vision', 'resnet18', pretrained=True
        )
        self.backbone.fc = nn.Linear(512, 6)  # rvec[3] + tvec[3]

    def forward(self, x):
        return self.backbone(x)

model = CameraParamPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### 主要モデル

#### 1. セグメンテーション: DeepLabV3+
```python
model = torch.hub.load(
    'pytorch/vision',
    'deeplabv3_resnet50',
    pretrained=True
)
```

#### 2. デプス推定: MiDaS Small
```python
model = torch.hub.load(
    'intel-isl/MiDaS',
    'MiDaS_small'
)
```

---

## GUI

### Qt 6.5+
**役割**: Windows GUI開発

**主要機能**:
- メインウィンドウ
- ビデオプレビュー
- 設定パネル
- アノテーションツール

**インストール**:
```bash
vcpkg install qt6:x64-windows
```

**使用例**:
```cpp
#include <QApplication>
#include <QMainWindow>
#include <QLabel>

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    QMainWindow window;
    QLabel* label = new QLabel("Virtual Ad", &window);
    window.setCentralWidget(label);
    window.show();

    return app.exec();
}
```

---

## 依存関係管理

### vcpkg
**役割**: C++パッケージマネージャー

**インストール**:
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

**パッケージインストール**:
```bash
vcpkg install opencv4[contrib,cuda]:x64-windows
vcpkg install qt6:x64-windows
vcpkg install onnxruntime-gpu:x64-windows
vcpkg install ffmpeg:x64-windows
```

### pip (Python)
**requirements.txt**:
```txt
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
```

**インストール**:
```bash
pip install -r python/requirements.txt
```

---

## ビルドシステム

### CMake構成

**CMakeLists.txt（ルート）**:
```cmake
cmake_minimum_required(VERSION 3.20)
project(VirtualAd VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# パッケージ検索
find_package(OpenCV 4.8 REQUIRED)
find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(CUDA 12.0 REQUIRED)

# サブディレクトリ
add_subdirectory(src/core)
add_subdirectory(src/tracking)
add_subdirectory(src/ai)
add_subdirectory(src/rendering)
add_subdirectory(src/gui)

# 実行ファイル
add_executable(VirtualAd src/main.cpp)
target_link_libraries(VirtualAd
    PRIVATE
        Core
        Tracking
        AI
        Rendering
        GUI
        ${OpenCV_LIBS}
        Qt6::Widgets
)
```

---

## 開発ツール

### デバッグ・プロファイリング

#### Visual Studio Debugger
- ブレークポイント
- ウォッチウィンドウ
- メモリダンプ

#### NVIDIA Nsight Graphics
- GPU プロファイリング
- DirectX/CUDA デバッグ
- フレーム解析

**インストール**:
```
https://developer.nvidia.com/nsight-graphics
```

#### RenderDoc
- DirectXキャプチャ
- シェーダーデバッグ

**インストール**:
```
https://renderdoc.org/
```

### パフォーマンス測定

#### Windows Performance Monitor
- CPU使用率
- メモリ使用量

#### nvidia-smi
- GPU使用率
- VRAM使用量
- 温度監視

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

---

## バージョン管理戦略

### Git構成

**.gitignore**:
```gitignore
# ビルド成果物
build/
bin/
lib/
*.exe
*.dll
*.lib
*.exp

# Visual Studio
.vs/
*.user
*.suo

# CMake
CMakeCache.txt
CMakeFiles/

# Python
__pycache__/
*.pyc
.venv/
venv/

# データ
data/samples/*.png
data/samples/*.json
models/*.onnx
logs/

# 一時ファイル
*.tmp
*.log
```

### ブランチ戦略

```
main          - 安定版
develop       - 開発版
feature/*     - 機能開発
hotfix/*      - 緊急修正
```

---

## 継続的インテグレーション（将来）

### GitHub Actions（想定）

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: windows-latest

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        vcpkg install opencv4:x64-windows
        vcpkg install qt6:x64-windows

    - name: Configure CMake
      run: cmake -B build -DCMAKE_BUILD_TYPE=Release

    - name: Build
      run: cmake --build build --config Release

    - name: Test
      run: ctest --test-dir build -C Release
```

---

## セキュリティ

### 依存関係の脆弱性チェック

- **C++**: `vcpkg x-update-baseline`
- **Python**: `pip-audit`

### コード品質

- **静的解析**: Visual Studio Code Analysis
- **フォーマット**: clang-format

---

## ドキュメント生成

### Doxygen
C++コードドキュメント自動生成

**インストール**:
```
https://www.doxygen.nl/download.html
```

**設定**:
```bash
doxygen -g  # Doxyfile生成
doxygen     # ドキュメント生成
```

---

## まとめ

### コア技術
| カテゴリ | 技術 | バージョン |
|---------|------|----------|
| 言語 | C++ | 20 |
| 言語 | Python | 3.10+ |
| ビルド | CMake | 3.20+ |
| 映像 | OpenCV | 4.8+ |
| GPU | DirectX 12 | - |
| GPU | CUDA | 12.x |
| AI推論 | ONNX Runtime | 1.16+ |
| AI推論 | TensorRT | 8.6+ |
| AI学習 | PyTorch | 2.0+ |
| GUI | Qt | 6.5+ |

### インストール順序

1. Visual Studio 2022
2. CUDA Toolkit 12.x
3. TensorRT 8.6+
4. vcpkg
5. CMake
6. Python 3.10+
7. vcpkg経由でライブラリインストール
8. pip経由でPythonライブラリインストール

詳細な環境構築手順は [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) を参照してください。
