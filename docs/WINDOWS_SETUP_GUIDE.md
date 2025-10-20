# Windows環境セットアップガイド

## 概要

このガイドは、MacからWindowsへプロジェクトを移行し、開発環境をセットアップする手順を説明します。

---

## 前提条件

- **Windows PC**: Windows 10/11 64-bit
- **GPU**: NVIDIA RTX 3060以上（CUDA対応必須）
- **RAM**: 16GB以上（推奨24GB）
- **Storage**: 256GB以上の空き容量（SSD推奨）
- **インターネット接続**: 依存関係のダウンロードに必要

---

## セットアップ手順

### Step 1: Gitリポジトリのクローン

```bash
# 適切なディレクトリに移動
cd C:\Users\YourName\Projects

# リポジトリをクローン
git clone <repository-url>
cd METABALL-VirtualAd

# ブランチ確認
git branch
```

---

### Step 2: Visual Studio 2022インストール

#### ダウンロード
https://visualstudio.microsoft.com/ja/downloads/

#### インストールするワークロード
1. **C++によるデスクトップ開発**
   - MSVC v143 - VS 2022 C++ x64/x86ビルドツール
   - Windows 10/11 SDK
   - CMake Tools for Windows
   - C++ ATL
   - Git for Windows

2. **オプション（推奨）**
   - C++ プロファイリングツール
   - IntelliCode

#### 確認
```bash
# Visual Studio起動確認
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe" --version
```

---

### Step 3: CUDA Toolkit 12.xインストール

#### ダウンロード
https://developer.nvidia.com/cuda-downloads

1. **CUDA Toolkit 12.6**（最新安定版）をダウンロード
2. インストーラーを実行（カスタムインストール推奨）
3. 以下をインストール:
   - CUDA Development
   - CUDA Runtime
   - CUDA Documentation（オプション）

#### cuDNN 8.xインストール
https://developer.nvidia.com/cudnn

1. NVIDIAアカウントでログイン
2. cuDNN 8.9 for CUDA 12.xをダウンロード
3. ZIPを解凍し、以下にコピー:
   ```
   bin/    → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\
   include/ → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\
   lib/    → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\
   ```

#### 環境変数確認
```bash
echo %CUDA_PATH%
# 出力例: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

nvcc --version
# CUDA 12.6が表示されればOK
```

---

### Step 4: TensorRT 8.6+インストール

#### ダウンロード
https://developer.nvidia.com/tensorrt

1. TensorRT 8.6 for Windows (CUDA 12.x)をダウンロード
2. ZIPを解凍: `C:\Program Files\TensorRT-8.6.x`

#### 環境変数設定
```bash
# システム環境変数に追加
setx PATH "%PATH%;C:\Program Files\TensorRT-8.6.x\lib"
setx TENSORRT_ROOT "C:\Program Files\TensorRT-8.6.x"
```

---

### Step 5: CMakeインストール

#### ダウンロード
https://cmake.org/download/

1. Windows x64 Installerをダウンロード
2. インストール時に「Add CMake to system PATH」を選択

#### 確認
```bash
cmake --version
# cmake version 3.28以上が表示されればOK
```

---

### Step 6: vcpkgセットアップ

#### インストール
```bash
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

#### 環境変数設定
```bash
setx VCPKG_ROOT "C:\vcpkg"
setx PATH "%PATH%;C:\vcpkg"
```

---

### Step 7: C++ライブラリインストール

プロジェクトディレクトリに戻ります:

```bash
cd C:\Users\YourName\Projects\METABALL-VirtualAd
```

#### OpenCV 4.8+ (CUDA対応)
```bash
vcpkg install opencv4[contrib,cuda]:x64-windows
```
⚠️ **注意**: ビルドに30-60分かかります

#### Qt 6.5+
```bash
vcpkg install qt6:x64-windows
```
⚠️ **注意**: ビルドに20-40分かかります

#### ONNX Runtime (GPU)
```bash
vcpkg install onnxruntime-gpu:x64-windows
```

#### FFmpeg
```bash
vcpkg install ffmpeg:x64-windows
```

---

### Step 8: Pythonセットアップ

#### Python 3.10+インストール
https://www.python.org/downloads/

1. Python 3.10以上をダウンロード
2. インストール時に「Add Python to PATH」を選択

#### 仮想環境作成
```bash
cd C:\Users\YourName\Projects\METABALL-VirtualAd
python -m venv venv
.\venv\Scripts\activate
```

#### PyTorch + CUDAインストール
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### その他のライブラリ
```bash
pip install -r python/requirements.txt
```

#### 確認
```python
python
>>> import torch
>>> torch.cuda.is_available()
True  # GPUが認識されていればTrue
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 4060 Laptop GPU'  # GPU名が表示される
>>> exit()
```

---

### Step 9: プロジェクトビルド

#### CMake設定
```bash
# プロジェクトディレクトリで実行
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

#### ビルド
```bash
cmake --build build --config Release
```

#### 実行
```bash
.\build\bin\Release\VirtualAd.exe
```

**期待される出力**:
```
=============================================
   METABALL Virtual Ad System v1.0.0
   バーチャル広告生成システム
=============================================

[システム情報]
  OS: Windows
  Compiler: MSVC 1939
  C++ Standard: 202002

...
環境構築が正常に完了しました！
=============================================
```

---

## トラブルシューティング

### 問題1: vcpkgでOpenCVビルドエラー

**症状**: `opencv4[cuda]`のビルドが失敗

**解決策**:
1. CUDA Toolkitが正しくインストールされているか確認
2. 環境変数`CUDA_PATH`が設定されているか確認
3. vcpkgを最新化: `git pull` → `.\bootstrap-vcpkg.bat`

### 問題2: CMakeがvcpkgを認識しない

**症状**: `find_package(OpenCV)`が失敗

**解決策**:
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```
ツールチェーンファイルを明示的に指定

### 問題3: PyTorchでCUDAが認識されない

**症状**: `torch.cuda.is_available()` が `False`

**解決策**:
1. NVIDIA Driverが最新か確認: `nvidia-smi`
2. PyTorchがCUDA版か確認:
   ```python
   import torch
   print(torch.version.cuda)  # 12.1などが表示されるべき
   ```
3. 再インストール:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 問題4: ビルド時にリンクエラー

**症状**: `LNK2019: 未解決の外部シンボル`

**解決策**:
1. vcpkgライブラリが正しくインストールされているか確認:
   ```bash
   vcpkg list
   ```
2. CMakeキャッシュをクリア:
   ```bash
   rm -r build
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
   ```

---

## 次のステップ

環境構築が完了したら、以下を確認してください:

1. ✅ `VirtualAd.exe`が正常に実行できる
2. ✅ `tasks/todo.md`でPhase 0の項目をチェック
3. ✅ ドキュメントを確認:
   - `docs/PROJECT_OVERVIEW.md`
   - `docs/REQUIREMENTS.md`
   - `docs/TECH_STACK.md`

**Phase 1開始準備完了！**

詳細な実装計画は `docs/IMPLEMENTATION_PLAN.md`（作成予定）を参照してください。

---

## 参考資料

- Visual Studio 2022: https://docs.microsoft.com/ja-jp/visualstudio/
- CUDA Toolkit: https://docs.nvidia.com/cuda/
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
- vcpkg: https://vcpkg.io/
- CMake: https://cmake.org/documentation/
- PyTorch: https://pytorch.org/docs/

---

**最終更新**: 2025/10/20
