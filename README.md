# METABALL Virtual Ad

野球中継のバックネット裏にリアルタイムでバーチャル広告を挿入するシステム

## 概要

Viz Arenaのようなプロフェッショナルなバーチャル広告生成ソフトウェア。AIを活用した高精度なカメラトラッキングとキーイング技術により、リアルタイム60fpsでバーチャル広告をシームレスに合成します。

## 主要機能

- ✅ **AIアシストキャリブレーション**: カメラ映像のみで高精度トラッキング
- ✅ **AIキーヤー**: デプスベースのコンポジット（選手が広告の前に表示）
- ✅ **ライブ/ファイルモード**: リアルタイム処理と検証用ファイル再生
- ✅ **データ収集ツール**: AI学習用のアノテーションツール

## 性能目標

- **60fps リアルタイム処理**
- **低レイテンシー** (100ms以内)
- **高精度トラッキング** (画角変化・カメラ移動対応)

## システム要件

### ハードウェア
- **OS**: Windows 10/11 64-bit
- **GPU**: NVIDIA RTX 3060以上（CUDA対応必須）
- **RAM**: 16GB以上（推奨: 24GB）
- **Storage**: SSD 100GB以上
- **キャプチャ**: USB HDMIキャプチャデバイス（Elgato HD60 S+等）

### ソフトウェア（Windows専用）
- Visual Studio 2022
- CUDA Toolkit 12.x
- TensorRT 8.6+
- CMake 3.20+
- Python 3.10+

## クイックスタート（Windows）

```bash
# 1. リポジトリクローン
git clone https://github.com/SITE4D/METABALL-VirtualAd.git
cd METABALL-VirtualAd

# 2. 依存関係インストール（vcpkg）
vcpkg install

# 3. ビルド
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# 4. AI推論デモ実行
build\Release\demo_tracking_ai.exe [video_path] [model_path] [mode] [blend_alpha]

# 例: PnPのみモードでデモ実行
build\Release\demo_tracking_ai.exe data/samples/test_video.mp4 models/camera_pose_net.onnx PNP_ONLY
```

## デモプログラム

### AI Tracking Demo (`demo_tracking_ai`)

AI-powered カメラポーズ推定のデモアプリケーション

**機能:**
- ビデオファイル再生
- リアルタイムカメラポーズ推定（PnP + AI統合）
- 3D座標軸の可視化
- 3つの動作モード切り替え
- パフォーマンス測定

**使用方法:**

```bash
# 基本的な使い方
demo_tracking_ai.exe [video_path] [model_path] [mode] [blend_alpha]

# パラメータ:
#   video_path   : 入力ビデオファイルパス（デフォルト: data/samples/test_video.mp4）
#   model_path   : ONNXモデルファイルパス（デフォルト: models/camera_pose_net.onnx）
#   mode         : PNP_ONLY | AI_ONLY | BLENDED（デフォルト: BLENDED）
#   blend_alpha  : ブレンディング係数 0.0-1.0（デフォルト: 0.5）

# 例1: PnPのみ
demo_tracking_ai.exe sample.mp4 model.onnx PNP_ONLY

# 例2: AIとPnPのブレンディング（75% AI、25% PnP）
demo_tracking_ai.exe sample.mp4 model.onnx BLENDED 0.75
```

**キーボード操作:**
- `1`: PNP_ONLY モードに切り替え
- `2`: AI_ONLY モードに切り替え
- `3`: BLENDED モードに切り替え
- `Q` / `ESC`: 終了

**出力:**
- リアルタイム映像表示（3D軸、検出コーナー、統計情報）
- コンソールに詳細な統計情報
- 処理時間測定（目標: ≤10ms/frame）
```

## 開発進捗

### Phase 1: C++コアシステム ✅
- ビデオI/O（ファイル再生/キャプチャ）
- フレームパイプライン
- 特徴検出・マッチング（ORB/AKAZE）
- PnPソルバー（OpenCV solvePnP）

### Phase 2: AI学習パイプライン ✅
- PyTorchモデル実装（CameraPoseNet）
- 学習・評価スクリプト
- ONNX エクスポート

### Phase 2.5: トラッキング基盤（C++）✅
- 特徴点検出（ORB/AKAZE）
- 特徴マッチング（Lowe's ratio test + RANSAC）
- FeatureTracker統合クラス
- プラナートラッキング実装

### Phase 3: C++統合とデモ ✅
- ONNX Runtime C++ ラッパー実装
- PnP + AI統合（CameraPoseRefiner）
- デモアプリケーション（`demo_tracking_ai`）
- 3つの動作モード（PNP_ONLY, AI_ONLY, BLENDED）

**Phase 2.5 完了日**: 2025/10/20
**Phase 3 完了日**: 2025/10/20
**最新コミット**: 9d0128a "Prepare demo_tracking_ai for FeatureTracker integration"

## ドキュメント

詳細なドキュメントは`docs/`ディレクトリを参照してください：

- [プロジェクト概要](docs/PROJECT_OVERVIEW.md)
- [要件定義](docs/REQUIREMENTS.md)
- [技術スタック](docs/TECH_STACK.md)
- [アーキテクチャ](docs/ARCHITECTURE.md)
- [実装計画](docs/IMPLEMENTATION_PLAN.md)
- [AI戦略](docs/AI_MODEL_STRATEGY.md)
- [開発ガイド](docs/DEVELOPMENT_GUIDE.md)

## プロジェクト構造

```
METABALL-VirtualAd/
├── docs/                      # ドキュメント
├── src/                       # C++ソースコード
├── python/                    # Python学習スクリプト
├── models/                    # AIモデル（ONNX）
├── data/                      # 学習データ
├── assets/                    # 広告素材
├── scripts/                   # ビルドスクリプト
└── tasks/                     # タスク管理
```

## タスク管理

タスクの進捗状況は [tasks/todo.md](tasks/todo.md) で管理しています。

## 技術スタック（Windows専用最適化）

- **C++20**: コア処理
- **Python 3.10+**: AI学習
- **DirectX 12**: レンダリング（Windows）
- **CUDA 12.x / TensorRT 8.6+**: GPU最適化・AI推論（NVIDIA専用）
- **OpenCV 4.8+**: 画像処理
- **Qt 6.5+**: GUI

## ライセンス

(ライセンスを指定してください)

## 参考

- [Viz Arena](https://www.vizrt.com/products/viz-arena/)
