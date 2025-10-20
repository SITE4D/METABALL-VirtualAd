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
git clone <repository-url>
cd METABALL-VirtualAd

# 2. 依存関係インストール
scripts\setup_windows.bat

# 3. ビルド
scripts\build.bat

# 4. 実行
build\Release\VirtualAd.exe
```

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
