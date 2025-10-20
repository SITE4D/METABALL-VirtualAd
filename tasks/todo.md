# METABALL Virtual Ad - タスク管理

## プロジェクト進捗

**開始日**: 2025/10/20
**目標完了日**: 2025/12/01 (6週間)
**現在のフェーズ**: Phase 3 - C++統合とデモ
**Phase 2完了日**: 2025/10/20 17:16

---

## Phase 0: 環境構築 (1-2日)

**方針**: タイムアウトを防ぐため、各ステップを小さく分割し、一つずつ確実に進める

### Step 1: 開発環境の確認
- [ ] Visual Studio 2022のインストール確認
  - [ ] バージョン確認コマンド実行
  - [ ] C++デスクトップ開発ワークロード確認
  - [ ] CMake統合ツール確認
- [ ] CUDA Toolkitのバージョン確認
  - [ ] `nvcc --version`実行
  - [ ] 環境変数CUDA_PATH確認
- [ ] CMakeのバージョン確認
  - [ ] `cmake --version`実行（3.20+必要）
- [ ] Git確認
  - [ ] `git --version`実行

### Step 2: CMakeプロジェクトのセットアップ（最小構成）
- [ ] ディレクトリ構造の整備
  - [ ] src/core/ディレクトリ作成
  - [ ] src/gui/ディレクトリ作成
  - [ ] tests/ディレクトリ作成
  - [ ] scripts/ディレクトリ作成
- [ ] 最小限のルートCMakeLists.txt作成
  - [ ] CMake最小バージョン指定
  - [ ] プロジェクト名設定
  - [ ] C++20標準設定
  - [ ] 基本的なビルド設定
- [ ] .gitignore更新
  - [ ] build/ディレクトリ追加
  - [ ] Visual Studio関連ファイル追加

### Step 3: Hello Worldアプリケーション
- [ ] 最小限のmain.cpp作成
  - [ ] "Hello, METABALL Virtual Ad!"出力
  - [ ] 正常終了コード
- [ ] CMakeLists.txtに実行可能ファイル追加
  - [ ] add_executable設定
- [ ] ビルド確認
  - [ ] `cmake -B build`実行
  - [ ] `cmake --build build`実行
- [ ] 実行確認
  - [ ] 生成された.exeファイル実行
  - [ ] 出力確認

### Step 4: vcpkgのセットアップ
- [ ] vcpkgインストール確認
  - [ ] vcpkgディレクトリ確認
  - [ ] `vcpkg version`実行
- [ ] Visual Studio統合確認
  - [ ] `vcpkg integrate install`実行済み確認
- [ ] CMakeLists.txtにvcpkg統合
  - [ ] CMAKE_TOOLCHAIN_FILE設定追加
  - [ ] vcpkg.jsonマニフェストファイル作成（後のステップ用）

### Step 5: OpenCVの基本インストールとテスト
- [ ] OpenCVインストール（基本版のみ）
  - [ ] `vcpkg install opencv4:x64-windows`実行
  - [ ] インストール完了確認（15-30分程度）
- [ ] CMakeLists.txtにOpenCV追加
  - [ ] find_package(OpenCV REQUIRED)
  - [ ] target_link_libraries設定
- [ ] 画像読み込みテストプログラム作成
  - [ ] main.cppを更新
  - [ ] 簡単な画像読み込みコード追加
  - [ ] テスト画像準備（assets/test.jpg）
- [ ] ビルド・実行確認
  - [ ] リビルド実行
  - [ ] 画像読み込み動作確認

### Phase 0完了基準
- [ ] Hello Worldアプリケーションが正常動作
- [ ] OpenCVを使った画像読み込みが成功
- [ ] GitHubリポジトリにコミット・プッシュ完了

**注意**: CUDA版OpenCV、Qt、TensorRTなどの大規模ライブラリは、Phase 1以降で必要になった時点でインストールする

---

## Phase 1: 映像I/Oパイプライン (4-6日)

### アーキテクチャ設計
- [x] IVideoSourceインターフェース設計
- [x] IVideoOutputインターフェース設計
- [x] FramePipelineクラス設計

### ライブキャプチャ実装
- [ ] DirectShowCaptureクラス実装
  - [ ] デバイス列挙
  - [ ] キャプチャ開始・停止
  - [ ] フレーム取得
  - [ ] 60fps確認
- [ ] エラーハンドリング
- [ ] ログ出力

### ファイル再生実装
- [x] FilePlaybackSourceクラス実装
  - [x] OpenCV VideoCapture利用
  - [x] 60fpsシミュレーション
  - [x] フレーム単位シーク
  - [x] 一時停止・再開
- [x] サンプル映像での動作確認

### 映像出力実装
- [x] WindowPreviewOutput実装（OpenCV imshow）
- [x] FileWriterOutput実装（H.264録画）
- [x] FramePipeline統合テスト
- [ ] HDMI出力クラス実装（Phase 1 Step 4以降）

### GUIプロトタイプ
- [ ] Qt6 メインウィンドウ作成
- [ ] ビデオプレビューウィジェット
- [ ] モード切り替えUI（ライブ/ファイル）
- [ ] 再生コントロール（再生・停止・シーク）

### パフォーマンス測定
- [ ] フレームレート測定機能実装
- [ ] CPU/GPU使用率モニタリング
- [ ] 60fps達成確認

### 完了基準
- [ ] ライブキャプチャで60fpsパススルー動作
- [ ] ファイル再生で60fpsシミュレーション
- [ ] モード切り替えが1秒以内
- [ ] フレームドロップ率0.1%以下

---

## Phase 1.5: データ収集ツール (3-4日)

### アノテーションツールGUI
- [ ] インタラクティブ座標指定機能
  - [ ] マウスクリックでバックネット4隅指定
  - [ ] グリッド表示
  - [ ] Undo/Redo機能
  - [ ] プレビュー表示
- [ ] サンプル保存機能
  - [ ] フレーム画像保存
  - [ ] JSON形式でメタデータ保存
- [ ] バッチ処理UI

### カメラパラメータ計算
- [ ] solvePnP統合
- [ ] カメラ行列推定
- [ ] 歪み係数推定
- [ ] パラメータ検証

### データ管理
- [ ] JSON Schema定義
- [ ] データローダー実装
- [ ] データセット統計表示

### サンプル収集
- [ ] サンプル映像から100フレーム以上収集
- [ ] 様々な画角・照明条件を含む
- [ ] データ品質確認

### 完了基準
- [ ] アノテーション効率100サンプル/時間以上
- [ ] 100サンプル以上収集完了
- [ ] データ整合性100%

---

## Phase 2: AI補正パイプライン（Python） - **完了** ✓

**完了日**: 2025/10/20 17:16

### Step 4: データセット・変換（3ファイル完了）
- [x] **python/training/dataset.py** (251行)
  - [x] CameraPoseDataset実装
  - [x] アノテーションJSON読み込み
  - [x] PyTorch Dataset統合

- [x] **python/training/transforms.py** (90行)
  - [x] データ拡張（RandomHorizontalFlip, ColorJitter, RandomRotation）
  - [x] 正規化（ImageNet平均・標準偏差）
  - [x] get_training_transforms(), get_inference_transforms()

- [x] **python/training/test_dataloader.py** (88行)
  - [x] データローダー動作確認スクリプト
  - [x] サンプル画像・ポーズ表示

### Step 5: モデル・学習・評価（3ファイル完了）
- [x] **python/training/models.py** (134行)
  - [x] CameraPoseNet: ResNet-18ベースのポーズ推定
  - [x] 出力: [rvec(3) + tvec(3)]
  - [x] create_model(): ResNet18/34/50対応

- [x] **python/training/train.py** (410行)
  - [x] 完全な学習パイプライン
  - [x] train_one_epoch(), validate(), save_checkpoint()
  - [x] Adam optimizer、MSE Loss
  - [x] チェックポイント保存・レジューム機能

- [x] **python/training/evaluate.py** (580行)
  - [x] モデル評価スクリプト
  - [x] 再投影誤差計算（cv2.projectPoints）
  - [x] 回転・並進誤差計算
  - [x] 可視化（4種類のプロット）
  - [x] JSONレポート生成

### Step 6: ONNX変換（1ファイル完了）
- [x] **python/training/export_onnx.py** (447行)
  - [x] PyTorch→ONNX変換
  - [x] 動的バッチサイズ対応
  - [x] ONNX検証（onnx.checker）
  - [x] ONNX Runtime推論テスト
  - [x] PyTorch vs ONNX精度比較

### 使用方法サマリー

```bash
# 1. 学習
python python/training/train.py --data_dir ./data --epochs 100 --batch_size 32

# 2. 評価
python python/training/evaluate.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data

# 3. ONNX変換
python python/training/export_onnx.py --checkpoint ./checkpoints/best_model.pth --output ./models/camera_pose_net.onnx
```

### 完了基準
- [x] 全7ファイル実装完了（約2,000行）
- [x] PyTorch学習パイプライン完成
- [x] ONNX変換機能実装
- [x] GitHubプッシュ完了（コミット f7c70a1）

---

## Phase 2.5: トラッキング基盤（C++）

### 特徴点ベーストラッキング
- [ ] ORB/AKAZE特徴点検出実装
- [ ] フレーム間マッチング
- [ ] ホモグラフィ推定
- [ ] RANSAC外れ値除去

### PnPソルバー統合
- [ ] OpenCV solvePnP実装
- [ ] カメラパラメータ更新
- [ ] トラッキング継続率測定

---

## Phase 3: C++統合とデモ（現在のフェーズ） - **実装中**

**目標**: Phase 1のトラッキングとPhase 2のAI補正を統合し、エンドツーエンドのパイプラインを完成

### Step 3-1: ONNX推論ラッパー実装（15-20分）
- [ ] ONNXInference.h/cpp作成
  - [ ] ONNX Runtime C++ API統合
  - [ ] loadModel()メソッド実装
  - [ ] infer()メソッド実装
  - [ ] 前処理（リサイズ、正規化）実装
  - [ ] エラーハンドリング

- [ ] テストプログラム作成
  - [ ] test_onnx_inference.cpp作成
  - [ ] サンプル画像での推論確認
  - [ ] 推論時間測定
  - [ ] 出力精度確認

### Step 3-2: AI補正統合（20-25分）
- [ ] CameraPoseRefiner.h/cpp作成
  - [ ] PnPSolver + ONNX推論の統合
  - [ ] refinePose()メソッド実装
  - [ ] PnP結果とAI推論のブレンディング
  - [ ] エラーハンドリング

- [ ] 統合テスト
  - [ ] test_pose_refinement.cpp作成
  - [ ] 精度比較（PnPのみ vs PnP+AI補正）
  - [ ] パフォーマンス測定

### Step 3-3: パイプライン統合とデモ（15-20分）
- [ ] FramePipeline拡張
  - [ ] AI補正ステージ追加
  - [ ] オプション切り替え機能
  - [ ] パイプライン統合

- [ ] デモアプリケーション
  - [ ] demo_main.cpp作成
  - [ ] ビデオ入力 → トラッキング → AI補正 → 結果表示
  - [ ] 結果可視化
  - [ ] パフォーマンス表示

### 完了基準
- [ ] ONNX推論がC++で動作
- [ ] AI補正がPnPSolverと統合
- [ ] エンドツーエンドのデモが動作
- [ ] 推論時間5ms/frame以内

---

## Phase 4: AIキーヤー (8-12日)

### セグメンテーションモデル開発
- [ ] 学習データ収集
  - [ ] SAMで半自動アノテーション
  - [ ] 100フレーム以上
  - [ ] マスク品質確認
- [ ] DeepLabV3+ファインチューニング（Python）
  - [ ] 事前学習モデルロード
  - [ ] データ拡張実装
  - [ ] 学習実行
  - [ ] IoU測定（目標: >0.90）
- [ ] ONNX変換
- [ ] TensorRT最適化（INT8量子化）

### デプス推定モデル統合
- [ ] MiDaS Smallモデル選定
- [ ] ONNX変換
- [ ] TensorRT最適化
- [ ] 推論速度確認（目標: 4ms以内）

### C++推論統合
- [ ] セグメンテーション推論実装
- [ ] デプス推定推論実装
- [ ] バッチ処理最適化
- [ ] GPU-GPU直接転送

### デプスベース合成実装
- [ ] CUDAカーネル実装
  - [ ] マスク合成
  - [ ] デプス比較
  - [ ] 最終合成
- [ ] エッジリファインメント
- [ ] 時間的平滑化（フレーム間一貫性）

### 品質検証
- [ ] 視覚品質評価
- [ ] オクルージョン正確性測定
- [ ] エッジ品質確認

### 完了基準
- [ ] セグメンテーション精度IoU > 0.90
- [ ] デプス推定動作確認
- [ ] 合成が自然（目視確認）
- [ ] 処理時間8ms/frame以内

---

## Phase 4: バーチャル広告レンダリング (5-7日)

### DirectX 12レンダラー実装
- [ ] デバイス・スワップチェーン作成
- [ ] コマンドキュー・リスト作成
- [ ] ディスクリプタヒープ作成

### 透視変換シェーダー
- [ ] 頂点シェーダー実装
- [ ] ピクセルシェーダー実装
- [ ] カメラパラメータ渡し
- [ ] テクスチャサンプリング

### 広告配置エンジン
- [ ] 3D平面定義（バックネット）
- [ ] 透視変換行列計算
- [ ] テクスチャマッピング
- [ ] 位置精度検証（目標: 3px以内）

### ライティング調整
- [ ] シーン輝度解析
- [ ] 広告テクスチャ調整
- [ ] 自然さ確認

### 合成パイプライン
- [ ] DirectX + CUDA連携
- [ ] フレームバッファ管理
- [ ] 最終合成出力

### 完了基準
- [ ] 広告が正しい位置に配置
- [ ] 透視変換が正確
- [ ] ライティングが自然
- [ ] 処理時間2ms/frame以内

---

## Phase 5: 統合・最適化 (5-7日)

### 全パイプライン統合
- [ ] カメラトラッキング → キーヤー → レンダリング
- [ ] エラーハンドリング統合
- [ ] ログシステム統合

### マルチスレッド最適化
- [ ] スレッドプール実装
- [ ] キャプチャスレッド
- [ ] AI推論スレッド
- [ ] レンダリングスレッド
- [ ] スレッド同期最適化

### GPU最適化
- [ ] 非同期コマンド実行
- [ ] フレームバッファプリフェッチ
- [ ] メモリコピー削減

### 60fps達成確認
- [ ] フレームレート測定
- [ ] 95%以上のフレームが16.67ms以内
- [ ] ボトルネック特定・改善

### メモリ最適化
- [ ] メモリプロファイリング
- [ ] リーク検出・修正
- [ ] 使用量削減（目標: 8GB以下）

### 安定性テスト
- [ ] 24時間連続動作テスト
- [ ] クラッシュ0回確認
- [ ] エラーリカバリ確認

### UI完成
- [ ] 設定保存・読み込み
- [ ] プリセット機能
- [ ] ステータス表示
- [ ] エラー通知

### ドキュメント最終化
- [ ] コードコメント追加
- [ ] README更新
- [ ] ユーザーマニュアル作成

### 完了基準
- [ ] 60fps安定動作（95%以上）
- [ ] 24時間連続動作確認
- [ ] メモリ使用量8GB以下
- [ ] 全機能統合完了

---

## レビューセクション

### 現在の状態
- **Phase**: Phase 3 実装中（C++統合とデモ）
- **進捗率**: 45%（Phase 1部分完了、Phase 2完了、Phase 3開始）
- **次のマイルストーン**: ONNX推論ラッパー実装（Step 3-1）

### 完了した作業（Phase 0）
- [x] プロジェクト計画策定
- [x] 技術スタック選定
- [x] ドキュメント作成
  - [x] README.md
  - [x] PROJECT_OVERVIEW.md
  - [x] REQUIREMENTS.md
  - [x] TECH_STACK.md
  - [x] WINDOWS_SETUP_GUIDE.md
- [x] **Windows環境セットアップ完了**
  - [x] Visual Studio 2022確認（Version 17.14.16）
  - [x] CMake 3.31統合確認
  - [x] vcpkgマニフェストモードセットアップ
  - [x] OpenCV 4.8.0インストール・統合成功
  - [x] ビルドシステム動作確認
- [x] **CMakeプロジェクト構築**
  - [x] ディレクトリ構造整備（src/core, src/gui, tests, scripts）
  - [x] vcpkg.jsonマニフェスト作成
  - [x] UTF-8エンコーディング対応（/utf-8フラグ）
  - [x] Hello World + OpenCVテストプログラム動作確認
- [x] **GitHubリポジトリ更新**
  - [x] コミット・プッシュ完了

### 変更内容の概要
#### 実装した機能
1. **CMakeビルドシステム**
   - C++20標準設定
   - vcpkgマニフェストモード統合
   - OpenCV 4.8.0自動検出・リンク
   - MSVC UTF-8エンコーディング対応

2. **vcpkg依存関係管理**
   - vcpkg.jsonマニフェストファイル作成
   - OpenCV 4.8.0および依存パッケージ自動インストール（約15分）
   - CMAKE_PREFIX_PATH適切な設定

3. **検証プログラム**
   - システム情報表示
   - OpenCVバージョン・ビルド情報表示
   - cv::Mat作成・描画テスト
   - 正常動作確認

#### 技術的な詳細
- **ビルド環境**: Visual Studio 2022 Community (MSVC 19.44)
- **CMake**: 3.31.6（VS統合版）
- **OpenCV**: 4.8.0（vcpkg経由）
- **依存パッケージ**: protobuf, libjpeg-turbo, libpng, tiff, libwebp, zlib等（13パッケージ）

### 課題・リスク
1. **CUDA Toolkit未インストール**: Phase 1以降で必要（GPU最適化・AI推論に必須）
2. **出力キャプチャ問題**: PowerShellでの出力キャプチャに技術的問題あり（プログラム自体は正常動作）

### 完了した作業（Phase 1 - 2025/10/20）
- [x] **映像I/Oアーキテクチャ実装**
  - [x] IVideoSourceインターフェース作成
  - [x] IVideoOutputインターフェース作成
  - [x] FramePipelineクラス実装
- [x] **FilePlaybackSource実装**
  - [x] 映像ファイル読み込み機能
  - [x] フレームレート制御（60fps対応）
  - [x] シーク機能（フレーム単位）
  - [x] 一時停止・再開機能
  - [x] ループ再生機能
- [x] **テストプログラム作成**
  - [x] FilePlaybackSource動作確認
  - [x] パフォーマンス測定機能
  - [x] サンプル映像での動作テスト成功

### 完了した作業（Phase 2 - 2025/10/20 17:16）
- [x] **AI補正パイプライン（Python）完成**
  - [x] CameraPoseDataset実装（251行）
  - [x] データ拡張・変換実装（90行）
  - [x] CameraPoseNet実装（134行）
  - [x] 学習パイプライン実装（410行）
  - [x] 評価スクリプト実装（580行）
  - [x] ONNX変換実装（447行）
  - [x] 合計7ファイル、約2,000行実装
  - [x] GitHubプッシュ完了（コミット f7c70a1）

### 次のアクション
1. **Phase 3 Step 3-1: ONNX推論ラッパー実装**
   - ONNX Runtime C++ API統合
   - ONNXInference.h/cpp作成
   - テストプログラム作成
   
2. **Phase 3 Step 3-2: AI補正統合**
   - CameraPoseRefiner.h/cpp作成
   - PnPSolverとONNX推論の統合
   
3. **Phase 3 Step 3-3: デモアプリケーション**
   - エンドツーエンドパイプライン統合
   - デモプログラム作成

---

## メモ

- サンプル映像パス: `C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\2025-10-08_13-47-52.mp4`
- 広告サンプル: `C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\AD_FVILLAGE.png`
- Viz Arena仕様書: `C:\\Users\\SITE4D\\Documents\\_Assets\\VirtualAd\\VizArena.pdf`

---

**最終更新**: 2025/10/20 17:18
**Phase 0完了**: 2025/10/20 12:44
**Phase 1開始**: 2025/10/20 12:57
**Phase 2完了**: 2025/10/20 17:16
**Phase 3開始**: 2025/10/20 17:18
