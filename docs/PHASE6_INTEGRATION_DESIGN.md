# PHASE 6: 統合・最適化 設計ドキュメント

**Phase**: 6
**目標**: 全コンポーネント統合、マルチスレッド最適化、60fps達成
**推定期間**: 5-7日
**作成日**: 2025/10/20
**ステータス**: 設計中

---

## 1. プロジェクト概要

### 1.1 Phase 6の位置づけ

Phase 6は、METABALL Virtual Adプロジェクトの最終フェーズであり、以下を達成します：

1. **全コンポーネント統合**: Phase 1-5で実装した各コンポーネントを統合
2. **マルチスレッド最適化**: 並列処理による性能向上
3. **60fps達成**: リアルタイム処理の最終目標達成
4. **安定性確保**: 24時間連続動作の実現

### 1.2 前提条件（Phase 1-5完了済み）

#### Phase 1: 映像I/O
- FilePlaybackSource実装済み
- WindowPreviewOutput実装済み
- FramePipeline基盤実装済み

#### Phase 2-3: AIトラッキング
- FeatureTracker実装済み（ORB/AKAZE）
- PnPSolver実装済み
- CameraPoseRefiner実装済み（PnP + AI補正）
- ONNX Runtime統合済み

#### Phase 4: AIキーヤー
- SegmentationInference実装済み（DeepLabV3+）
- DepthEstimator実装済み（MiDaS）
- DepthCompositor実装済み

#### Phase 5: レンダリング
- AdRenderer実装済み（透視変換）
- 3つのブレンディングモード実装済み

### 1.3 現在の状態

**実装完了コンポーネント**: 5/6フェーズ完了（約85%）

```
Phase 1 (部分)  Phase 2-3      Phase 4        Phase 5
┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Video I/O  │→ │ Tracking │→ │  Keyer   │→ │Rendering │
└────────────┘  └──────────┘  └──────────┘  └──────────┘
     ✓              ✓              ✓              ✓
                                                    
                    Phase 6: Integration
              ┌──────────────────────────┐
              │  統合パイプライン構築    │
              │  マルチスレッド最適化    │
              │  60fps達成               │
              └──────────────────────────┘
                         ⬇
                  Final Product
```

**実装済みファイル数**: 約50ファイル
**総実装行数**: 約9,337行
**実行可能プログラム**: 11個

---

## 2. Phase 6の目標

### 2.1 機能的目標

1. **エンドツーエンドパイプライン実装**
   - ビデオ入力 → トラッキング → キーイング → レンダリング → 出力
   - すべてのコンポーネントをシームレスに統合

2. **リアルタイム処理実現**
   - 60fps安定動作（95%以上のフレームが16.67ms以内）
   - フレームドロップ率0.1%以下

3. **高品質合成**
   - デプスベースキーイング動作
   - 選手がバーチャル広告の前に自然に表示
   - ブレンディングモード切り替え可能

4. **安定性確保**
   - 24時間連続動作
   - クラッシュ0回
   - 適切なエラーハンドリング

### 2.2 性能目標

#### フレーム処理時間内訳（目標: 合計≤16.67ms）

| コンポーネント        | 目標時間 | 実測時間 | ステータス |
|----------------------|---------|---------|-----------|
| ビデオ入力           | ≤1ms    | TBD     | 未計測     |
| 特徴トラッキング     | ≤5ms    | TBD     | 未計測     |
| カメラポーズ推定     | ≤5ms    | TBD     | 未計測     |
| セグメンテーション   | ≤3ms    | TBD     | 未計測     |
| デプス推定           | ≤2ms    | TBD     | 未計測     |
| 広告レンダリング     | ≤2ms    | TBD     | 未計測     |
| デプス合成           | ≤2ms    | TBD     | 未計測     |
| ビデオ出力           | ≤1ms    | TBD     | 未計測     |
| **合計**             | **≤21ms** | **TBD** | **要最適化** |

**注**: 現在の目標合計21msは16.67msを超えているため、マルチスレッド最適化が必須

#### メモリ使用量目標

- **合計メモリ**: ≤8GB
- **GPU VRAM**: ≤4GB（RTX 3060想定）

### 2.3 開発目標

1. **コード品質**
   - すべてのパブリックAPIにドキュメンテーションコメント
   - エラーハンドリング完備
   - ログ出力統一

2. **テスト**
   - 統合テストプログラム実装
   - パフォーマンステスト実装
   - 安定性テスト（24時間）

3. **ドキュメント**
   - ユーザーマニュアル作成
   - 開発者ガイド更新
   - トラブルシューティングガイド作成

---

## 3. 現在のアーキテクチャ分析

### 3.1 既存コンポーネント

#### 3.1.1 Core（Phase 1）

**FilePlaybackSource** (`src/core/FilePlaybackSource.h/cpp`)
- 役割: ビデオファイルからフレーム読み込み
- API: `IVideoSource`インターフェース実装
- 性能: 60fps対応、シーク機能あり

**WindowPreviewOutput** (`src/core/WindowPreviewOutput.h/cpp`)
- 役割: プレビュー表示
- API: `IVideoOutput`インターフェース実装
- 性能: OpenCV imshow使用

**FramePipeline** (`src/core/FramePipeline.h/cpp`)
- 役割: フレーム処理パイプライン基盤
- API: ソース・出力・プロセッサ登録
- 制限: 現状は単純なフロー、マルチスレッド未対応

#### 3.1.2 Tracking（Phase 2-3）

**FeatureTracker** (`src/tracking/FeatureTracker.h/cpp`)
- 役割: ORB/AKAZE特徴点トラッキング
- API: `initialize()`, `track()`, `getStatistics()`
- 性能: 目標≤5ms/frame

**PnPSolver** (`src/tracking/PnPSolver.h/cpp`)
- 役割: カメラポーズ推定（OpenCV solvePnP）
- API: `solve()`, `estimateHomography()`
- 性能: 高速（数ms）

**CameraPoseRefiner** (`src/inference/CameraPoseRefiner.h/cpp`)
- 役割: PnP + AI補正統合
- API: `refinePose()`, `setMode()`, `setBlendAlpha()`
- モード: PNP_ONLY, AI_ONLY, BLENDED
- 性能: 目標≤10ms/frame

#### 3.1.3 Keyer（Phase 4）

**SegmentationInference** (`src/keyer/SegmentationInference.h/cpp`)
- 役割: セグメンテーション推論（DeepLabV3+）
- API: `loadModel()`, `segment()`
- 性能: 目標≤3ms/frame

**DepthEstimator** (`src/keyer/DepthEstimator.h/cpp`)
- 役割: デプス推定（MiDaS）
- API: `loadModel()`, `estimate()`
- 性能: 目標≤2ms/frame

**DepthCompositor** (`src/keyer/DepthCompositor.h/cpp`)
- 役割: デプスベース合成
- API: `composite()`, `compositeSimple()`, `compositePixelwise()`
- 性能: 目標≤2ms/frame

#### 3.1.4 Rendering（Phase 5）

**AdRenderer** (`src/rendering/AdRenderer.h/cpp`)
- 役割: 広告レンダリング（透視変換）
- API: `initialize()`, `render()`, `setBlendMode()`
- ブレンディングモード: REPLACE, ALPHA_BLEND, ADDITIVE
- 性能: 目標≤2ms/frame

### 3.2 統合の課題

#### 3.2.1 性能課題

**現在の処理時間合計**: 約21ms（目標16.67msを超過）

**ボトルネック予測**:
1. セグメンテーション推論（約3ms）
2. 特徴トラッキング（約5ms）
3. カメラポーズ推定（約5ms）

**解決策**:
- マルチスレッド化（パイプライン並列処理）
- GPU最適化（CUDA/TensorRT）
- 処理スキップ戦略（フレームごとに処理を分散）

#### 3.2.2 データフロー課題

**現在の状態**: 各コンポーネントが独立

**必要な統合**:
- フレームバッファ共有
- カメラポーズ情報の伝播
- セグメンテーション/デプスマスクの伝播
- レンダリング結果の合成

**データ構造設計が必要**:
```cpp
struct FrameData {
    cv::Mat image;                    // 入力画像
    cv::Mat rvec, tvec;               // カメラポーズ
    cv::Mat segmentation_mask;        // セグメンテーション
    cv::Mat depth_map;                // デプスマップ
    cv::Mat rendered_ad;              // レンダリング済み広告
    cv::Mat final_output;             // 最終出力
    double timestamp;                 // タイムスタンプ
};
```

#### 3.2.3 エラーハンドリング課題

**現在の状態**: 各コンポーネントが独自のエラー処理

**統一が必要**:
- エラーコード体系
- 例外処理戦略
- ログ出力フォーマット
- リカバリ戦略

---

## 4. 統合パイプライン設計

### 4.1 統合アーキテクチャ概要

#### 4.1.1 シングルスレッド版（ベースライン）

**目的**: 機能統合の確認、デバッグ容易性

```
┌──────────────────────────────────────────────────────┐
│                    Main Thread                        │
│                                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │ Video   │→ │Tracking │→ │ Keyer   │→ │Rendering││
│  │ Input   │  │         │  │         │  │         ││
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘│
│       ↓            ↓            ↓            ↓       │
│  FrameData    CameraPose   SegMask/     Rendered    │
│                              DepthMap      Output    │
└──────────────────────────────────────────────────────┘
```

**処理フロー**:
1. ビデオ入力（1ms）
2. 特徴トラッキング（5ms）
3. カメラポーズ推定（5ms）
4. セグメンテーション推論（3ms）
5. デプス推定（2ms）
6. 広告レンダリング（2ms）
7. デプス合成（2ms）
8. ビデオ出力（1ms）

**合計**: 約21ms/frame → **47fps**（目標60fps未達）

#### 4.1.2 マルチスレッド版（最適化）

**目的**: 60fps達成、並列処理による性能向上

```
┌──────────────────────────────────────────────────────────────┐
│                     Main Thread                               │
│                   (Orchestrator)                              │
│                                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Frame    │→  │ Frame    │→  │ Frame    │→  │ Video    │ │
│  │ Producer │   │ Queue    │   │ Consumer │   │ Output   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ↓              ↓              ↓              ↑          │
└──────┼──────────────┼──────────────┼──────────────┼──────────┘
       │              │              │              │
┌──────┼──────────────┼──────────────┼──────────────┼──────────┐
│      ↓              ↓              ↓              │           │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐      │
│  │Tracking│    │ Keyer  │    │Render/ │    │ Output │      │
│  │ Thread │    │ Thread │    │Composite│   │ Thread │      │
│  │        │    │        │    │ Thread  │   │        │      │
│  └────────┘    └────────┘    └────────┘    └────────┘      │
│      │              │              │              │           │
│      └──────────────┴──────────────┴──────────────┘          │
│                   Thread Pool                                 │
└──────────────────────────────────────────────────────────────┘
```

**スレッド構成**:

1. **Main Thread（Orchestrator）**
   - フレーム読み込み
   - スレッド間同期
   - 結果収集

2. **Tracking Thread**
   - 特徴トラッキング（5ms）
   - カメラポーズ推定（5ms）
   - 合計: 10ms

3. **Keyer Thread**
   - セグメンテーション推論（3ms）
   - デプス推定（2ms）
   - 合計: 5ms

4. **Render/Composite Thread**
   - 広告レンダリング（2ms）
   - デプス合成（2ms）
   - 合計: 4ms

**並列実行時の理論処理時間**: max(10ms, 5ms, 4ms) = **10ms/frame** → **100fps**（目標達成）

### 4.2 データ構造設計

#### 4.2.1 FrameData構造体

```cpp
namespace VirtualAd {
namespace Integration {

/**
 * @brief フレーム処理に必要なすべてのデータを保持
 */
struct FrameData {
    // 基本情報
    int64_t frame_id;                  // フレームID
    double timestamp;                   // タイムスタンプ（秒）
    
    // 画像データ
    cv::Mat image;                      // 入力画像（BGR、1920x1080）
    
    // トラッキング結果
    bool tracking_success;              // トラッキング成功フラグ
    cv::Mat rvec;                       // 回転ベクトル（3x1）
    cv::Mat tvec;                       // 並進ベクトル（3x1）
    std::vector<cv::Point2f> corners;   // 検出コーナー（4点）
    int inlier_count;                   // インライア数
    
    // キーヤー結果
    cv::Mat segmentation_mask;          // セグメンテーションマスク（CV_8UC1）
    cv::Mat depth_map;                  // デプスマップ（CV_32FC1、正規化済み）
    
    // レンダリング結果
    cv::Mat rendered_ad;                // レンダリング済み広告（BGR）
    cv::Mat final_output;               // 最終出力（BGR、1920x1080）
    
    // パフォーマンス統計
    double tracking_time_ms;            // トラッキング処理時間
    double keyer_time_ms;               // キーヤー処理時間
    double rendering_time_ms;           // レンダリング処理時間
    double total_time_ms;               // 総処理時間
    
    /**
     * @brief コンストラクタ
     */
    FrameData() 
        : frame_id(0)
        , timestamp(0.0)
        , tracking_success(false)
        , inlier_count(0)
        , tracking_time_ms(0.0)
        , keyer_time_ms(0.0)
        , rendering_time_ms(0.0)
        , total_time_ms(0.0)
    {}
};

} // namespace Integration
} // namespace VirtualAd
```

#### 4.2.2 スレッドセーフキュー

```cpp
/**
 * @brief スレッドセーフなフレームキュー
 */
template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}
    
    /**
     * @brief キューに要素を追加（タイムアウトあり）
     * @param value 追加する要素
     * @param timeout_ms タイムアウト時間（ms）
     * @return 成功時true、タイムアウト時false
     */
    bool push(const T& value, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_not_full_.wait_for(lock, 
                std::chrono::milliseconds(timeout_ms),
                [this]{ return queue_.size() < max_size_; })) {
            return false;  // タイムアウト
        }
        queue_.push(value);
        cond_not_empty_.notify_one();
        return true;
    }
    
    /**
     * @brief キューから要素を取得（タイムアウトあり）
     * @param value 取得した要素の格納先
     * @param timeout_ms タイムアウト時間（ms）
     * @return 成功時true、タイムアウト時false
     */
    bool pop(T& value, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_not_empty_.wait_for(lock,
                std::chrono::milliseconds(timeout_ms),
                [this]{ return !queue_.empty(); })) {
            return false;  // タイムアウト
        }
        value = queue_.front();
        queue_.pop();
        cond_not_full_.notify_one();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    std::queue<T> queue_;
    size_t max_size_;
    mutable std::mutex mutex_;
    std::condition_variable cond_not_full_;
    std::condition_variable cond_not_empty_;
};
```

### 4.3 統合パイプラインクラス設計

#### 4.3.1 IntegratedPipelineクラス

```cpp
namespace VirtualAd {
namespace Integration {

/**
 * @brief 統合パイプラインクラス
 * 
 * すべてのコンポーネント（Tracking, Keyer, Rendering）を統合し、
 * マルチスレッド処理を実現します。
 */
class IntegratedPipeline {
public:
    /**
     * @brief 動作モード
     */
    enum class Mode {
        SINGLE_THREAD,  // シングルスレッド（デバッグ用）
        MULTI_THREAD    // マルチスレッド（本番用）
    };
    
    /**
     * @brief コンストラクタ
     */
    IntegratedPipeline();
    
    /**
     * @brief デストラクタ
     */
    ~IntegratedPipeline();
    
    /**
     * @brief パイプライン初期化
     * @param config 設定パラメータ
     * @return 成功時true
     */
    bool initialize(const PipelineConfig& config);
    
    /**
     * @brief パイプライン開始
     * @return 成功時true
     */
    bool start();
    
    /**
     * @brief パイプライン停止
     */
    void stop();
    
    /**
     * @brief 1フレーム処理
     * @param frame 入力フレーム
     * @param output 出力フレーム
     * @return 成功時true
     */
    bool processFrame(const cv::Mat& frame, cv::Mat& output);
    
    /**
     * @brief 統計情報取得
     */
    PipelineStatistics getStatistics() const;
    
private:
    // コンポーネント
    std::unique_ptr<Tracking::FeatureTracker> tracker_;
    std::unique_ptr<Inference::CameraPoseRefiner> pose_refiner_;
    std::unique_ptr<Keyer::SegmentationInference> segmentation_;
    std::unique_ptr<Keyer::DepthEstimator> depth_estimator_;
    std::unique_ptr<Keyer::DepthCompositor> compositor_;
    std::unique_ptr<Rendering::AdRenderer> renderer_;
    
    // スレッド管理
    Mode mode_;
    std::atomic<bool> running_;
    std::thread tracking_thread_;
    std::thread keyer_thread_;
    std::thread render_thread_;
    
    // キュー
    ThreadSafeQueue<std::shared_ptr<FrameData>> input_queue_;
    ThreadSafeQueue<std::shared_ptr<FrameData>> tracking_queue_;
    ThreadSafeQueue<std::shared_ptr<FrameData>> keyer_queue_;
    ThreadSafeQueue<std::shared_ptr<FrameData>> render_queue_;
    ThreadSafeQueue<std::shared_ptr<FrameData>> output_queue_;
    
    // 統計情報
    mutable std::mutex stats_mutex_;
    PipelineStatistics statistics_;
    
    // スレッド関数
    void trackingThreadFunc();
    void keyerThreadFunc();
    void renderThreadFunc();
    
    // シングルスレッド処理
    bool processFrameSingleThread(std::shared_ptr<FrameData> frame_data);
};

} // namespace Integration
} // namespace VirtualAd
```

---

## 5. 次のセクション予定

### Part 3（次回実装）
- 実装計画（ステップバイステップ）
- テスト戦略
- パフォーマンス測定計画

---

**現在のドキュメント進捗**: Part 2/3完了
**次のステップ**: Part 3実装（実装計画・テスト戦略）
