# Phase 4: AIキーヤー設計ドキュメント

**作成日**: 2025/10/20
**ステータス**: 設計中

## 1. 概要

Phase 4では、選手がバーチャル広告の前に自然に表示されるよう、AIベースのセグメンテーションとデプス推定を統合します。

### 目標
- セグメンテーション精度: IoU > 0.90
- デプス推定動作確認
- 合成が自然（目視確認）
- 処理時間: 8ms/frame以内

## 2. システムアーキテクチャ

```
入力画像
  ├─→ セグメンテーションモデル → 前景/背景マスク
  ├─→ デプス推定モデル → デプスマップ
  └─→ カメラトラッキング → ポーズ情報
           ↓
    デプスベース合成
           ↓
    最終出力（選手が広告の前に表示）
```

## 3. セグメンテーションモデル

### 3.1 モデル選定
**選択**: DeepLabV3+ (MobileNetV3 Backbone)

**理由**:
- 推論速度: ~4-5ms/frame（FP16、RTX 3060）
- 精度: COCO dataset mIoU ~75%
- TensorRT最適化対応
- 軽量バックボーン（MobileNetV3）

**代替案**:
- Mask R-CNN: 精度高いが遅い（~15ms/frame）
- U-Net: シンプルだが精度不足
- SegFormer: 最新だが検証不足

### 3.2 ファインチューニング計画

#### データ収集
- **目標**: 100フレーム以上
- **方法**: SAM（Segment Anything Model）で半自動アノテーション
- **内容**: 野球選手、審判、バックネット、観客
- **多様性**: 
  - 照明条件（昼・夜、晴・曇）
  - カメラアングル（正面・側面）
  - 選手位置（近景・遠景）

#### アノテーション手順
1. SAMで初期セグメンテーション
2. 手動で境界調整
3. JSON形式で保存（COCO format）
4. 品質確認（境界精度チェック）

#### 学習設定
```python
# モデル設定
model = deeplabv3plus_mobilenet(
    num_classes=4,  # 選手、審判、バックネット、背景
    output_stride=16,
    pretrained_backbone=True
)

# 学習パラメータ
batch_size = 8
epochs = 50
learning_rate = 1e-4
optimizer = Adam
loss = CrossEntropyLoss + DiceLoss（加重平均）
augmentation = [
    RandomHorizontalFlip(),
    ColorJitter(),
    RandomRotation(±10°),
    RandomScale(0.8-1.2)
]
```

#### 評価指標
- mIoU（各クラス平均）
- Boundary IoU（境界精度）
- 推論時間（目標: <5ms）

### 3.3 ONNX変換
```python
# PyTorch → ONNX
torch.onnx.export(
    model,
    dummy_input,
    "segmentation_model.onnx",
    input_names=['image'],
    output_names=['mask'],
    dynamic_axes={'image': {0: 'batch'}},
    opset_version=17
)

# ONNX → TensorRT（INT8量子化）
trtexec --onnx=segmentation_model.onnx \
        --saveEngine=segmentation_model.engine \
        --int8 \
        --calib=calibration_cache.bin \
        --fp16
```

## 4. デプス推定モデル

### 4.1 モデル選定
**選択**: MiDaS Small

**理由**:
- 推論速度: ~3-4ms/frame（FP16、RTX 3060）
- 相対デプス推定（絶対値不要）
- 事前学習済み（ファインチューニング不要）
- TensorRT最適化対応

**代替案**:
- MiDaS Large: 精度高いが遅い（~10ms/frame）
- DPT: Transformer-based、精度良いが重い

### 4.2 統合方法
```cpp
// C++統合例
class DepthEstimator {
public:
    bool loadModel(const std::string& model_path);
    bool estimate(const cv::Mat& image, cv::Mat& depth_map);
    bool isLoaded() const;
    
private:
    std::unique_ptr<Ort::Session> session_;
    cv::Size input_size_{384, 384};  // MiDaS Small入力サイズ
};
```

## 5. デプスベース合成実装

### 5.1 合成アルゴリズム

```cpp
/**
 * @brief デプスベース合成
 * 
 * @param image 入力画像
 * @param segmentation_mask セグメンテーションマスク（4チャンネル）
 * @param depth_map デプスマップ
 * @param ad_texture 広告テクスチャ
 * @param camera_pose カメラポーズ
 * @param output 出力画像
 */
void compositeWithDepth(
    const cv::Mat& image,
    const cv::Mat& segmentation_mask,
    const cv::Mat& depth_map,
    const cv::Mat& ad_texture,
    const CameraPose& camera_pose,
    cv::Mat& output
) {
    // 1. 広告をバックネット平面に投影
    cv::Mat projected_ad = projectAdToBacknet(ad_texture, camera_pose);
    
    // 2. デプス比較
    // - depth_map: 相対デプス（小さいほど手前）
    // - backnet_depth: バックネット平面のデプス値（固定）
    float backnet_depth = estimateBacknetDepth(camera_pose);
    
    // 3. ピクセル単位合成
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // セグメンテーション判定
            int class_id = segmentation_mask.at<int>(y, x);
            float pixel_depth = depth_map.at<float>(y, x);
            
            if (class_id == PLAYER || class_id == UMPIRE) {
                // 選手/審判: デプス比較
                if (pixel_depth < backnet_depth) {
                    // 選手が手前 → 元画像
                    output.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
                } else {
                    // 選手が奥 → 広告
                    output.at<cv::Vec3b>(y, x) = projected_ad.at<cv::Vec3b>(y, x);
                }
            } else if (class_id == BACKNET) {
                // バックネット → 広告
                output.at<cv::Vec3b>(y, x) = projected_ad.at<cv::Vec3b>(y, x);
            } else {
                // 背景 → 元画像
                output.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
            }
        }
    }
}
```

### 5.2 CUDA最適化版

```cuda
__global__ void compositeKernel(
    const uchar3* image,
    const int* seg_mask,
    const float* depth_map,
    const uchar3* projected_ad,
    float backnet_depth,
    uchar3* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int class_id = seg_mask[idx];
    float pixel_depth = depth_map[idx];
    
    if (class_id == PLAYER || class_id == UMPIRE) {
        output[idx] = (pixel_depth < backnet_depth) ? 
            image[idx] : projected_ad[idx];
    } else if (class_id == BACKNET) {
        output[idx] = projected_ad[idx];
    } else {
        output[idx] = image[idx];
    }
}
```

## 6. エッジリファインメント

### 6.1 境界平滑化
```cpp
void refineEdges(cv::Mat& composite, const cv::Mat& seg_mask) {
    // ガイデッドフィルタで境界をソフトに
    cv::ximgproc::guidedFilter(
        composite,  // guide image
        composite,  // src
        composite,  // dst
        5,          // radius
        0.1         // epsilon
    );
}
```

### 6.2 時間的平滑化
```cpp
class TemporalSmoother {
public:
    void addFrame(const cv::Mat& mask);
    cv::Mat getSmoothedMask();
    
private:
    std::deque<cv::Mat> mask_history_;
    const int HISTORY_SIZE = 5;
};
```

## 7. 実装計画

### Phase 4-1: セグメンテーションモデル開発（3-4日）
- [x] モデル選定（DeepLabV3+）
- [ ] SAMアノテーション環境構築
- [ ] データ収集（100フレーム）
- [ ] ファインチューニング実装
- [ ] ONNX変換・TensorRT最適化

### Phase 4-2: デプス推定統合（1-2日）
- [ ] MiDaS Smallダウンロード
- [ ] ONNX変換
- [ ] C++推論実装
- [ ] パフォーマンス測定

### Phase 4-3: C++推論統合（2-3日）
- [ ] SegmentationInference.h/cpp実装
- [ ] DepthEstimator.h/cpp実装
- [ ] バッチ処理最適化
- [ ] GPU-GPU転送（CUDA Interop）

### Phase 4-4: デプスベース合成（2-3日）
- [ ] compositeWithDepth()実装（CPU版）
- [ ] CUDAカーネル実装
- [ ] エッジリファインメント
- [ ] 時間的平滑化

### Phase 4-5: 品質検証（1-2日）
- [ ] セグメンテーション精度測定（IoU）
- [ ] デプス推定動作確認
- [ ] 視覚品質評価
- [ ] パフォーマンス測定（目標: 8ms/frame）

## 8. ファイル構成（予定）

```
src/keyer/
├── SegmentationInference.h/cpp    # セグメンテーション推論
├── DepthEstimator.h/cpp            # デプス推定
├── DepthCompositor.h/cpp           # デプスベース合成
├── EdgeRefiner.h/cpp               # エッジリファインメント
├── TemporalSmoother.h/cpp          # 時間的平滑化
├── test_segmentation.cpp           # セグメンテーションテスト
├── test_depth_estimation.cpp       # デプス推定テスト
└── test_keyer_pipeline.cpp         # 統合テスト

python/keyer/
├── sam_annotation.py               # SAMアノテーションツール
├── train_segmentation.py           # セグメンテーション学習
├── export_segmentation_onnx.py     # ONNX変換
└── evaluate_segmentation.py        # セグメンテーション評価
```

## 9. 性能目標

| コンポーネント | 目標処理時間 |
|---------------|-------------|
| セグメンテーション推論 | < 5ms |
| デプス推定推論 | < 3ms |
| デプスベース合成 | < 0.5ms |
| エッジリファインメント | < 0.5ms |
| **合計** | **< 8ms** |

## 10. リスク管理

### 高リスク
1. **セグメンテーション精度不足**
   - 対策: データ収集強化、モデル変更検討
2. **処理時間オーバー**
   - 対策: TensorRT INT8量子化、CUDAカーネル最適化

### 中リスク
1. **デプス推定誤差**
   - 対策: MiDaS Largeへアップグレード検討
2. **エッジ品質不足**
   - 対策: より高度なリファインメント手法（Matting）

## 11. 次のステップ

**即座に開始可能**:
1. SAMアノテーション環境構築
2. サンプルフレーム選定
3. DeepLabV3+ PyTorch実装セットアップ

**準備中**:
- GPU環境確認（CUDA対応）
- TensorRT環境構築
