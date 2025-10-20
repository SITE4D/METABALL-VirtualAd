# Phase 5: レンダリング - 設計ドキュメント

## 概要

Phase 5では、バーチャル広告を3D空間に配置し、カメラポーズに基づいて透視変換を適用してレンダリングするシステムを実装します。

**目標**:
- 3D広告テクスチャのレンダリング
- 透視変換による自然な配置
- カメラポーズに基づく正確な位置合わせ
- リアルタイム60fps処理

---

## アーキテクチャ

### コンポーネント構成

```
Phase 5: レンダリング
├── AdRenderer (C++)
│   ├── 3D平面定義（バックネット）
│   ├── 透視変換計算
│   ├── テクスチャマッピング
│   └── 合成処理
└── test_ad_renderer (C++)
    └── レンダリングテスト
```

### データフロー

```
カメラポーズ（rvec, tvec）
    ↓
AdRenderer
    ↓
3D→2D投影計算
    ↓
透視変換行列
    ↓
テクスチャマッピング
    ↓
合成画像出力
```

---

## AdRendererクラス設計

### 責務

1. **3D平面定義**: バックネット位置・サイズを3D座標で定義
2. **透視変換**: カメラポーズから2D投影を計算
3. **テクスチャマッピング**: 広告画像を投影
4. **合成処理**: 元画像に広告を合成

### クラス構造

```cpp
namespace VirtualAd {
namespace Rendering {

class AdRenderer {
public:
    // コンストラクタ・デストラクタ
    AdRenderer();
    ~AdRenderer();
    
    // 初期化
    bool initialize(const cv::Mat& camera_matrix, 
                   const cv::Mat& dist_coeffs);
    
    // バックネット3D平面設定
    void setBacknetPlane(const std::vector<cv::Point3f>& corners_3d);
    
    // 広告テクスチャ設定
    bool setAdTexture(const cv::Mat& ad_texture);
    
    // レンダリング実行
    bool render(const cv::Mat& image,
               const cv::Mat& rvec,
               const cv::Mat& tvec,
               cv::Mat& output);
    
    // ブレンディングモード設定
    enum class BlendMode {
        REPLACE,      // 完全置き換え
        ALPHA_BLEND,  // アルファブレンディング
        ADDITIVE      // 加算合成
    };
    void setBlendMode(BlendMode mode);
    void setAlpha(float alpha);  // 0.0-1.0
    
    // 統計情報取得
    double getProcessingTime() const;
    std::string getLastError() const;
    
private:
    // 内部メソッド
    void projectPoints(const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      std::vector<cv::Point2f>& projected_points);
    
    void computePerspectiveTransform(
        const std::vector<cv::Point2f>& src_points,
        const std::vector<cv::Point2f>& dst_points,
        cv::Mat& transform_matrix);
    
    void applyTexture(const cv::Mat& image,
                     const cv::Mat& transform_matrix,
                     const cv::Mat& ad_texture,
                     cv::Mat& output);
    
    bool validateInputs(const cv::Mat& image,
                       const cv::Mat& rvec,
                       const cv::Mat& tvec);
    
    // メンバ変数
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> backnet_corners_3d_;
    cv::Mat ad_texture_;
    BlendMode blend_mode_;
    float alpha_;
    double processing_time_;
    std::string last_error_;
    bool is_initialized_;
};

} // namespace Rendering
} // namespace VirtualAd
```

---

## 実装詳細

### 1. 3D平面定義

バックネットを3D空間の4隅で定義：

```cpp
// バックネット3D座標（例: 10m × 5m）
std::vector<cv::Point3f> backnet_corners = {
    cv::Point3f(-5.0f, 2.5f, 10.0f),  // 左上
    cv::Point3f( 5.0f, 2.5f, 10.0f),  // 右上
    cv::Point3f( 5.0f, -2.5f, 10.0f), // 右下
    cv::Point3f(-5.0f, -2.5f, 10.0f)  // 左下
};
```

### 2. 透視変換計算

**処理フロー**:
1. `cv::projectPoints()`: 3D→2D投影
2. `cv::getPerspectiveTransform()`: 透視変換行列計算
3. `cv::warpPerspective()`: テクスチャ変換

**コード例**:

```cpp
// 3D→2D投影
std::vector<cv::Point2f> projected_points;
cv::projectPoints(backnet_corners_3d_, rvec, tvec,
                 camera_matrix_, dist_coeffs_,
                 projected_points);

// 透視変換行列
std::vector<cv::Point2f> src_points = {
    cv::Point2f(0, 0),
    cv::Point2f(ad_texture_.cols, 0),
    cv::Point2f(ad_texture_.cols, ad_texture_.rows),
    cv::Point2f(0, ad_texture_.rows)
};
cv::Mat transform = cv::getPerspectiveTransform(src_points, projected_points);

// テクスチャ変換
cv::Mat warped_ad;
cv::warpPerspective(ad_texture_, warped_ad, transform,
                   image.size(), cv::INTER_LINEAR);
```

### 3. ブレンディング

**REPLACEモード**: 直接置き換え

```cpp
output = image.clone();
warped_ad.copyTo(output, mask);
```

**ALPHA_BLENDモード**: 透明度合成

```cpp
cv::addWeighted(image, 1.0 - alpha_, warped_ad, alpha_, 0.0, output);
```

**ADDITIVEモード**: 加算合成

```cpp
cv::add(image, warped_ad, output, mask);
```

---

## テストプログラム設計

### test_ad_renderer.cpp

**テスト項目**:
1. **初期化テスト**: カメラ行列設定
2. **基本レンダリングテスト**: 固定ポーズでレンダリング
3. **ブレンディングモードテスト**: 3種類のモード確認
4. **パフォーマンステスト**: 100イテレーション、目標<2ms/frame

**テストヘルパー**:
- `createDummyCameraMatrix()`: テスト用カメラ行列生成
- `createDummyPose()`: テスト用カメラポーズ生成
- `createDummyAdTexture()`: テスト用広告画像生成

---

## パフォーマンス目標

- **レンダリング時間**: <2ms/frame（1920x1080）
- **メモリ使用量**: 広告テクスチャ分のみ（数MB）
- **CPU使用率**: 1コア未満

---

## 依存関係

### 既存コンポーネント
- **Tracking**: PnPSolver（カメラポーズ取得）
- **Inference**: CameraPoseRefiner（AI補正済みポーズ）

### 外部ライブラリ
- **OpenCV**: 透視変換、画像処理

---

## 実装順序

### Step 5-1: AdRenderer.h作成（15分、約150行）
- クラス宣言
- メソッド定義
- enum定義

### Step 5-2: AdRenderer.cpp実装（45分、約300行を3パートに分割）

**Part 1**（約100行）:
- コンストラクタ・デストラクタ
- initialize()
- setBacknetPlane()
- setAdTexture()
- getter/setter

**Part 2**（約100行）:
- projectPoints()
- computePerspectiveTransform()
- validateInputs()

**Part 3**（約100行）:
- render()
- applyTexture()
- ブレンディング処理

### Step 5-3: test_ad_renderer.cpp実装（60分、約300行を3パートに分割）

**Part 1**（約100行）:
- テストヘルパー関数
- テスト1: 初期化テスト

**Part 2**（約100行）:
- テスト2: 基本レンダリング
- テスト3: ブレンディングモード

**Part 3**（約100行）:
- テスト4: パフォーマンステスト
- main関数

### Step 5-4: CMakeLists.txt更新（5分）
- Renderingライブラリ追加
- TestAdRendererターゲット追加

---

## 完了基準

- [x] AdRenderer.h/cpp実装完了
- [x] test_ad_renderer.cpp実装完了
- [x] すべてのテストPASS
- [x] パフォーマンス目標達成（<2ms/frame）
- [x] CMakeLists.txt更新完了
- [x] ビルド成功

---

## 次のフェーズ

Phase 5完了後、Phase 6（統合・最適化）に進みます：
- すべてのコンポーネント統合
- エンドツーエンドパイプライン
- 60fps達成確認
- 最終テスト

---

**作成日**: 2025/10/20
**Phase 5目標完了日**: 2025/10/22
