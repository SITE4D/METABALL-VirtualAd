#!/usr/bin/env python3
"""
SAM (Segment Anything Model) Annotation Tool

SAMを使用してセグメンテーションアノテーションを半自動で作成します。
野球映像のフレームから選手、審判、バックネット、背景のマスクを生成します。

使用方法:
    python sam_annotation.py --input_dir data/samples --output_dir data/segmentation

依存関係:
    pip install segment-anything opencv-python numpy
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np

# SAMインポート（別途インストール必要）
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("WARNING: segment-anything not installed. Install with:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
    SAM_AVAILABLE = False


class SAMAnnotationTool:
    """SAMベースのアノテーションツール"""
    
    # クラス定義
    CLASS_NAMES = {
        0: 'background',
        1: 'player',
        2: 'umpire',
        3: 'backnet'
    }
    
    # 各クラスの色（BGR）
    CLASS_COLORS = {
        0: (128, 128, 128),  # 背景: グレー
        1: (0, 0, 255),      # 選手: 赤
        2: (0, 255, 255),    # 審判: 黄
        3: (255, 0, 0)       # バックネット: 青
    }
    
    def __init__(self, sam_checkpoint: str = "sam_vit_h_4b8939.pth", 
                 model_type: str = "vit_h"):
        """
        初期化
        
        Args:
            sam_checkpoint: SAMモデルチェックポイントパス
            model_type: SAMモデルタイプ（vit_h, vit_l, vit_b）
        """
        if not SAM_AVAILABLE:
            raise RuntimeError("segment-anything not installed")
        
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.predictor = None
        
        # 状態管理
        self.current_image = None
        self.current_image_path = None
        self.masks = {}  # クラスIDごとのマスク
        self.points = {}  # クラスIDごとのクリック座標
        self.current_class = 1  # デフォルト: 選手
        
    def load_model(self):
        """SAMモデルをロード"""
        if not os.path.exists(self.sam_checkpoint):
            print(f"ERROR: SAM checkpoint not found: {self.sam_checkpoint}")
            print("Download SAM checkpoint from:")
            print("  https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return False
        
        print(f"Loading SAM model: {self.model_type} from {self.sam_checkpoint}")
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device='cuda')  # GPU使用（CPUの場合は'cpu'）
        self.predictor = SamPredictor(sam)
        print("SAM model loaded successfully")
        return True
    
    def load_image(self, image_path: str) -> bool:
        """
        画像をロード
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            成功したらTrue
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Failed to load image: {image_path}")
            return False
        
        self.current_image = image
        self.current_image_path = image_path
        self.masks.clear()
        self.points.clear()
        
        # SAMに画像をセット
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        print(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        return True
    
    def add_point(self, x: int, y: int, class_id: int, is_positive: bool = True):
        """
        セグメンテーション用のポイントを追加
        
        Args:
            x, y: 座標
            class_id: クラスID
            is_positive: ポジティブサンプル（True）かネガティブサンプル（False）
        """
        if class_id not in self.points:
            self.points[class_id] = {'positive': [], 'negative': []}
        
        if is_positive:
            self.points[class_id]['positive'].append((x, y))
        else:
            self.points[class_id]['negative'].append((x, y))
        
        # マスクを更新
        self._update_mask(class_id)
    
    def _update_mask(self, class_id: int):
        """
        指定クラスのマスクを更新
        
        Args:
            class_id: クラスID
        """
        if class_id not in self.points:
            return
        
        points_data = self.points[class_id]
        if not points_data['positive']:
            return
        
        # SAMに渡すポイント配列を作成
        point_coords = np.array(points_data['positive'] + points_data['negative'])
        point_labels = np.array(
            [1] * len(points_data['positive']) + 
            [0] * len(points_data['negative'])
        )
        
        # SAMで予測
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        
        # 最良のマスクを保存
        self.masks[class_id] = masks[0]
        
        print(f"Updated mask for class {class_id} ({self.CLASS_NAMES[class_id]})")
    
    def get_combined_mask(self) -> np.ndarray:
        """
        すべてのクラスマスクを結合
        
        Returns:
            結合マスク（各ピクセルにクラスID）
        """
        if self.current_image is None:
            return None
        
        height, width = self.current_image.shape[:2]
        combined = np.zeros((height, width), dtype=np.uint8)
        
        # クラスIDの昇順で結合（後のクラスが優先）
        for class_id in sorted(self.masks.keys()):
            mask = self.masks[class_id]
            combined[mask > 0] = class_id
        
        return combined
    
    def visualize(self) -> np.ndarray:
        """
        アノテーション結果を可視化
        
        Returns:
            可視化画像
        """
        if self.current_image is None:
            return None
        
        vis = self.current_image.copy()
        combined_mask = self.get_combined_mask()
        
        # マスクをオーバーレイ
        if combined_mask is not None:
            for class_id, color in self.CLASS_COLORS.items():
                if class_id == 0:  # 背景はスキップ
                    continue
                mask_region = (combined_mask == class_id)
                if mask_region.any():
                    overlay = vis.copy()
                    overlay[mask_region] = color
                    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # ポイントを描画
        for class_id, points_data in self.points.items():
            color = self.CLASS_COLORS[class_id]
            # ポジティブポイント
            for x, y in points_data['positive']:
                cv2.circle(vis, (x, y), 5, color, -1)
                cv2.circle(vis, (x, y), 6, (255, 255, 255), 2)
            # ネガティブポイント
            for x, y in points_data['negative']:
                cv2.circle(vis, (x, y), 5, (0, 0, 0), -1)
                cv2.circle(vis, (x, y), 6, color, 2)
        
        return vis
    
    def save_annotation(self, output_dir: str):
        """
        アノテーションを保存（COCO format）
        
        Args:
            output_dir: 出力ディレクトリ
        """
        if self.current_image is None:
            print("ERROR: No image loaded")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ファイル名取得
        image_name = Path(self.current_image_path).stem
        
        # マスク画像保存
        combined_mask = self.get_combined_mask()
        if combined_mask is not None:
            mask_path = output_path / f"{image_name}_mask.png"
            cv2.imwrite(str(mask_path), combined_mask)
            print(f"Saved mask: {mask_path}")
        
        # メタデータ保存（COCO format）
        annotation = {
            'image': {
                'file_name': Path(self.current_image_path).name,
                'height': self.current_image.shape[0],
                'width': self.current_image.shape[1]
            },
            'annotations': []
        }
        
        for class_id, mask in self.masks.items():
            # マスクから輪郭抽出
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # 面積が小さいものは除外
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                # セグメンテーション座標
                segmentation = contour.flatten().tolist()
                
                # バウンディングボックス
                x, y, w, h = cv2.boundingRect(contour)
                
                annotation['annotations'].append({
                    'category_id': int(class_id),
                    'category_name': self.CLASS_NAMES[class_id],
                    'segmentation': [segmentation],
                    'area': float(area),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
        
        # JSON保存
        json_path = output_path / f"{image_name}_annotation.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        print(f"Saved annotation: {json_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SAM Annotation Tool for Baseball Segmentation')
    parser.add_argument('--sam_checkpoint', type=str, 
                       default='models/sam_vit_h_4b8939.pth',
                       help='SAM model checkpoint path')
    parser.add_argument('--model_type', type=str, 
                       default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type')
    parser.add_argument('--input_dir', type=str, 
                       default='data/samples',
                       help='Input image directory')
    parser.add_argument('--output_dir', type=str, 
                       default='data/segmentation',
                       help='Output annotation directory')
    
    args = parser.parse_args()
    
    if not SAM_AVAILABLE:
        print("ERROR: segment-anything not installed")
        return
    
    # ツール初期化
    tool = SAMAnnotationTool(args.sam_checkpoint, args.model_type)
    
    # モデルロード
    if not tool.load_model():
        return
    
    # 入力画像リスト取得
    input_path = Path(args.input_dir)
    image_files = sorted(list(input_path.glob('*.jpg')) + 
                        list(input_path.glob('*.png')))
    
    if not image_files:
        print(f"ERROR: No images found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print("\n=== Instructions ===")
    print("1. Click on objects to add positive points")
    print("2. Right-click to add negative points")
    print("3. Press '1' for Player class")
    print("4. Press '2' for Umpire class")
    print("5. Press '3' for Backnet class")
    print("6. Press 's' to save annotation")
    print("7. Press 'n' for next image")
    print("8. Press 'p' for previous image")
    print("9. Press 'q' to quit")
    print("====================\n")
    
    # インタラクティブアノテーション
    current_idx = 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal tool
        if event == cv2.EVENT_LBUTTONDOWN:
            tool.add_point(x, y, tool.current_class, is_positive=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            tool.add_point(x, y, tool.current_class, is_positive=False)
    
    cv2.namedWindow('SAM Annotation Tool')
    cv2.setMouseCallback('SAM Annotation Tool', mouse_callback)
    
    tool.load_image(str(image_files[current_idx]))
    
    while True:
        # 可視化
        vis = tool.visualize()
        if vis is not None:
            # 情報表示
            info = f"Image {current_idx + 1}/{len(image_files)} | Class: {tool.CLASS_NAMES[tool.current_class]}"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
            cv2.imshow('SAM Annotation Tool', vis)
        
        # キー入力
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            tool.current_class = 1  # Player
            print(f"Current class: {tool.CLASS_NAMES[tool.current_class]}")
        elif key == ord('2'):
            tool.current_class = 2  # Umpire
            print(f"Current class: {tool.CLASS_NAMES[tool.current_class]}")
        elif key == ord('3'):
            tool.current_class = 3  # Backnet
            print(f"Current class: {tool.CLASS_NAMES[tool.current_class]}")
        elif key == ord('s'):
            tool.save_annotation(args.output_dir)
        elif key == ord('n'):
            # Next image
            if current_idx < len(image_files) - 1:
                current_idx += 1
                tool.load_image(str(image_files[current_idx]))
        elif key == ord('p'):
            # Previous image
            if current_idx > 0:
                current_idx -= 1
                tool.load_image(str(image_files[current_idx]))
    
    cv2.destroyAllWindows()
    print("Annotation tool closed")


if __name__ == '__main__':
    main()
