/**
 * @file test_segmentation.cpp
 * @brief Test program for SegmentationInference
 * 
 * セグメンテーション推論のテストプログラム。
 * 画像を読み込み、セグメンテーションマスクを生成し、結果を表示・保存します。
 */

#include "SegmentationInference.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace VirtualAd::Keyer;

/**
 * @brief 使用方法を表示
 */
void printUsage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " <model_path> <image_path> [output_path]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  model_path   : Path to ONNX segmentation model (.onnx)" << std::endl;
    std::cout << "  image_path   : Path to input image" << std::endl;
    std::cout << "  output_path  : (Optional) Path to save output image" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " models/segmentation.onnx data/samples/test.jpg output.jpg" << std::endl;
}

/**
 * @brief メイン関数
 */
int main(int argc, char* argv[])
{
    // コマンドライン引数チェック
    if (argc < 3) {
        std::cerr << "ERROR: Insufficient arguments" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "";
    
    std::cout << "========================================" << std::endl;
    std::cout << "Segmentation Inference Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    if (!output_path.empty()) {
        std::cout << "Output: " << output_path << std::endl;
    }
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // 画像読み込み
    std::cout << "Loading image..." << std::endl;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "ERROR: Failed to load image: " << image_path << std::endl;
        return 1;
    }
    std::cout << "  Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << std::endl;
    
    // SegmentationInference初期化
    std::cout << "Initializing SegmentationInference..." << std::endl;
    SegmentationInference seg_inference(512);  // 入力サイズ512x512
    std::cout << std::endl;
    
    // モデルロード
    std::cout << "Loading model..." << std::endl;
    if (!seg_inference.loadModel(model_path)) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        std::cerr << "  " << seg_inference.getLastError() << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << std::endl;
    
    // 推論実行
    std::cout << "Running inference..." << std::endl;
    cv::Mat mask;
    if (!seg_inference.infer(image, mask)) {
        std::cerr << "ERROR: Inference failed" << std::endl;
        std::cerr << "  " << seg_inference.getLastError() << std::endl;
        return 1;
    }
    
    double inference_time = seg_inference.getInferenceTime();
    std::cout << "Inference completed successfully!" << std::endl;
    std::cout << "  Inference time: " << inference_time << " ms" << std::endl;
    std::cout << "  Mask size: " << mask.cols << "x" << mask.rows << std::endl;
    std::cout << std::endl;
    
    // マスク統計情報
    std::cout << "Mask statistics:" << std::endl;
    int class_counts[4] = {0, 0, 0, 0};
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            uchar class_id = mask.at<uchar>(y, x);
            if (class_id < 4) {
                class_counts[class_id]++;
            }
        }
    }
    
    int total_pixels = mask.rows * mask.cols;
    std::cout << "  Background: " << class_counts[0] << " pixels (" 
              << (100.0 * class_counts[0] / total_pixels) << "%)" << std::endl;
    std::cout << "  Player:     " << class_counts[1] << " pixels (" 
              << (100.0 * class_counts[1] / total_pixels) << "%)" << std::endl;
    std::cout << "  Umpire:     " << class_counts[2] << " pixels (" 
              << (100.0 * class_counts[2] / total_pixels) << "%)" << std::endl;
    std::cout << "  Backnet:    " << class_counts[3] << " pixels (" 
              << (100.0 * class_counts[3] / total_pixels) << "%)" << std::endl;
    std::cout << std::endl;
    
    return 0;
}
