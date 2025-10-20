// AnnotationTool.cpp
// バックネット座標アノテーションツールの実装

#include "AnnotationTool.h"
#include <iostream>
#include <fstream>

namespace VirtualAd {
namespace Tools {

// 静的メンバの定義
const cv::Scalar AnnotationTool::POINT_COLOR(0, 255, 0);      // 緑
const cv::Scalar AnnotationTool::LINE_COLOR(0, 255, 255);     // 黄色
const cv::Scalar AnnotationTool::TEXT_COLOR(255, 255, 255);   // 白

AnnotationTool::AnnotationTool(const std::string& windowName)
    : windowName_(windowName)
    , currentFrameNumber_(-1)
    , annotationComplete_(false)
{
    cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName_, mouseCallback, this);
}

AnnotationTool::~AnnotationTool()
{
    cv::destroyWindow(windowName_);
}

BacknetCorners AnnotationTool::annotateFrame(const cv::Mat& frame, int frameNumber)
{
    currentFrame_ = frame.clone();
    displayFrame_ = currentFrame_.clone();
    currentFrameNumber_ = frameNumber;
    points_.clear();
    annotationComplete_ = false;

    std::cout << "\n[AnnotationTool] Frame " << frameNumber << std::endl;
    std::cout << "  Click 4 corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right" << std::endl;
    std::cout << "  Press 'u' to undo last point" << std::endl;
    std::cout << "  Press 'r' to reset" << std::endl;
    std::cout << "  Press 'Enter' when done (4 points)" << std::endl;
    std::cout << "  Press 'Esc' to skip this frame" << std::endl;

    draw();

    // マウス入力とキーボード入力を待つ
    while (!annotationComplete_) {
        int key = cv::waitKey(1);

        if (key == 27) { // ESC
            std::cout << "  Skipped" << std::endl;
            BacknetCorners result;
            result.frameNumber = frameNumber;
            result.valid = false;
            return result;
        }
        else if (key == 13 || key == 10) { // Enter
            if (points_.size() == 4) {
                annotationComplete_ = true;
            } else {
                std::cout << "  Need 4 points (current: " << points_.size() << ")" << std::endl;
            }
        }
        else if (key == 'u' || key == 'U') { // Undo
            undo();
        }
        else if (key == 'r' || key == 'R') { // Reset
            reset();
        }
    }

    // 結果を作成
    BacknetCorners result;
    result.frameNumber = frameNumber;
    result.valid = true;
    result.topLeft = points_[0];
    result.topRight = points_[1];
    result.bottomLeft = points_[2];
    result.bottomRight = points_[3];

    std::cout << "  Annotation completed" << std::endl;
    std::cout << "    TL: (" << result.topLeft.x << ", " << result.topLeft.y << ")" << std::endl;
    std::cout << "    TR: (" << result.topRight.x << ", " << result.topRight.y << ")" << std::endl;
    std::cout << "    BL: (" << result.bottomLeft.x << ", " << result.bottomLeft.y << ")" << std::endl;
    std::cout << "    BR: (" << result.bottomRight.x << ", " << result.bottomRight.y << ")" << std::endl;

    return result;
}

bool AnnotationTool::saveAnnotations(const std::string& outputPath, 
                                     const std::vector<BacknetCorners>& annotations)
{
    // 簡易的なJSON形式で保存
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "[AnnotationTool] Failed to open file: " << outputPath << std::endl;
        return false;
    }

    file << "{\n";
    file << "  \"annotations\": [\n";

    for (size_t i = 0; i < annotations.size(); ++i) {
        const auto& ann = annotations[i];
        if (!ann.valid) continue;

        file << "    {\n";
        file << "      \"frame\": " << ann.frameNumber << ",\n";
        file << "      \"topLeft\": [" << ann.topLeft.x << ", " << ann.topLeft.y << "],\n";
        file << "      \"topRight\": [" << ann.topRight.x << ", " << ann.topRight.y << "],\n";
        file << "      \"bottomLeft\": [" << ann.bottomLeft.x << ", " << ann.bottomLeft.y << "],\n";
        file << "      \"bottomRight\": [" << ann.bottomRight.x << ", " << ann.bottomRight.y << "]\n";
        file << "    }";
        
        if (i < annotations.size() - 1) {
            file << ",";
        }
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();

    std::cout << "[AnnotationTool] Saved " << annotations.size() << " annotations to " << outputPath << std::endl;

    return true;
}

std::vector<BacknetCorners> AnnotationTool::loadAnnotations(const std::string& inputPath)
{
    std::vector<BacknetCorners> annotations;

    // TODO: JSON読み込み実装（現在は未実装）
    std::cout << "[AnnotationTool] Load from " << inputPath << " (not implemented yet)" << std::endl;

    return annotations;
}

void AnnotationTool::reset()
{
    points_.clear();
    draw();
    std::cout << "  Reset" << std::endl;
}

void AnnotationTool::undo()
{
    if (!points_.empty()) {
        points_.pop_back();
        draw();
        std::cout << "  Undo (remaining: " << points_.size() << ")" << std::endl;
    }
}

void AnnotationTool::mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    AnnotationTool* tool = static_cast<AnnotationTool*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (tool->points_.size() < 4) {
            tool->points_.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            std::cout << "  Point " << tool->points_.size() << ": (" << x << ", " << y << ")" << std::endl;
            tool->draw();
        }
    }
}

void AnnotationTool::draw()
{
    displayFrame_ = currentFrame_.clone();

    // ポイントを描画
    for (size_t i = 0; i < points_.size(); ++i) {
        cv::circle(displayFrame_, points_[i], POINT_RADIUS, POINT_COLOR, -1);
        
        // ポイント番号を描画
        std::string label = std::to_string(i + 1);
        cv::putText(displayFrame_, label, 
                    cv::Point(static_cast<int>(points_[i].x + 10), static_cast<int>(points_[i].y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2);
    }

    // ラインを描画（4点揃ったら）
    if (points_.size() == 4) {
        cv::line(displayFrame_, points_[0], points_[1], LINE_COLOR, 2);
        cv::line(displayFrame_, points_[1], points_[3], LINE_COLOR, 2);
        cv::line(displayFrame_, points_[3], points_[2], LINE_COLOR, 2);
        cv::line(displayFrame_, points_[2], points_[0], LINE_COLOR, 2);
    }

    // ヘルプテキストを描画
    drawHelp();

    cv::imshow(windowName_, displayFrame_);
}

void AnnotationTool::drawHelp()
{
    int y = 30;
    const int lineHeight = 25;
    const cv::Scalar bgColor(0, 0, 0, 128);

    // 背景を描画
    cv::rectangle(displayFrame_, cv::Point(5, 5), cv::Point(450, 150), bgColor, -1);

    cv::putText(displayFrame_, "Frame: " + std::to_string(currentFrameNumber_), 
                cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
    y += lineHeight;

    cv::putText(displayFrame_, "Points: " + std::to_string(points_.size()) + "/4", 
                cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
    y += lineHeight;

    cv::putText(displayFrame_, "Click: Add point | U: Undo | R: Reset", 
                cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
    y += lineHeight;

    cv::putText(displayFrame_, "Enter: Confirm | Esc: Skip", 
                cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
}

} // namespace Tools
} // namespace VirtualAd
