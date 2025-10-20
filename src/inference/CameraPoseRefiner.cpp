#include "CameraPoseRefiner.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace metaball {

CameraPoseRefiner::CameraPoseRefiner(std::shared_ptr<VirtualAd::Tracking::PnPSolver> pnp_solver)
    : pnp_solver_(pnp_solver)
    , ai_inference_(nullptr)
    , mode_(Mode::BLENDED)
    , blend_alpha_(0.5f)
    , last_pnp_error_(0.0)
    , last_processing_time_(0.0)
    , last_error_("")
{
}

CameraPoseRefiner::~CameraPoseRefiner() {
}

bool CameraPoseRefiner::loadModel(const std::string& model_path) {
    try {
        ai_inference_ = std::make_unique<ONNXInference>();
        
        if (!ai_inference_->loadModel(model_path)) {
            setError("Failed to load AI model: " + ai_inference_->getLastError());
            ai_inference_.reset();
            return false;
        }
        
        std::cout << "CameraPoseRefiner: AI model loaded successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        setError(std::string("Exception while loading model: ") + e.what());
        return false;
    }
}

bool CameraPoseRefiner::refinePose(const cv::Mat& image,
                                   const std::vector<cv::Point3f>& object_points,
                                   const std::vector<cv::Point2f>& image_points,
                                   VirtualAd::Tracking::CameraPose& pose,
                                   std::vector<uchar>& inlier_mask) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = false;
    
    switch (mode_) {
        case Mode::PNP_ONLY:
            success = estimateWithPnP(object_points, image_points, pose, inlier_mask);
            break;
            
        case Mode::AI_ONLY:
            if (!isModelLoaded()) {
                setError("AI model not loaded for AI_ONLY mode");
                return false;
            }
            success = estimateWithAI(image, pose);
            break;
            
        case Mode::BLENDED:
            if (!isModelLoaded()) {
                // AIモデル未ロードの場合はPnPのみ使用
                std::cerr << "Warning: AI model not loaded, falling back to PNP_ONLY mode" << std::endl;
                success = estimateWithPnP(object_points, image_points, pose, inlier_mask);
            } else {
                // PnPとAIの両方を実行してブレンド
                VirtualAd::Tracking::CameraPose pnp_pose, ai_pose;
                
                if (!estimateWithPnP(object_points, image_points, pnp_pose, inlier_mask)) {
                    setError("PnP estimation failed in BLENDED mode");
                    return false;
                }
                
                if (!estimateWithAI(image, ai_pose)) {
                    // AI失敗時はPnPのみ使用
                    std::cerr << "Warning: AI estimation failed, using PnP result" << std::endl;
                    pose = pnp_pose;
                    success = true;
                } else {
                    // ブレンディング
                    success = blendPoses(pnp_pose, ai_pose, pose);
                }
            }
            break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return success;
}

void CameraPoseRefiner::setBlendAlpha(float alpha) {
    blend_alpha_ = std::clamp(alpha, 0.0f, 1.0f);
}

bool CameraPoseRefiner::estimateWithPnP(const std::vector<cv::Point3f>& object_points,
                                        const std::vector<cv::Point2f>& image_points,
                                        VirtualAd::Tracking::CameraPose& pose,
                                        std::vector<uchar>& inlier_mask) {
    if (!pnp_solver_) {
        setError("PnP solver not initialized");
        return false;
    }
    
    if (object_points.size() < 4 || image_points.size() < 4) {
        setError("Insufficient points for PnP (need at least 4)");
        return false;
    }
    
    if (object_points.size() != image_points.size()) {
        setError("Point count mismatch");
        return false;
    }
    
    if (!pnp_solver_->solvePnP(object_points, image_points, pose, inlier_mask)) {
        setError("PnP solve failed");
        return false;
    }
    
    // 再投影誤差計算
    last_pnp_error_ = pnp_solver_->calculateReprojectionError(object_points, image_points, pose);
    
    return true;
}

bool CameraPoseRefiner::estimateWithAI(const cv::Mat& image, VirtualAd::Tracking::CameraPose& pose) {
    if (!ai_inference_ || !ai_inference_->isLoaded()) {
        setError("AI inference not initialized or model not loaded");
        return false;
    }
    
    if (image.empty()) {
        setError("Input image is empty");
        return false;
    }
    
    // AI推論実行
    std::vector<float> pose_vector;
    if (!ai_inference_->infer(image, pose_vector)) {
        setError("AI inference failed: " + ai_inference_->getLastError());
        return false;
    }
    
    // vector<float> → CameraPose変換
    if (pose_vector.size() != 6) {
        setError("Invalid AI output size (expected 6)");
        return false;
    }
    
    // rvec: [0, 1, 2]
    pose.rvec = (cv::Mat_<double>(3, 1) << 
                 pose_vector[0], pose_vector[1], pose_vector[2]);
    
    // tvec: [3, 4, 5]
    pose.tvec = (cv::Mat_<double>(3, 1) << 
                 pose_vector[3], pose_vector[4], pose_vector[5]);
    
    // 回転行列を更新
    pose.updateRotationMatrix();
    
    return true;
}

bool CameraPoseRefiner::blendPoses(const VirtualAd::Tracking::CameraPose& pnp_pose,
                                   const VirtualAd::Tracking::CameraPose& ai_pose,
                                   VirtualAd::Tracking::CameraPose& pose) {
    if (!pnp_pose.isValid() || !ai_pose.isValid()) {
        setError("Invalid input poses for blending");
        return false;
    }
    
    // 線形補間でブレンド
    // pose = (1 - alpha) * pnp_pose + alpha * ai_pose
    
    // rvec補間
    pose.rvec = cv::Mat::zeros(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        double pnp_val = pnp_pose.rvec.at<double>(i, 0);
        double ai_val = ai_pose.rvec.at<double>(i, 0);
        pose.rvec.at<double>(i, 0) = (1.0 - blend_alpha_) * pnp_val + blend_alpha_ * ai_val;
    }
    
    // tvec補間
    pose.tvec = cv::Mat::zeros(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        double pnp_val = pnp_pose.tvec.at<double>(i, 0);
        double ai_val = ai_pose.tvec.at<double>(i, 0);
        pose.tvec.at<double>(i, 0) = (1.0 - blend_alpha_) * pnp_val + blend_alpha_ * ai_val;
    }
    
    // 回転行列を更新
    pose.updateRotationMatrix();
    
    return true;
}

void CameraPoseRefiner::setError(const std::string& error) {
    last_error_ = error;
    std::cerr << "CameraPoseRefiner Error: " << error << std::endl;
}

} // namespace metaball
