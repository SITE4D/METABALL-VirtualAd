#include "FeatureDetector.h"
#include <stdexcept>
#include <iostream>

namespace VirtualAd {
namespace Tracking {

FeatureDetector::FeatureDetector(FeatureDetectorType type, int maxFeatures)
    : m_type(type)
    , m_maxFeatures(maxFeatures)
{
    if (maxFeatures <= 0) {
        throw std::invalid_argument("maxFeatures must be greater than 0");
    }
    initializeDetector();
}

void FeatureDetector::initializeDetector() {
    switch (m_type) {
        case FeatureDetectorType::ORB:
            // Create ORB detector with specified parameters
            // nfeatures: maximum number of features to retain
            // scaleFactor: pyramid decimation ratio (1.2f is good balance)
            // nlevels: number of pyramid levels (8 is standard)
            // edgeThreshold: size of border where features are not detected
            // firstLevel: level of pyramid to put source image to
            // WTA_K: number of points to produce each element of descriptor
            // scoreType: HARRIS_SCORE or FAST_SCORE
            // patchSize: size of patch used by oriented BRIEF descriptor
            // fastThreshold: threshold for FAST detector
            m_detector = cv::ORB::create(
                m_maxFeatures,  // nfeatures
                1.2f,           // scaleFactor
                8,              // nlevels
                31,             // edgeThreshold
                0,              // firstLevel
                2,              // WTA_K
                cv::ORB::HARRIS_SCORE,  // scoreType
                31,             // patchSize
                20              // fastThreshold
            );
            break;

        case FeatureDetectorType::AKAZE:
            // Create AKAZE detector with specified parameters
            // descriptor_type: type of descriptor (MLDB is fastest)
            // descriptor_size: size of descriptor (0 means full size)
            // descriptor_channels: number of channels in descriptor
            // threshold: detector response threshold
            // nOctaves: maximum octave evolution
            // nOctaveLayers: number of sublevels per octave
            // diffusivity: diffusivity type
            m_detector = cv::AKAZE::create(
                cv::AKAZE::DESCRIPTOR_MLDB,  // descriptor_type
                0,              // descriptor_size
                3,              // descriptor_channels
                0.001f,         // threshold
                4,              // nOctaves
                4,              // nOctaveLayers
                cv::KAZE::DIFF_PM_G2  // diffusivity
            );
            break;

        default:
            throw std::runtime_error("Unknown feature detector type");
    }

    if (!m_detector) {
        throw std::runtime_error("Failed to create feature detector");
    }
}

bool FeatureDetector::detectFeatures(const cv::Mat& image,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    cv::Mat& descriptors) {
    if (image.empty()) {
        std::cerr << "[FeatureDetector] Error: Input image is empty" << std::endl;
        return false;
    }

    if (!m_detector) {
        std::cerr << "[FeatureDetector] Error: Detector not initialized" << std::endl;
        return false;
    }

    // Convert to grayscale if needed
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 1) {
        grayImage = image;
    } else {
        std::cerr << "[FeatureDetector] Error: Unsupported image format (channels: " 
                  << image.channels() << ")" << std::endl;
        return false;
    }

    try {
        // Clear previous results
        keypoints.clear();
        descriptors.release();

        // Detect and compute
        m_detector->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors);

        // Limit to maxFeatures if needed (ORB already does this, but AKAZE may not)
        if (static_cast<int>(keypoints.size()) > m_maxFeatures) {
            // Sort by response (stronger features first)
            std::sort(keypoints.begin(), keypoints.end(),
                     [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                         return a.response > b.response;
                     });
            
            // Keep only top maxFeatures
            keypoints.resize(m_maxFeatures);
            descriptors = descriptors.rowRange(0, m_maxFeatures).clone();
        }

        return !keypoints.empty();
    }
    catch (const cv::Exception& e) {
        std::cerr << "[FeatureDetector] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "[FeatureDetector] Exception: " << e.what() << std::endl;
        return false;
    }
}

std::string FeatureDetector::getName() const {
    switch (m_type) {
        case FeatureDetectorType::ORB:
            return "ORB";
        case FeatureDetectorType::AKAZE:
            return "AKAZE";
        default:
            return "Unknown";
    }
}

void FeatureDetector::setMaxFeatures(int maxFeatures) {
    if (maxFeatures <= 0) {
        throw std::invalid_argument("maxFeatures must be greater than 0");
    }
    
    if (maxFeatures != m_maxFeatures) {
        m_maxFeatures = maxFeatures;
        // Reinitialize detector with new max features
        initializeDetector();
    }
}

} // namespace Tracking
} // namespace VirtualAd
