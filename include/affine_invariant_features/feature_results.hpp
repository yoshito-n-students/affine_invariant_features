#ifndef AFFINE_INVARIANT_FEATURES_FEATURE_RESULTS
#define AFFINE_INVARIANT_FEATURES_FEATURE_RESULTS

#include <vector>

#include <affine_invariant_features/cv_serializable.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace affine_invariant_features {

struct Results : public CvSerializable {
public:
  Results() {}

  virtual ~Results() {}

  virtual void read(const cv::FileNode &fn) {
    fn["keypoints"] >> keypoints;
    fn["descriptors"] >> descriptors;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "keypoints" << keypoints;
    fs << "descriptors" << descriptors;
    fs << "}";
  }

public:
  std::vector< cv::KeyPoint > keypoints;
  cv::Mat descriptors;
};
}

#endif