#ifndef AFFINE_INVARIANT_FEATURES_RESULTS
#define AFFINE_INVARIANT_FEATURES_RESULTS

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
    fn["normType"] >> normType;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "keypoints" << keypoints;
    fs << "descriptors" << descriptors;
    fs << "normType" << normType;
    fs << "}";
  }

  virtual std::string getDefaultName() const { return "Results"; }

public:
  std::vector< cv::KeyPoint > keypoints;
  cv::Mat descriptors;
  int normType; // cv::NormTypes
};
}

#endif