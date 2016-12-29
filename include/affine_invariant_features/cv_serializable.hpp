#ifndef AFFINE_INVARIANT_FEATURES_CV_SERIALIZABLE
#define AFFINE_INVARIANT_FEATURES_CV_SERIALIZABLE

#include <string>

#include <opencv2/core.hpp>

namespace affine_invariant_features {

struct CvSerializable {
  CvSerializable() {}

  virtual ~CvSerializable() {}

  virtual void read(const cv::FileNode &) = 0;

  virtual void write(cv::FileStorage &) const = 0;

  virtual std::string getDefaultName() const = 0;
};

static inline void read(const cv::FileNode &fn, CvSerializable &val, const CvSerializable &) {
  val.read(fn);
}

static inline void write(cv::FileStorage &fs, const std::string &, const CvSerializable &val) {
  val.write(fs);
}
}

#endif