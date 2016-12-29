#ifndef AFFINE_INVARIANT_FEATURES_AFFINE_INVARIANT_FEATURE_BASE
#define AFFINE_INVARIANT_FEATURES_AFFINE_INVARIANT_FEATURE_BASE

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace affine_invariant_features {

//
// A base class of AffineInvariantFeature to hold a backend feature algorithm
//

class AffineInvariantFeatureBase : public cv::Feature2D {
protected:
  AffineInvariantFeatureBase(const cv::Ptr< cv::Feature2D > base_feature)
      : base_feature_(base_feature) {}

public:
  virtual ~AffineInvariantFeatureBase() {}

  //
  // inherited functions from cv::Feature2D or its base class.
  // These just call corresponding one of the base feature.
  // detectAndCompute() and  getDefaultName() are overloaded in AffineInvariantFeature.
  // detect() and compute() are not overloaded
  // because the default implementation calls detectAndCompute().
  //

  virtual int defaultNorm() const {
    CV_Assert(base_feature_);
    return base_feature_->defaultNorm();
  }

  virtual int descriptorSize() const {
    CV_Assert(base_feature_);
    return base_feature_->descriptorSize();
  }

  virtual int descriptorType() const {
    CV_Assert(base_feature_);
    return base_feature_->descriptorType();
  }

  virtual bool empty() const {
    CV_Assert(base_feature_);
    return base_feature_->empty();
  }

  virtual void read(const cv::FileNode &fn) {
    CV_Assert(base_feature_);
    base_feature_->read(fn);
  }

  virtual void write(cv::FileStorage &fs) const {
    CV_Assert(base_feature_);
    base_feature_->write(fs);
  }

  virtual void clear() {
    CV_Assert(base_feature_);
    base_feature_->clear();
  }

  virtual void save(const cv::String &filename) const {
    CV_Assert(base_feature_);
    base_feature_->save(filename);
  }

protected:
  const cv::Ptr< cv::Feature2D > base_feature_;
};
}
#endif