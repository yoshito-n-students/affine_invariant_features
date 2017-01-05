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
  AffineInvariantFeatureBase(const cv::Ptr< cv::Feature2D > detector,
                             const cv::Ptr< cv::Feature2D > extractor)
      : detector_(detector), extractor_(extractor) {}

public:
  virtual ~AffineInvariantFeatureBase() {}

  //
  // inherited functions from cv::Feature2D or its base class.
  // These just call corresponding one of the base feature.
  // detect(), compute(), detectAndCompute() and getDefaultName()
  // are overloaded in AffineInvariantFeature.
  //

  virtual int defaultNorm() const {
    CV_Assert(extractor_);
    return extractor_->defaultNorm();
  }

  virtual int descriptorSize() const {
    CV_Assert(extractor_);
    return extractor_->descriptorSize();
  }

  virtual int descriptorType() const {
    CV_Assert(extractor_);
    return extractor_->descriptorType();
  }

  virtual bool empty() const {
    if (detector_) {
      if (!detector_->empty()) {
        return false;
      }
    }
    if (extractor_) {
      if (!extractor_->empty()) {
        return false;
      }
    }
    return true;
  }

  virtual void read(const cv::FileNode &fn) {
    if (detector_) {
      detector_->read(fn);
    }
    if (extractor_ && extractor_ != detector_) {
      extractor_->read(fn);
    }
  }

  virtual void write(cv::FileStorage &fs) const {
    if (detector_) {
      detector_->write(fs);
    }
    if (extractor_ && extractor_ != detector_) {
      extractor_->write(fs);
    }
  }

  virtual void clear() {
    if (detector_) {
      detector_->clear();
    }
    if (extractor_ && extractor_ != detector_) {
      extractor_->clear();
    }
  }

  virtual void save(const cv::String &filename) const {
    if (detector_) {
      detector_->save(filename);
    }
    if (extractor_ && extractor_ != detector_) {
      extractor_->save(filename);
    }
  }

protected:
  const cv::Ptr< cv::Feature2D > detector_;
  const cv::Ptr< cv::Feature2D > extractor_;
};
}
#endif