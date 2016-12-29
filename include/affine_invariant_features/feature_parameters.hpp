#ifndef AFFINE_INVARIANT_FEATURES_FEATURE_PARAMETERS
#define AFFINE_INVARIANT_FEATURES_FEATURE_PARAMETERS

#include <affine_invariant_features/cv_serializable.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace affine_invariant_features {

//
// A base class for parameter sets of feature detectors and descriptor extractors
//

struct FeatureParameters : public CvSerializable {
public:
  FeatureParameters() {}

  virtual ~FeatureParameters() {}

  virtual cv::Ptr< cv::Feature2D > createFeature() const = 0;
};

//
// SIFT
//

struct SIFTParameters : public FeatureParameters {
public:
  // opencv does not provide interfaces to access default SIFT parameters.
  // thus values below are copied from online reference.
  SIFTParameters()
      : nfeatures(0), nOctaveLayers(3), contrastThreshold(0.04), edgeThreshold(10.), sigma(1.6) {}

  virtual ~SIFTParameters() {}

  virtual cv::Ptr< cv::Feature2D > createFeature() const {
    return cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold,
                                         sigma);
  }

  virtual void read(const cv::FileNode &fn) {
    fn["nfeatures"] >> nfeatures;
    fn["nOctaveLayers"] >> nOctaveLayers;
    fn["contrastThreshold"] >> contrastThreshold;
    fn["edgeThreshold"] >> edgeThreshold;
    fn["sigma"] >> sigma;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "nfeatures" << nfeatures;
    fs << "nOctaveLayers" << nOctaveLayers;
    fs << "contrastThreshold" << contrastThreshold;
    fs << "edgeThreshold" << edgeThreshold;
    fs << "sigma" << sigma;
    fs << "}";
  }

  virtual std::string getDefaultName() const { return "SIFTParameters"; }

public:
  int nfeatures;
  int nOctaveLayers;
  double contrastThreshold;
  double edgeThreshold;
  double sigma;
};

//
// AKAZE
//

struct AKAZEParameters : public FeatureParameters {
public:
  AKAZEParameters()
      : descriptorType(defaultAKAZE().getDescriptorType()),
        descriptorSize(defaultAKAZE().getDescriptorSize()),
        descriptorChannels(defaultAKAZE().getDescriptorChannels()),
        threshold(defaultAKAZE().getThreshold()), nOctaves(defaultAKAZE().getNOctaves()),
        nOctaveLayers(defaultAKAZE().getNOctaveLayers()),
        diffusivity(defaultAKAZE().getDiffusivity()) {}

  virtual ~AKAZEParameters() {}

  virtual cv::Ptr< cv::Feature2D > createFeature() const {
    return cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold,
                             nOctaves, nOctaveLayers, diffusivity);
  }

  virtual void read(const cv::FileNode &fn) {
    fn["descriptorType"] >> descriptorType;
    fn["descriptorSize"] >> descriptorSize;
    fn["descriptorChannels"] >> descriptorChannels;
    fn["threshold"] >> threshold;
    fn["nOctaves"] >> nOctaves;
    fn["nOctaveLayers"] >> nOctaveLayers;
    fn["diffusivity"] >> diffusivity;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "descriptorType" << descriptorType;
    fs << "descriptorSize" << descriptorSize;
    fs << "descriptorChannels" << descriptorChannels;
    fs << "threshold" << threshold;
    fs << "nOctaves" << nOctaves;
    fs << "nOctaveLayers" << nOctaveLayers;
    fs << "diffusivity" << diffusivity;
    fs << "}";
  }

  virtual std::string getDefaultName() const { return "AKAZEParameters"; }

protected:
  static const cv::AKAZE &defaultAKAZE() {
    static cv::Ptr< cv::AKAZE > default_akaze(cv::AKAZE::create());
    CV_Assert(default_akaze);
    return *default_akaze;
  }

public:
  int descriptorType;
  int descriptorSize;
  int descriptorChannels;
  float threshold;
  int nOctaves;
  int nOctaveLayers;
  int diffusivity;
};

//
// BRISK
//

struct BRISKParameters : public FeatureParameters {
public:
  // Note: like SIFT, no interface to access default BRISK parameters
  BRISKParameters() : threshold(30), nOctaves(3), patternScale(1.0f) {}

  virtual ~BRISKParameters() {}

  virtual cv::Ptr< cv::Feature2D > createFeature() const {
    return cv::BRISK::create(threshold, nOctaves, patternScale);
  }

  virtual void read(const cv::FileNode &fn) {
    fn["threshold"] >> threshold;
    fn["nOctaves"] >> nOctaves;
    fn["patternScale"] >> patternScale;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "threshold" << threshold;
    fs << "nOctaves" << nOctaves;
    fs << "patternScale" << patternScale;
    fs << "}";
  }

  virtual std::string getDefaultName() const { return "BRISKParameters"; }

public:
  int threshold;
  int nOctaves;
  float patternScale;
};

//
// Utility functions to create or read variants of FeatureParameters
//

#define AIF_RETURN_IF_CREATE(type)                                                                 \
  {                                                                                                \
    const cv::Ptr< type > params(new type());                                                      \
    if (type_name == params->getDefaultName()) {                                                   \
      return params;                                                                               \
    }                                                                                              \
  }

static inline cv::Ptr< FeatureParameters > createFeatureParameters(const std::string &type_name) {
  AIF_RETURN_IF_CREATE(AKAZEParameters);
  AIF_RETURN_IF_CREATE(BRISKParameters);
  AIF_RETURN_IF_CREATE(SIFTParameters);
  return cv::Ptr< FeatureParameters >();
}

#define AIF_RETURN_IF_READ(type)                                                                   \
  {                                                                                                \
    const cv::Ptr< type > params(new type());                                                      \
    const cv::FileNode param_node(fn[params->getDefaultName()]);                                   \
    if (!param_node.empty()) {                                                                     \
      param_node >> *params;                                                                       \
      return params;                                                                               \
    }                                                                                              \
  }

static inline cv::Ptr< FeatureParameters > readFeatureParameters(const cv::FileNode &fn) {
  AIF_RETURN_IF_READ(AKAZEParameters);
  AIF_RETURN_IF_READ(BRISKParameters);
  AIF_RETURN_IF_READ(SIFTParameters);
  return cv::Ptr< FeatureParameters >();
}
}

#endif