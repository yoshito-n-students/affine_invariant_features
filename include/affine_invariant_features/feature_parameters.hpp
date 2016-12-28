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

  virtual cv::Ptr< cv::DescriptorMatcher > createMatcher() const = 0;
};

//
// MSER
//

struct MSERParameters : public CvSerializable {
public:
  MSERParameters()
      : delta(5), minArea(60), maxArea(14400), maxVariation(0.25), minDiversity(2),
        maxEvolution(200), areaThreshold(1.01), minMargin(0.003), edgeBlurSize(5) {}

  virtual ~MSERParameters() {}

  virtual cv::Ptr< cv::Feature2D > createFeature() const {
    return cv::MSER::create(delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution,
                            areaThreshold, minMargin, edgeBlurSize);
  }

  virtual cv::Ptr< cv::DescriptorMatcher > createMatcher() const {
    return cv::Ptr< cv::DescriptorMatcher >();
  }

  virtual void read(const cv::FileNode &fn) {
    fn["delta"] >> delta;
    fn["minArea"] >> minArea;
    fn["maxArea"] >> maxArea;
    fn["maxVariation"] >> maxVariation;
    fn["minDiversity"] >> minDiversity;
    fn["maxEvolution"] >> maxEvolution;
    fn["areaThreshold"] >> areaThreshold;
    fn["minMargin"] >> minMargin;
    fn["edgeBlurSize"] >> edgeBlurSize;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "delta" << delta;
    fs << "minArea" << minArea;
    fs << "maxArea" << maxArea;
    fs << "maxVariation" << maxVariation;
    fs << "minDiversity" << minDiversity;
    fs << "maxEvolution" << maxEvolution;
    fs << "areaThreshold" << areaThreshold;
    fs << "minMargin" << minMargin;
    fs << "edgeBlurSize" << edgeBlurSize;
    fs << "}";
  }

public:
  int delta;
  int minArea;
  int maxArea;
  double maxVariation;
  double minDiversity;
  int maxEvolution;
  double areaThreshold;
  double minMargin;
  int edgeBlurSize;
};

//
// SIFT
//

struct SIFTParameters : public FeatureParameters {
public:
  SIFTParameters()
      : nfeatures(0), nOctaveLayers(3), contrastThreshold(0.04), edgeThreshold(10.), sigma(1.6) {}

  virtual ~SIFTParameters() {}

  virtual cv::Ptr< cv::Feature2D > createFeature() const {
    return cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold,
                                         sigma);
  }

  virtual cv::Ptr< cv::DescriptorMatcher > createMatcher() const {
    return cv::DescriptorMatcher::create("FlannBased");
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

public:
  int nfeatures;
  int nOctaveLayers;
  double contrastThreshold;
  double edgeThreshold;
  double sigma;
};
}

#endif