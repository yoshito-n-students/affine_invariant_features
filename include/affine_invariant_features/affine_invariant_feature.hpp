#ifndef AFFINE_INVARIANT_FEATURES_AFFINE_INVARIANT_FEATURE
#define AFFINE_INVARIANT_FEATURES_AFFINE_INVARIANT_FEATURE

#include <algorithm>
#include <vector>

#include <affine_invariant_features/affine_invariant_feature_base.hpp>
#include <affine_invariant_features/parallel_tasks.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace affine_invariant_features {

//
// AffineInvariantFeature that samples features in various affine transformation space
//

class AffineInvariantFeature : public AffineInvariantFeatureBase {
private:
  // the private constructor. users must use create() to instantiate an AffineInvariantFeature
  AffineInvariantFeature(const cv::Ptr< cv::Feature2D > detector,
                         const cv::Ptr< cv::Feature2D > extractor)
      : AffineInvariantFeatureBase(detector, extractor) {
    // generate parameters for affine invariant sampling
    phi_params_.push_back(0.);
    tilt_params_.push_back(1.);
    for (int i = 1; i < 6; ++i) {
      const double tilt(std::pow(2., 0.5 * i));
      for (double phi = 0.; phi < 180.; phi += 72. / tilt) {
        phi_params_.push_back(phi);
        tilt_params_.push_back(tilt);
      }
    }
    ntasks_ = phi_params_.size();
  }

public:
  virtual ~AffineInvariantFeature() {}

  //
  // unique interfaces to instantiate an AffineInvariantFeature
  //

  static cv::Ptr< AffineInvariantFeature > create(const cv::Ptr< cv::Feature2D > feature) {
    return new AffineInvariantFeature(feature, feature);
  }

  static cv::Ptr< AffineInvariantFeature > create(const cv::Ptr< cv::Feature2D > detector,
                                                  const cv::Ptr< cv::Feature2D > extractor) {
    return new AffineInvariantFeature(detector, extractor);
  }

  //
  // overloaded functions from AffineInvariantFeatureBase or its base class
  //

  virtual void compute(cv::InputArray image, std::vector< cv::KeyPoint > &keypoints,
                       cv::OutputArray descriptors) {
    // extract the input
    const cv::Mat image_mat(image.getMat());

    // prepare outputs of following parallel processing
    std::vector< std::vector< cv::KeyPoint > > keypoints_array(ntasks_, keypoints);
    std::vector< cv::Mat > descriptors_array(ntasks_);

    // bind each parallel task and arguments
    ParallelTasks tasks(ntasks_);
    for (int i = 0; i < ntasks_; ++i) {
      tasks[i] = boost::bind(&AffineInvariantFeature::computeTask, this, boost::ref(image_mat),
                             boost::ref(keypoints_array[i]), boost::ref(descriptors_array[i]),
                             phi_params_[i], tilt_params_[i]);
    }

    // do parallel tasks
    cv::parallel_for_(cv::Range(0, ntasks_), tasks);

    // fill the final outputs
    extendKeypoints(keypoints_array, keypoints);
    extendDescriptors(descriptors_array, descriptors);
  }

  virtual void detect(cv::InputArray image, std::vector< cv::KeyPoint > &keypoints,
                      cv::InputArray mask = cv::noArray()) {
    // extract inputs
    const cv::Mat image_mat(image.getMat());
    const cv::Mat mask_mat(mask.getMat());

    // prepare an output of following parallel processing
    std::vector< std::vector< cv::KeyPoint > > keypoints_array(ntasks_);

    // bind each parallel task and arguments
    ParallelTasks tasks(ntasks_);
    for (std::size_t i = 0; i < ntasks_; ++i) {
      tasks[i] = boost::bind(&AffineInvariantFeature::detectTask, this, boost::ref(image_mat),
                             boost::ref(mask_mat), boost::ref(keypoints_array[i]), phi_params_[i],
                             tilt_params_[i]);
    }

    // do parallel tasks
    cv::parallel_for_(cv::Range(0, ntasks_), tasks);

    // fill the final output
    extendKeypoints(keypoints_array, keypoints);
  }

  // detect and compute affine invariant keypoints and descriptors
  virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                                std::vector< cv::KeyPoint > &keypoints, cv::OutputArray descriptors,
                                bool useProvidedKeypoints = false) {
    // just compute descriptors if the keypoints is provided
    if (useProvidedKeypoints) {
      compute(image, keypoints, descriptors);
      return;
    }

    // extract inputs
    const cv::Mat image_mat(image.getMat());
    const cv::Mat mask_mat(mask.getMat());

    // prepare outputs of following parallel processing
    std::vector< std::vector< cv::KeyPoint > > keypoints_array(ntasks_);
    std::vector< cv::Mat > descriptors_array(ntasks_);

    // bind each parallel task and arguments
    ParallelTasks tasks(ntasks_);
    for (std::size_t i = 0; i < ntasks_; ++i) {
      tasks[i] =
          boost::bind(&AffineInvariantFeature::detectAndComputeTask, this, boost::ref(image_mat),
                      boost::ref(mask_mat), boost::ref(keypoints_array[i]),
                      boost::ref(descriptors_array[i]), phi_params_[i], tilt_params_[i]);
    }

    // do parallel tasks
    cv::parallel_for_(cv::Range(0, ntasks_), tasks);

    // fill the final outputs
    extendKeypoints(keypoints_array, keypoints);
    extendDescriptors(descriptors_array, descriptors);
  }

  virtual cv::String getDefaultName() const { return "AffineInvariantFeature"; }

private:
  void computeTask(const cv::Mat &src_image, std::vector< cv::KeyPoint > &keypoints,
                   cv::Mat &descriptors, const double phi, const double tilt) const {
    // apply the affine transformation to the image on the basis of the given parameters
    cv::Mat image(src_image.clone());
    cv::Matx23f affine;
    warpImage(image, affine, phi, tilt);

    // apply the affine transformation to keypoints
    transformKeypoints(keypoints, affine);

    // extract descriptors on the skewed image and keypoints
    CV_Assert(extractor_);
    extractor_->compute(image, keypoints, descriptors);

    // invert keypoints
    cv::Matx23f invert_affine;
    cv::invertAffineTransform(affine, invert_affine);
    transformKeypoints(keypoints, invert_affine);
  }

  void detectTask(const cv::Mat &src_image, const cv::Mat &src_mask,
                  std::vector< cv::KeyPoint > &keypoints, const double phi,
                  const double tilt) const {
    // apply the affine transformation to the image on the basis of the given parameters
    cv::Mat image(src_image.clone());
    cv::Matx23f affine;
    warpImage(image, affine, phi, tilt);

    // apply the affine transformation to the mask
    cv::Mat mask(src_mask.empty() ? cv::Mat(src_image.size(), CV_8UC1, 255) : src_mask.clone());
    warpMask(mask, affine, image.size());

    // detect keypoints on the skewed image and mask
    CV_Assert(detector_);
    detector_->detect(image, keypoints, mask);

    // invert keypoints
    cv::Matx23f invert_affine;
    cv::invertAffineTransform(affine, invert_affine);
    transformKeypoints(keypoints, invert_affine);
  }

  void detectAndComputeTask(const cv::Mat &src_image, const cv::Mat &src_mask,
                            std::vector< cv::KeyPoint > &keypoints, cv::Mat &descriptors,
                            const double phi, const double tilt) {
    // apply the affine transformation to the image on the basis of the given parameters
    cv::Mat image(src_image.clone());
    cv::Matx23f affine;
    warpImage(image, affine, phi, tilt);

    // if keypoints are not provided, first apply the affine transformation to the mask
    cv::Mat mask(src_mask.empty() ? cv::Mat(src_image.size(), CV_8UC1, 255) : src_mask.clone());
    warpMask(mask, affine, image.size());

    // detect keypoints on the skewed image and mask
    // and extract descriptors on the image and keypoints
    if (detector_ == extractor_) {
      CV_Assert(detector_);
      detector_->detectAndCompute(image, mask, keypoints, descriptors, false);
    } else {
      CV_Assert(detector_);
      CV_Assert(extractor_);
      detector_->detect(image, keypoints, mask);
      extractor_->compute(image, keypoints, descriptors);
    }

    // invert the positions of the detected keypoints
    cv::Matx23f invert_affine;
    cv::invertAffineTransform(affine, invert_affine);
    transformKeypoints(keypoints, invert_affine);
  }

  static void warpImage(cv::Mat &image, cv::Matx23f &affine, const double phi, const double tilt) {
    // initiate output
    affine = cv::Matx23f::eye();

    if (phi != 0.) {
      // rotate the source frame
      affine = cv::getRotationMatrix2D(cv::Point2f(0., 0.), phi, 1.);
      cv::Rect tmp_rect;
      {
        std::vector< cv::Point2f > corners(4);
        corners[0] = cv::Point2f(0., 0.);
        corners[1] = cv::Point2f(image.cols, 0.);
        corners[2] = cv::Point2f(image.cols, image.rows);
        corners[3] = cv::Point2f(0., image.rows);
        std::vector< cv::Point2f > tmp_corners;
        cv::transform(corners, tmp_corners, affine);
        tmp_rect = cv::boundingRect(tmp_corners);
      }

      // cancel the offset of the rotated frame
      affine(0, 2) = -tmp_rect.x;
      affine(1, 2) = -tmp_rect.y;

      // apply the final transformation to the image
      cv::warpAffine(image, image, affine, tmp_rect.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }
    if (tilt != 1.) {
      // shrink the image in width
      cv::GaussianBlur(image, image, cv::Size(0, 0), 0.8 * std::sqrt(tilt * tilt - 1.), 0.01);
      cv::resize(image, image, cv::Size(0, 0), 1. / tilt, 1., cv::INTER_NEAREST);
      affine(0, 0) /= tilt;
      affine(0, 1) /= tilt;
      affine(0, 2) /= tilt;
    }
  }

  static void warpMask(cv::Mat &mask, const cv::Matx23f &affine, const cv::Size size) {
    if (affine == cv::Matx23f::eye()) {
      return;
    }
    cv::warpAffine(mask, mask, affine, size, cv::INTER_NEAREST);
  }

  static void transformKeypoints(std::vector< cv::KeyPoint > &keypoints,
                                 const cv::Matx23f &affine) {
    if (affine == cv::Matx23f::eye()) {
      return;
    }
    for (std::vector< cv::KeyPoint >::iterator keypoint = keypoints.begin();
         keypoint != keypoints.end(); ++keypoint) {
      // convert cv::Point2f to cv::Mat (1x1,2ch) without copying data.
      // this is required because cv::transform does not accept cv::Point2f.
      cv::Mat pt(cv::Mat(keypoint->pt, false).reshape(2));
      cv::transform(pt, pt, affine);
    }
  }

  static void extendKeypoints(const std::vector< std::vector< cv::KeyPoint > > &src,
                              std::vector< cv::KeyPoint > &dst) {
    dst.clear();
    for (std::size_t i = 0; i < src.size(); ++i) {
      dst.insert(dst.end(), src[i].begin(), src[i].end());
    }
  }

  void extendDescriptors(const std::vector< cv::Mat > &src, cv::OutputArray dst) const {
    // create the output array
    int nrows(0);
    for (std::size_t i = 0; i < src.size(); ++i) {
      nrows += src[i].rows;
    }
    dst.create(nrows, descriptorSize(), descriptorType());

    // fill the output array
    cv::Mat dst_mat(dst.getMat());
    int rows(0);
    for (std::size_t i = 0; i < src.size(); ++i) {
      src[i].copyTo(dst_mat.rowRange(rows, rows + src[i].rows));
      rows += src[i].rows;
    }
  }

private:
  std::vector< double > phi_params_;
  std::vector< double > tilt_params_;
  std::size_t ntasks_;
};
}

#endif