#ifndef AFFINE_INVARIANT_FEATURES_RESULT_MATCHER
#define AFFINE_INVARIANT_FEATURES_RESULT_MATCHER

#include <vector>

#include <affine_invariant_features/parallel_tasks.hpp>
#include <affine_invariant_features/results.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace affine_invariant_features {

class ResultMatcher {
public:
  ResultMatcher(const Results &reference) : reference_(reference) {
    switch (reference_.normType) {
    case cv::NORM_L2:
      matcher_ = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(4));
      break;
    case cv::NORM_HAMMING:
      matcher_ = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(6, 12, 1));
      break;
    }

    CV_Assert(matcher_);

    matcher_->add(reference_.descriptors);
    matcher_->train();
  }

  virtual ~ResultMatcher() {}

  const Results &getReference() const { return reference_; }

  void match(const Results &source, cv::Matx33f &transform,
             std::vector< cv::DMatch > &matches) const {
    // initiate outputs
    transform = cv::Matx33f::eye();
    matches.clear();

    // find the 1st & 2nd matches for each descriptor in the source
    std::vector< std::vector< cv::DMatch > > all_matches;
    matcher_->knnMatch(source.descriptors, all_matches, 2);

    // filter unique matches whose 1st is enough better than 2nd
    std::vector< cv::DMatch > unique_matches;
    for (std::vector< std::vector< cv::DMatch > >::const_iterator m = all_matches.begin();
         m != all_matches.end(); ++m) {
      if (m->size() < 2) {
        continue;
      }
      if ((*m)[0].distance > 0.75 * (*m)[1].distance) {
        continue;
      }
      unique_matches.push_back((*m)[0]);
    }
    if (unique_matches.size() < 4) {
      // cv::findHomography requires 4 or more point pairs
      return;
    }

    // further filter matches compatible to a registration
    std::vector< unsigned char > mask;
    {
      std::vector< cv::Point2f > source_points;
      std::vector< cv::Point2f > reference_points;
      for (std::vector< cv::DMatch >::const_iterator m = unique_matches.begin();
           m != unique_matches.end(); ++m) {
        source_points.push_back(source.keypoints[m->queryIdx].pt);
        reference_points.push_back(reference_.keypoints[m->trainIdx].pt);
      }
      transform = cv::findHomography(source_points, reference_points, cv::RANSAC, 5., mask);
    }

    // pack the final matches
    for (std::size_t i = 0; i < unique_matches.size(); ++i) {
      if (mask[i] == 0) {
        continue;
      }
      matches.push_back(unique_matches[i]);
    }
  }

  static void parallelMatch(const std::vector< ResultMatcher > &matchers, const Results &source,
                            std::vector< cv::Matx33f > &transforms,
                            std::vector< std::vector< cv::DMatch > > &matches_array) {
    // initiate output
    const int ntasks(matchers.size());
    transforms.resize(ntasks);
    matches_array.resize(ntasks);

    // populate tasks
    ParallelTasks tasks;
    for (int i = 0; i < ntasks; ++i) {
      tasks.push_back(boost::bind(&ResultMatcher::match, &matchers[i], boost::ref(source),
                                  boost::ref(transforms[i]), boost::ref(matches_array[i])));
    }

    // do paralell matching
    cv::parallel_for_(cv::Range(0, ntasks), tasks);
  }

private:
  const Results reference_;
  cv::Ptr< cv::DescriptorMatcher > matcher_;
};
}

#endif