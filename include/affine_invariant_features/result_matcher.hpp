#ifndef AFFINE_INVARIANT_FEATURES_RESULT_MATCHER
#define AFFINE_INVARIANT_FEATURES_RESULT_MATCHER

#include <vector>

#include <affine_invariant_features/results.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace affine_invariant_features {

class ResultMatcher {
public:
  ResultMatcher(const Results &train) : train_(train) {
    switch (train_.normType) {
    case cv::NORM_L2:
      matcher_ = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(4));
      break;
    case cv::NORM_HAMMING:
      matcher_ = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(6, 12, 1));
      break;
    }

    CV_Assert(matcher_);

    matcher_->add(train_.descriptors);
    matcher_->train();
  }

  virtual ~ResultMatcher() {}

  void match(const Results &query, cv::Matx33f &query2train,
             std::vector< cv::DMatch > &matches) const {
    // initiate outputs
    query2train = cv::Matx33f::eye();
    matches.clear();

    // find the 1st & 2nd matches for each query result
    std::vector< std::vector< cv::DMatch > > all_matches;
    matcher_->knnMatch(query.descriptors, all_matches, 2);

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

    // further filter matches compatible to a query-to-train registration
    std::vector< unsigned char > mask;
    {
      std::vector< cv::Point2f > query_points;
      std::vector< cv::Point2f > train_points;
      for (std::vector< cv::DMatch >::const_iterator m = unique_matches.begin();
           m != unique_matches.end(); ++m) {
        query_points.push_back(query.keypoints[m->queryIdx].pt);
        train_points.push_back(train_.keypoints[m->trainIdx].pt);
      }
      query2train = cv::findHomography(query_points, train_points, cv::RANSAC, 5., mask);
    }

    // pack the final matches
    for (std::size_t i = 0; i < unique_matches.size(); ++i) {
      if (mask[i] == 0) {
        continue;
      }
      matches.push_back(unique_matches[i]);
    }
  }

private:
  const Results train_;
  cv::Ptr< cv::DescriptorMatcher > matcher_;
};
}

#endif