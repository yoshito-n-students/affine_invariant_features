#include <iostream>
#include <string>

#include <affine_invariant_features/feature_parameters.hpp>
#include <affine_invariant_features/target.hpp>
#include <affine_invariant_features/results.hpp>
#include <affine_invariant_features/result_matcher.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace aif = affine_invariant_features;

bool readEverything(const std::string &path, aif::TargetData &target_data, aif::Results &results) {
  const cv::FileStorage file(path, cv::FileStorage::READ);

  aif::TargetDescription target_desc;
  file[target_desc.getDefaultName()] >> target_desc;
  target_data = target_desc.toData();
  if (target_data.image.empty()) {
    std::cerr << "Could not load an image described in " << path << std::endl;
    return false;
  }

  file[results.getDefaultName()] >> results;
  if (results.keypoints.empty() || results.descriptors.empty()) {
    std::cerr << "Could not load extracted features from " << path << std::endl;
    return false;
  }

  return true;
}

cv::Mat shadeImage(const cv::Mat &src, const cv::Mat &mask) {
  cv::Mat dst;
  if (mask.empty()) {
    dst = src.clone();
  } else {
    dst = src / 4;
    src.copyTo(dst, mask);
  }
  return dst;
}

int main(int argc, char *argv[]) {

  if (argc != 3) {
    std::cout << "Usage: match_features <feature_file1> <feature_file2>" << std::endl;
    std::cout << "Note: use extract_features to generate feature files" << std::endl;
    return 0;
  }

  const std::string feature_path1(argv[1]);
  const std::string feature_path2(argv[2]);

  aif::TargetData target1;
  aif::Results results1;
  if (!readEverything(feature_path1, target1, results1)) {
    return 1;
  }
  std::cout << "loaded " << results1.keypoints.size() << " feature points from " << feature_path1
            << std::endl;

  aif::TargetData target2;
  aif::Results results2;
  if (!readEverything(feature_path2, target2, results2)) {
    return 1;
  }
  std::cout << "loaded " << results2.keypoints.size() << " feature points from " << feature_path2
            << std::endl;

  aif::ResultMatcher matcher(results2);
  std::cout << "Matching feature points. This may take seconds." << std::endl;
  cv::Matx33f transform;
  std::vector< cv::DMatch > matches;
  matcher.match(results1, transform, matches);
  std::cout << "found " << matches.size() << " matches" << std::endl;

  const cv::Mat image1(shadeImage(target1.image, target1.mask));
  const cv::Mat image2(shadeImage(target2.image, target2.mask));
  cv::Mat image;
  cv::drawMatches(image1, results1.keypoints, image2, results2.keypoints, matches, image);

  std::cout << "Showing feature points and matches. Press any key to continue." << std::endl;
  cv::imshow("Matches", image);
  cv::waitKey(0);

  return 0;
}
