#include <iostream>
#include <string>

#include <affine_invariant_features/target.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[]) {
  namespace aif = affine_invariant_features;

  if (argc != 3) {
    std::cout << "Usage: generate_target_file <target_image> <target_file>" << std::endl;
    return 0;
  }

  const std::string image_path(argv[1]);
  const std::string file_path(argv[2]);

  const cv::Mat image(cv::imread(image_path));
  if (image.empty()) {
    std::cerr << "No image file at " << image_path << std::endl;
    return 1;
  }

  aif::TargetDescription target;
  target.imagePath = aif::TargetDescription::absolutePath(image_path);
  target.imageMD5 = aif::TargetDescription::md5(image_path);
  target.contour.push_back(cv::Point(0, 0));
  target.contour.push_back(cv::Point(image.cols - 1, 0));
  target.contour.push_back(cv::Point(image.cols - 1, image.rows - 1));
  target.contour.push_back(cv::Point(0, image.rows - 1));

  cv::FileStorage file(file_path, cv::FileStorage::WRITE);
  file << target.getDefaultName() << target;
  std::cout << "Wrote a description of " << image_path << " to " << file_path << std::endl;

  return 0;
}
