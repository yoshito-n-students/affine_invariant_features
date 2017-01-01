#include <iostream>
#include <string>

#include <affine_invariant_features/target.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[]) {
  namespace aif = affine_invariant_features;

  const cv::CommandLineParser args(
      argc, argv, "{ help | | }"
                  "{ write-relative | false | write an image path with respect to the file path }"
                  "{ @target-image | <none> | }"
                  "{ @target-file | <none> | }");

  if (args.has("help")) {
    args.printMessage();
    return 0;
  }

  const bool write_relative(args.get< bool >("write-relative"));
  const std::string image_path(args.get< std::string >("@target-image"));
  const std::string file_path(args.get< std::string >("@target-file"));
  if (!args.check()) {
    args.printErrors();
    return 1;
  }

  const cv::Mat image(cv::imread(image_path));
  if (image.empty()) {
    std::cerr << "No image file at " << image_path << std::endl;
    return 1;
  }

  aif::TargetDescription target;
  if (write_relative) {
    // TODO: properly handle write_relative
    // target.imagePath = aif::TargetDescription::relativePath(image_path, file_path);
    target.imagePath = aif::TargetDescription::absolutePath(image_path);
  } else {
    target.imagePath = aif::TargetDescription::absolutePath(image_path);
  }
  target.imageMD5 = aif::TargetDescription::md5(image_path);
  target.contour.push_back(cv::Point(0, 0));
  target.contour.push_back(cv::Point(image.cols - 1, 0));
  target.contour.push_back(cv::Point(image.cols - 1, image.rows - 1));
  target.contour.push_back(cv::Point(0, image.rows - 1));

  cv::FileStorage file(file_path, cv::FileStorage::WRITE);
  if (!file.isOpened()) {
    std::cerr << "Could not open or create " << file_path << std::endl;
    return 1;
  }
  
  file << target.getDefaultName() << target;
  std::cout << "Wrote a description of " << image_path << " to " << file_path << std::endl;

  return 0;
}
