#include <iostream>
#include <string>

#include <affine_invariant_features/target.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[]) {
  namespace aif = affine_invariant_features;

  const cv::CommandLineParser args(
      argc, argv, "{ help | | }"
                  "{ @image | <none> | absolute, or relative to the current path or a ROS package }"
                  "{ @file | <none> | output file describing the image }"
                  "{ @package | | optional path of a ROS package where the image is }");

  if (args.has("help")) {
    args.printMessage();
    return 0;
  }

  const std::string image_path(args.get< std::string >("@image"));
  const std::string file_path(args.get< std::string >("@file"));
  const std::string package_name(args.get< std::string >("@package"));
  if (!args.check()) {
    args.printErrors();
    return 1;
  }

  const std::string resolved_path(aif::TargetDescription::resolvePath(package_name, image_path));
  const cv::Mat image(cv::imread(resolved_path));
  if (image.empty()) {
    std::cerr << "No image file at " << resolved_path << std::endl;
    return 1;
  }

  aif::TargetDescription target;
  target.package = package_name;
  target.path = image_path;
  target.md5 = aif::TargetDescription::generateMD5(resolved_path);
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
  std::cout << "Wrote a description of " << resolved_path << " to " << file_path << std::endl;

  return 0;
}
