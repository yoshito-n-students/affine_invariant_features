#include <iostream>
#include <string>

#include <affine_invariant_features/feature_parameters.hpp>

#include <opencv2/core.hpp>

int main(int argc, char *argv[]) {

  namespace aif = affine_invariant_features;

  const cv::CommandLineParser args(
      argc, argv, "{ help | | }"
                  "{ @type | <none> | SIFTParameters, BRISKParameters, AKAZEParameters, ... }"
                  "{ @file | <none> | output file }");

  if (args.has("help")) {
    args.printMessage();
    return 0;
  }

  const std::string type(args.get< std::string >("@type"));
  const std::string path(args.get< std::string >("@file"));
  if (!args.check()) {
    args.printErrors();
    return 1;
  }

  const cv::Ptr< const aif::FeatureParameters > params(aif::createFeatureParameters(type));
  if (!params) {
    std::cerr << "Could not create a parameter set whose type is " << type << std::endl;
    return 1;
  }

  cv::FileStorage file(path, cv::FileStorage::WRITE);
  if (!file.isOpened()) {
    std::cerr << "Could not open or create " << path << std::endl;
    return 1;
  }

  file << params->getDefaultName() << *params;
  std::cout << "Wrote a parameter set whose type is " << type << " to " << path << std::endl;

  return 0;
}
