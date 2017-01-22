#include <iostream>
#include <string>

#include <affine_invariant_features/feature_parameters.hpp>

#include <opencv2/core.hpp>

#include "aif_assert.hpp"

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
  AIF_Assert(params, "Could not create a parameter set whose type is %s", type.c_str());

  cv::FileStorage file(path, cv::FileStorage::WRITE);
  AIF_Assert(file.isOpened(), "Could not open or create %s", path.c_str());

  params->save(file);
  std::cout << "Wrote a parameter set whose type is " << type << " to " << path << std::endl;

  return 0;
}
