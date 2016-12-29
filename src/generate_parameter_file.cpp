#include <iostream>
#include <string>

#include <affine_invariant_features/feature_parameters.hpp>

#include <opencv2/core.hpp>

int main(int argc, char *argv[]) {
  namespace aif = affine_invariant_features;

  if (argc != 3) {
    std::cout << "Usage: generate_parameter_file <parameter_type> <parameter_file>" << std::endl;
    std::cout << "Available parameter types: AKAZEParameters, BRISKParameters, SIFTParameters, ..."
              << std::endl;
    return 0;
  }

  const std::string type(argv[1]);
  const std::string path(argv[2]);

  const cv::Ptr< const aif::FeatureParameters > params(aif::createFeatureParameters(type));
  if (!params) {
    std::cerr << "Could not create parameter set whose type is " << type << std::endl;
    return 1;
  }

  cv::FileStorage file(path, cv::FileStorage::WRITE);
  file << type << *params;
  std::cout << "Wrote parameter set whose type is " << type << " to " << path << std::endl;

  return 0;
}
