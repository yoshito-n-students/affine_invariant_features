#include <iostream>
#include <string>

#include <affine_invariant_features/affine_invariant_feature.hpp>
#include <affine_invariant_features/feature_parameters.hpp>
#include <affine_invariant_features/feature_target.hpp>
#include <affine_invariant_features/feature_results.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

int main(int argc, char *argv[]) {
  namespace aif = affine_invariant_features;

  if (argc != 4) {
    std::cout << "Usage: extract_features <parameter_file> <target_file> <result_file>"
              << std::endl;
    std::cout << "Note: use generate_*_file to generate parameter or target files" << std::endl;
    return 0;
  }

  const std::string param_path(argv[1]);
  const std::string target_path(argv[2]);
  const std::string result_path(argv[3]);

  const cv::FileStorage param_file(param_path, cv::FileStorage::READ);
  const cv::Ptr< const aif::FeatureParameters > params(
      aif::readFeatureParameters(param_file.root()));
  if (!params) {
    std::cerr << "Could not load a parameter set from " << param_path << std::endl;
    return 1;
  }

  const cv::FileStorage target_file(target_path, cv::FileStorage::READ);
  aif::TargetDescription target_desc;
  target_file[target_desc.getDefaultName()] >> target_desc;
  if (target_desc.imagePath.empty()) {
    std::cerr << "Could not load an image path from " << target_path << std::endl;
    return 1;
  }

  const aif::TargetData target_data(target_desc.toData());
  if (target_data.image.empty()) {
    std::cerr << "Could not load a target image described in " << target_path << std::endl;
    return 1;
  }

  std::cout << "Extracting features. This may take seconds or minutes." << std::endl;
  const cv::Ptr< cv::Feature2D > feature(
      aif::AffineInvariantFeature::create(params->createFeature()));
  aif::Results results;
  feature->detectAndCompute(target_data.image, target_data.mask, results.keypoints,
                            results.descriptors);

  cv::FileStorage result_file(result_path, cv::FileStorage::WRITE);
  result_file << params->getDefaultName() << *params;
  result_file << target_desc.getDefaultName() << target_desc;
  result_file << results.getDefaultName() << results;

  std::cout << "Wrote context and results of feature extraction to " << result_path << std::endl;

  return 0;
}
