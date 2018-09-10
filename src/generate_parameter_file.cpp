#include <iostream>
#include <string>
#include <vector>

#include <affine_invariant_features/feature_parameters.hpp>

#include <opencv2/core.hpp>

#include "aif_assert.hpp"

int main(int argc, char *argv[]) {

  namespace aif = affine_invariant_features;

  const cv::CommandLineParser args(
      argc, argv,
      "{ help | | }"
      "{ non-aif | | generate non affine invariant feature parameters }"
      "{ list | | list available type names of parameter sets }"
      "{ @type | <none> | type of first parameter set }"
      "{ @file | <none> | output file }"
      "{ @type2 | | type of second parameter set (optional) }");

  if (args.has("help")) {
    args.printMessage();
    return 0;
  }

  if (args.has("list")) {
    const std::vector< std::string > names(aif::getFeatureParameterNames());
    for (std::vector< std::string >::const_iterator name = names.begin(); name != names.end();
         ++name) {
      std::cout << *name << std::endl;
    }
    return 0;
  }

  const std::string type(args.get< std::string >("@type"));
  const std::string type2(args.get< std::string >("@type2"));
  const std::string path(args.get< std::string >("@file"));
  const bool non_aif(args.has("non-aif"));
  if (!args.check()) {
    args.printErrors();
    return 1;
  }

  aif::AIFParameters params;

  params.push_back(aif::createFeatureParameters(type));
  AIF_Assert(params.back(), "Could not create the first parameter set whose type is %s",
             type.c_str());

  if (!type2.empty()) {
    params.push_back(aif::createFeatureParameters(type2));
    AIF_Assert(params.back(), "Could not create the second parameter set whose type is %s",
               type2.c_str());
  }

  cv::FileStorage file(path, cv::FileStorage::WRITE);
  AIF_Assert(file.isOpened(), "Could not open or create %s", path.c_str());

  if (non_aif) {
    params[0]->save(file);
    std::cout << "Wrote a parameter set whose type is " << type << " to " << path << std::endl;
  } else {
    params.save(file);
    std::cout << "Wrote a parameter set whose type is " << params.getDefaultName() << " to " << path
              << std::endl;
  }

  return 0;
}
