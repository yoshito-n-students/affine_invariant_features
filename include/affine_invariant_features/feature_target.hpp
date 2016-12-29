#ifndef AFFINE_INVARIANT_FEATURES_FEATURE_TARGET
#define AFFINE_INVARIANT_FEATURES_FEATURE_TARGET

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <affine_invariant_features/cv_serializable.hpp>

#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <openssl/md5.h>

namespace affine_invariant_features {

struct TargetData {
public:
  cv::Mat image;
  cv::Mat mask;
};

struct TargetDescription : public CvSerializable {
public:
  TargetDescription() {}

  virtual ~TargetDescription() {}

  TargetData toData(const bool check_md5 = false) const {
    TargetData data;

    if (check_md5) {
      const std::string actual_md5(md5(imagePath));
      if (imageMD5 != actual_md5) {
        return data;
      }
    }

    data.image = cv::imread(imagePath);
    if (data.image.empty()) {
      return data;
    }

    if (!contour.empty()) {
      data.mask = cv::Mat::zeros(data.image.size(), CV_8UC1);
      std::vector< std::vector< cv::Point2f > > points(1);
      points[0].insert(points[0].end(), contour.begin(), contour.end());
      cv::fillPoly(data.mask, points, 255);
    }

    return data;
  }

  virtual void read(const cv::FileNode &fn) {
    fn["imagePath"] >> imagePath;
    fn["imageMD5"] >> imageMD5;
    fn["contour"] >> contour;
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "{";
    fs << "imagePath" << imagePath;
    fs << "imageMD5" << imageMD5;
    fs << "contour" << contour;
    fs << "}";
  }

public:
  static std::string absolutePath(const std::string &path) {
    return boost::filesystem::absolute(path).string();
  }

  static std::string md5(const std::string &path) {
    // open the given path as a binary file
    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs) {
      return std::string();
    }

    // calculate the MD5 hash using openSSL library
    unsigned char md5[MD5_DIGEST_LENGTH];
    {
      MD5_CTX ctx;
      MD5_Init(&ctx);
      char buf[4096];
      while (ifs.read(buf, 4096) || ifs.gcount()) {
        MD5_Update(&ctx, buf, ifs.gcount());
      }
      MD5_Final(md5, &ctx);
    }
    // stringaze the MD5 hash
    std::ostringstream oss;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0') << static_cast< int >(md5[i]);
    }

    return oss.str();
  }

public:
  std::string imagePath;
  std::string imageMD5;
  std::vector< cv::Point > contour;
};
}

#endif