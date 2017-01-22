#ifndef AFFINE_INVARIANT_FEATURES_TARGET
#define AFFINE_INVARIANT_FEATURES_TARGET

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ros/package.h>

#include <affine_invariant_features/cv_serializable.hpp>

#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <openssl/md5.h>

namespace affine_invariant_features {

struct TargetDescription : public CvSerializable {
public:
  TargetDescription() {}

  virtual ~TargetDescription() {}

  virtual void read(const cv::FileNode &fn) {
    fn["package"] >> package;
    fn["path"] >> path;
    fn["md5"] >> md5;
    const cv::FileNode contour_node(fn["contour"]);
    const std::size_t contour_size(contour_node.isSeq() ? contour_node.size() : 0);
    contour.resize(contour_size);
    for (std::size_t i = 0; i < contour_size; ++i) {
      contour_node[i] >> contour[i];
    }
  }

  virtual void write(cv::FileStorage &fs) const {
    fs << "package" << package;
    fs << "path" << path;
    fs << "md5" << md5;
    fs << "contour";
    fs << "[:";
    for (std::vector< cv::Point >::const_iterator point = contour.begin(); point != contour.end();
         ++point) {
      fs << *point;
    }
    fs << "]";
  }

  virtual std::string getDefaultName() const { return "TargetDescription"; }

public:
  static std::string resolvePath(const std::string &package, const std::string &path) {
    namespace bf = boost::filesystem;
    namespace rp = ros::package;

    const bf::path root_path(package.empty() ? std::string() : rp::getPath(package));
    const bf::path leaf_path(path);
    if (root_path.empty() || leaf_path.empty() || leaf_path.is_absolute()) {
      return leaf_path.string();
    }
    return (root_path / leaf_path).string();
  }

  static std::string generateMD5(const std::string &path) {
    // open the file path as a binary file
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
  std::string package;
  std::string path;
  std::string md5;
  std::vector< cv::Point > contour;
};

struct TargetData : public CvSerializable {
public:
  TargetData() {}

  virtual ~TargetData() {}

  virtual void read(const cv::FileNode &fn) {
    TargetDescription desc;
    desc.read(fn);

    const cv::Ptr< const TargetData > data(retrieve(desc));
    *this = data ? *data : TargetData();
  }

  virtual void write(cv::FileStorage &fs) const { CV_Error(cv::Error::StsNotImplemented, ""); }

  virtual std::string getDefaultName() const { return "TargetData"; }

public:
  static cv::Ptr< TargetData > retrieve(const TargetDescription &desc,
                                        const bool check_md5 = false) {
    const std::string path(TargetDescription::resolvePath(desc.package, desc.path));
    if (path.empty()) {
      return cv::Ptr< TargetData >();
    }

    if (check_md5) {
      if (desc.md5.empty() || desc.md5 != TargetDescription::generateMD5(path)) {
        return cv::Ptr< TargetData >();
      }
    }

    const cv::Ptr< TargetData > data(new TargetData());
    data->image = cv::imread(path);
    if (data->image.empty()) {
      return cv::Ptr< TargetData >();
    }

    if (!desc.contour.empty()) {
      data->mask = cv::Mat::zeros(data->image.size(), CV_8UC1);
      cv::fillPoly(data->mask, std::vector< std::vector< cv::Point > >(1, desc.contour), 255);
    }
    return data;
  }

public:
  cv::Mat image;
  cv::Mat mask;
};

template <> cv::Ptr< TargetData > load< TargetData >(const cv::FileNode &fn) {
  const cv::Ptr< const TargetDescription > desc(load< TargetDescription >(fn));
  return desc ? TargetData::retrieve(*desc) : cv::Ptr< TargetData >();
}
}

#endif