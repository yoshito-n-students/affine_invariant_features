#ifndef AFFINE_INVARIANT_FEATURES_CV_SERIALIZABLE
#define AFFINE_INVARIANT_FEATURES_CV_SERIALIZABLE

#include <string>

#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>

#include <opencv2/core.hpp>

namespace affine_invariant_features {

struct CvSerializable {
  CvSerializable() {}

  virtual ~CvSerializable() {}

  // return the name of type
  virtual std::string getDefaultName() const = 0;

  // read name-value pairs corresponding member variables
  virtual void read(const cv::FileNode &) = 0;

  // write name-value pairs corresponding member variables
  virtual void write(cv::FileStorage &) const = 0;

  // write the name of type and members
  virtual void save(cv::FileStorage &fs) const {
    fs << getDefaultName() << "{";
    write(fs);
    fs << "}";
  }
};

// read members belonging the name of type, if there
template < typename T >
typename boost::enable_if< boost::is_base_of< CvSerializable, T >, cv::Ptr< T > >::type
load(const cv::FileNode &fn) {
  const cv::Ptr< T > val(new T());
  const cv::FileNode node(fn[val->getDefaultName()]);
  if (node.empty()) {
    return cv::Ptr< T >();
  } else {
    val->read(node);
    return val;
  }
}

// called by operator>>(cv::FileNode, T)
template < typename T >
typename boost::enable_if< boost::is_base_of< CvSerializable, T > >::type
read(const cv::FileNode &fn, T &val, const T &default_val) {
  if (fn.empty()) {
    val = default_val;
  } else {
    val.read(fn);
  }
}

// called by operator<<(cv::FileStorage, T)
static inline void write(cv::FileStorage &fs, const std::string &, const CvSerializable &val) {
  val.write(fs);
}
}

#endif