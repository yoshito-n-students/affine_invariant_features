#ifndef AFFINE_INVARIANT_FEATURES_PARALLEL_TASKS
#define AFFINE_INVARIANT_FEATURES_PARALLEL_TASKS

#include <vector>

#include <boost/function.hpp>

#include <opencv2/core.hpp>

namespace affine_invariant_features {

class ParallelTasks : public std::vector< boost::function< void() > >, public cv::ParallelLoopBody {
public:
  ParallelTasks() {}

  ParallelTasks(const size_type count) : std::vector< boost::function< void() > >(count) {}

  virtual ~ParallelTasks() {}

  virtual void operator()(const cv::Range &range) const {
    for (int i = range.start; i < range.end; ++i) {
      const value_type &task((*this)[i]);
      if (task) {
        task();
      }
    }
  }
};
}

#endif