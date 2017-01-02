#ifndef AFFINE_INVARIANT_FEATURES_PARALLEL_TASKS
#define AFFINE_INVARIANT_FEATURES_PARALLEL_TASKS

#include <vector>

#include <boost/function.hpp>

#include <opencv2/core.hpp>

namespace affine_invariant_features {

struct ParallelTasks : public std::vector< boost::function< void() > >,
                       public cv::ParallelLoopBody {
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