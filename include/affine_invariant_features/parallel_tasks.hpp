#ifndef AFFINE_INVARIANT_FEATURES_PARALLEL_TASKS
#define AFFINE_INVARIANT_FEATURES_PARALLEL_TASKS

#include <iostream>
#include <stdexcept>
#include <vector>

#include <boost/function.hpp>

#include <opencv2/core.hpp>

namespace affine_invariant_features {

class ParallelTasks : public std::vector< boost::function< void() > >, public cv::ParallelLoopBody {
private:
  typedef std::vector< boost::function< void() > > Base;

public:
  ParallelTasks() : Base() {}

  ParallelTasks(const size_type count) : Base(count) {}

  ParallelTasks(const size_type count, const value_type &value) : Base(count, value) {}

  virtual ~ParallelTasks() {}

  virtual void operator()(const cv::Range &range) const {
    for (int i = range.start; i < range.end; ++i) {
      // handle an exception from the task
      // because it cannot be catched by the main thread running cv::parallel_for_()
      try {
        // at() may throw std::out_of_range unlike the operator []
        const value_type &task(at(i));
        CV_Assert(task);
        task();
      } catch (const std::exception &error) {
        std::cerr << "Parallel task [" << i << "]: " << error.what() << std::endl;
      } catch (...) {
        std::cerr << "Parallel task [" << i << "]: Non-standard error" << std::endl;
      }
    }
  }
};

} // namespace affine_invariant_features

#endif