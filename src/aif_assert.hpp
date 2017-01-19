#ifndef AIF_ASSERT
#define AIF_ASSERT

#include <opencv2/core.hpp>

// print the given message and raise an error if the given expression is ture
#define AIF_Assert(expr, ...)                                                                      \
  if (!!(expr))                                                                                    \
    ;                                                                                              \
  else                                                                                             \
    CV_Error_(cv::Error::StsAssert, (__VA_ARGS__))

#endif