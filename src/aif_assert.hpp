#ifndef AIF_ASSERT
#define AIF_ASSERT

#include <opencv2/core.hpp>

#define AIF_Assert(expr, ...)                                                                      \
  if (!!(expr))                                                                                    \
    ;                                                                                              \
  else                                                                                             \
    CV_Error_(cv::Error::StsError, (__VA_ARGS__))

#endif