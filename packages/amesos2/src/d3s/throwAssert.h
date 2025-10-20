#pragma once

#define ThrowAssert(ierr, msg)        \
  if ((ierr) == 0) {                  \
    fprintf(stderr, ">> Error in file %s, line %d, error %d \n",__FILE__,__LINE__,ierr); \
    fprintf(stderr, "   %s\n", msg);  \
    throw std::runtime_error(msg);    \
  }

