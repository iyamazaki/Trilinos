#ifndef TEST_SERIAL_HPP
#define TEST_SERIAL_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosKernels_config.h>

#if defined(KOKKOSKERNELS_TEST_ETI_ONLY) && !defined(KOKKOSKERNELS_ETI_ONLY)
#define KOKKOSKERNELS_ETI_ONLY
#endif

class serial : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
  }

  static void TearDownTestCase()
  {
  }
};

#define TestCategory serial
#define TestExecSpace Kokkos::Serial

#endif // TEST_SERIAL_HPP
