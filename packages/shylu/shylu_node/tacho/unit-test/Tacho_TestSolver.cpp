// clang-format off
// @HEADER
// *****************************************************************************
//                            Tacho package
//
// Copyright 2022 NTESS and the Tacho contributors.
// SPDX-License-Identifier: BSD-2-Clause
// *****************************************************************************
// @HEADER
// clang-format on
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "Tacho_CommandLineParser.hpp"
#include "Tacho_CrsMatrixBase.hpp"
#include "Tacho_Driver.hpp"
#include "Tacho_MatrixMarket.hpp"

using ordinal_type = Tacho::ordinal_type;

template <typename value_type> int driver(std::string file, std::string method_name, int variant, int nrhs) {
  int method = 1; // 1 - Chol, 2 - LDL, 3 - SymLU
  if (method_name == "chol")
    method = 1;
  else if (method_name == "ldl")
    method = 2;
  else if (method_name == "lu")
    method = 3;
  else {
    std::cout << "Error: not supported solution method\n";
    return -1;
  }

  using device_type = typename Tacho::UseThisDevice<Kokkos::DefaultExecutionSpace>::type;
  using host_device_type = typename Tacho::UseThisDevice<Kokkos::DefaultHostExecutionSpace>::type;

  const bool detail = false;
  const bool verbose = false;
  Tacho::printExecSpaceConfiguration<typename device_type::execution_space>("DeviceSpace", detail);
  Tacho::printExecSpaceConfiguration<typename host_device_type::execution_space>("HostSpace", detail);
  std::cout << std::endl << "    --------------------- " << std::endl;
  std::cout << "     Method Name:: " << method_name << std::endl;
  std::cout << "     Solver Type:: " << variant << std::endl;
  std::cout << "          # RHSs:: " << nrhs << std::endl;
  std::cout << "          Matrix:: " << file;
  std::cout << std::endl << "    --------------------- " << std::endl << std::endl;

  int r_val = 0;
  try {
    /// crs matrix format and dense multi vector
    using CrsMatrixBaseTypeHost = Tacho::CrsMatrixBase<value_type, host_device_type>;
    using DenseMultiVectorType = Kokkos::View<value_type **, Kokkos::LayoutLeft, device_type>;

    /// read a spd matrix of matrix market format
    CrsMatrixBaseTypeHost A;
    {
      std::ifstream in;
      in.open(file);
      if (!in.good()) {
        std::cout << "Failed in open the file: " << file << std::endl;
        return -1;
      }
      bool sanitize = false;
      Tacho::MatrixMarket<value_type>::read(file, A, sanitize, verbose);
    }

    /// read graph file if available
    using size_type_array_host = typename CrsMatrixBaseTypeHost::size_type_array;
    using ordinal_type_array_host = typename CrsMatrixBaseTypeHost::ordinal_type_array;

    Tacho::Driver<value_type, device_type> solver;

    /// common options
    solver.setVerbose(verbose);
    solver.setSolutionMethod(method);
    solver.setLevelSetOptionAlgorithmVariant(variant);

    auto values_on_device = Kokkos::create_mirror_view(typename device_type::memory_space(), A.Values());
    Kokkos::deep_copy(values_on_device, A.Values());

    /// initialize
    r_val = solver.analyze(A.NumRows(), A.RowPtr(), A.Cols());
    if(r_val == 0) {
      r_val = solver.initialize();
    }
    /// create numeric tools and levelset tools
    if(r_val == 0) {
      r_val = solver.factorize(values_on_device);
    }
    /// solve
    DenseMultiVectorType b("b", A.NumRows(), nrhs), // rhs multivector
        x("x", A.NumRows(), nrhs),                  // solution multivector
        t("t", A.NumRows(), nrhs);                  // temp workspace (store permuted rhs)
    if(r_val == 0) {
      const value_type zero(0.0);
      const value_type one (1.0);
      if (true) {
        Kokkos::deep_copy (b, one);
      } else {
        Kokkos::deep_copy (x, one);
        solver.computeSpMV(values_on_device, x, b);
        Kokkos::deep_copy (x, zero);
      }
      r_val = solver.solve(x, b, t);
    }
    if(r_val == 0) {
      const double tol = 0.0000000001;
      const double res = solver.computeRelativeResidual(values_on_device, x, b);
      r_val = (res <= tol ? 0 : -1);
      std::cout << " res = " << res << " tol = " << tol << "(rval = " << r_val << ")"
                << std::endl << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught: \n" << e.what() << "\n";
  }
  return r_val;
}

TEST( Solver, Chol ) {
  // Chol
  std::string file = "bcsstk15.mtx";
  // > single RHS
  EXPECT_EQ(driver<double>(file, "chol", 0, 1), 0);
  EXPECT_EQ(driver<double>(file, "chol", 1, 1), 0);
  EXPECT_EQ(driver<double>(file, "chol", 2, 1), 0);
  EXPECT_EQ(driver<double>(file, "chol", 3, 1), 0);
  // > multiple RHSs
  EXPECT_EQ(driver<double>(file, "chol", 0, 5), 0);
  EXPECT_EQ(driver<double>(file, "chol", 1, 5), 0);
  EXPECT_EQ(driver<double>(file, "chol", 2, 5), 0);
  EXPECT_EQ(driver<double>(file, "chol", 3, 5), 0);
}

TEST( Solver, LU ) {
  // LU
  std::string file = "sherman3.mtx";
  // > single RHS
  EXPECT_EQ(driver<double>(file, "lu", 0, 1), 0);
  EXPECT_EQ(driver<double>(file, "lu", 1, 1), 0);
  EXPECT_EQ(driver<double>(file, "lu", 2, 1), 0);
  EXPECT_EQ(driver<double>(file, "lu", 3, 1), 0);
  // > multiple RHSs
  EXPECT_EQ(driver<double>(file, "lu", 0, 5), 0);
  EXPECT_EQ(driver<double>(file, "lu", 1, 5), 0);
  EXPECT_EQ(driver<double>(file, "lu", 2, 5), 0);
  EXPECT_EQ(driver<double>(file, "lu", 3, 5), 0);
}

TEST( Solver, LDL ) {
  // LDL
  std::string file = "bcsstk15.mtx";
  // > single RHS
  EXPECT_EQ(driver<double>(file, "ldl", 0, 1), 0);
  EXPECT_EQ(driver<double>(file, "ldl", 1, 1), 0);
  EXPECT_EQ(driver<double>(file, "ldl", 2, 1), 0);
  // > multiple RHSs
  EXPECT_EQ(driver<double>(file, "ldl", 0, 5), 0);
  EXPECT_EQ(driver<double>(file, "ldl", 1, 5), 0);
  EXPECT_EQ(driver<double>(file, "ldl", 2, 5), 0);
}


int main(int argc, char *argv[]) {
  int info = 0;
  Kokkos::initialize(argc, argv);
  {
     ::testing::InitGoogleTest(&argc, argv);
     info = RUN_ALL_TESTS();
  }
  Kokkos::finalize();
  return info;
}

