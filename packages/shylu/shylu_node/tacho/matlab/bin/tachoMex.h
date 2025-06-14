// clang-format off
// @HEADER
// *****************************************************************************
//                            Tacho package
//
// Copyright 2022 NTESS and the Tacho contributors.
// SPDX-License-Identifier: BSD-2-Clause
// *****************************************************************************
// @HEADER

#ifndef MUEMEX_H
#define MUEMEX_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <complex>
#include <stdexcept>

#include "Tacho_CrsMatrixBase.hpp"
#include "Tacho_Driver.hpp"
#include "Tacho_MatlabUtils.hpp"

#include "mex.h"

namespace Tacho
{

//Mode value passed to MATLAB function as 1st arg (int)
typedef enum
  {
    MODE_SETUP,   //0
    MODE_FACTOR,  //1
    MODE_SOLVE,   //2
    MODE_CLEANUP, //3
    MODE_OPTION,  //4
    MODE_DIAG,    //5
    MODE_ERROR    //6
  } MODE_TYPE;


//Scalar can be double or std::complex<double> (complex_t)
template<typename value_type>
class TachoSystem 
{
 public:
  TachoSystem();
  ~TachoSystem();
  using host_device_type = typename Tacho::UseThisDevice<Kokkos::DefaultHostExecutionSpace>::type;  
  using CrsMatrixBaseTypeHost = Tacho::CrsMatrixBase<value_type, host_device_type>;
  using DenseMultiVectorType = Kokkos::View<value_type **, Kokkos::LayoutLeft, host_device_type>;
  using DenseVectorType = Kokkos::View<value_type *, Kokkos::LayoutLeft, host_device_type>;
  int option(const mxArray** mx);
  int setup(const mxArray* mx);
  int factor(const mxArray* mx);
  mxArray* solve(const mxArray* mx);
  mxArray* diag();

  bool verbose() { return _verbose; };
 private:
  bool _verbose;
  bool _max_match;
  bool _max_weight;
  bool _scale_mat;
  int _dofs_per_node;
  CrsMatrixBaseTypeHost A;
  Tacho::Driver<value_type, host_device_type> solver;
};

}// end namespace

#endif //MUEMEX_H
