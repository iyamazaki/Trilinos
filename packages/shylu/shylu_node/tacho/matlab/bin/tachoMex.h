// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
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
    MODE_ERROR    //5
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
  int setup(const mxArray* mx);
  int factor(const mxArray* mx);
  mxArray* solve(const mxArray* mx);
  int option(const mxArray** mx);

 private:
  bool verbose;
  int dofs_per_node;
  CrsMatrixBaseTypeHost A;
  Tacho::Driver<value_type, host_device_type> solver;
  int dofs_per_node;
};

}// end namespace

#endif //MUEMEX_H
