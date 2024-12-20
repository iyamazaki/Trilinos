// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TACHO_MATLABUTILS_DECL_HPP
#define TACHO_MATLABUTILS_DECL_HPP

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <complex>
#include <stdexcept>

#include "Tacho_config.h"
#include "Tacho_CrsMatrixBase.hpp"
#include "Tacho_Driver.hpp"

#if !defined(TACHO_HAVE_MATLAB)
#error "Tacho_Matlab_decl requires MATLAB."
#else

// Matlab fwd style declarations
struct mxArray_tag;
typedef struct mxArray_tag mxArray;

namespace Tacho {

template <typename T>
T loadDataFromMatlab(const mxArray* mxa);


template <typename ValueType, typename DeviceType>
int loadMatrixFromMatlab(const mxArray* mxa, CrsMatrixBase<ValueType, DeviceType> &A);

template <typename ValueType, typename DeviceType>
int loadMultiVectorsFromMatlab(const mxArray* mxa, Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType> &B);


template <typename ValueType, typename DeviceType>
mxArray* saveMultiVectorsToMatlab(Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType> &X);

template <typename ValueType, typename DeviceType>
mxArray* saveVectorToMatlab(Kokkos::View<ValueType *, Kokkos::LayoutLeft, DeviceType> &X);

}

#endif
#endif
