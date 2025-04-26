// clang-format off
// @HEADER
// *****************************************************************************
//                            Tacho package
//
// Copyright 2022 NTESS and the Tacho contributors.
// SPDX-License-Identifier: BSD-2-Clause
// *****************************************************************************
// @HEADER

#ifndef TACHO_MATLABUTILS_DEF_HPP
#define TACHO_MATLABUTILS_DEF_HPP

#include "Tacho_MatlabUtils_decl.hpp"
#include <mex.h>

#if !defined(TACHO_HAVE_MATLAB)
#error "Tacho_Matlab_def requires MATLAB."
#else

namespace Tacho {

/* ******************************* */
/* loadDataFromMatlab              */
/* ******************************* */

template <>
int loadDataFromMatlab<int>(const mxArray* mxa) {
  mxClassID probIDtype = mxGetClassID(mxa);
  int rv = 0;
  if (probIDtype == mxINT32_CLASS) {
    rv = *((int*)mxGetData(mxa));
  } else if (probIDtype == mxLOGICAL_CLASS) {
    rv = (int)*((bool*)mxGetData(mxa));
  } else if (probIDtype == mxDOUBLE_CLASS) {
    rv = (int)*((double*)mxGetData(mxa));
  } else if (probIDtype == mxUINT32_CLASS) {
    rv = (int)*((unsigned int*)mxGetData(mxa));
  } else {
    rv = -1;
    throw std::runtime_error("Error: Unrecognized numerical type.");
  }
  return rv;
}

template <>
std::string loadDataFromMatlab<std::string>(const mxArray* mxa) {
  std::string rv = "";
  if (mxGetClassID(mxa) != mxCHAR_CLASS) {
    throw std::runtime_error("Can't construct string from anything but a char array.");
  }
  rv = std::string(mxArrayToString(mxa));
  return rv;
}

template <typename ScalarType, typename DeviceType>
int loadMatrixFromMatlab(const mxArray* mxa, CrsMatrixBase<ScalarType, DeviceType> &A) {

  const size_t m = mxGetM(mxa);
  const size_t n = mxGetN(mxa);
  double *nzvals = mxGetPr(mxa);
  mwIndex *rowind = mxGetIr(mxa);
  mwIndex *colptr = mxGetJc(mxa);
  int nnz = colptr[n];
  //printf( " m=%d, n=%d\n",m,n );
  /*printf("A=[\n");
  for (int j=0; j<n; j++) {
    for (int k=colptr[j]; k<colptr[j+1]; k++) {
      printf("%d, %d %d %e\n",k, rowind[k],j,nzvals[k]);
    }
  }
  printf("];\n");*/

  Kokkos::View<size_type *, DeviceType>    ap("ap", m + 1);
  Kokkos::View<ordinal_type *, DeviceType> aj("aj", nnz);
  Kokkos::View<ScalarType *, DeviceType>   av("ax", nnz);

  int rv = 0;
  // Transopose to convert to CSR
  for (int i=0; i<=m; i++) {
    ap(i) = 0;
  }
  for (int k=0; k<nnz; k++) {
    if (rowind[k] < m-1) {
      ap(rowind[k]+2) ++;
    }
  }
  for (int i=1; i < m; i++) {
    ap(i+1) += ap(i);
  }
  for (int j=0; j<n; j++) {
    for (int k=colptr[j]; k<colptr[j+1]; k++) {
      aj(ap(1+rowind[k])) = j;
      av(ap(1+rowind[k])) = nzvals[k];
      ap(1+rowind[k]) ++;
    }
  }
  /*printf("A=[\n");
  for (int i=0; i<m; i++) {
    for (int k=ap(i); k<ap(i+1); k++) {
      printf("%d, %d %d %e\n",k, i,aj(k),av(k));
    }
  }
  printf("];\n");*/

  A.clear();
  A.setExternalMatrix(m, n, nnz, ap, aj, av);
  return rv;
}

template <typename ValueType, typename DeviceType>
int loadMultiVectorsFromMatlab(const mxArray* mxa, Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType> &B) {
  const size_t m = mxGetM(mxa);
  const size_t n = mxGetN(mxa);

  int rv = 0;
  double* pr = mxGetPr(mxa);
  Kokkos::resize(B, m,n);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      B(i,j) = pr[i + j*m];
    }
  }
  return rv;
}

/* ******************************* */
/* saveDataToMatlab                */
/* ******************************* */

template <typename ValueType, typename DeviceType>
mxArray* saveMultiVectorsToMatlab(Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType> &X) {
  mwSize m = X.extent(0);
  mwSize n = X.extent(1);

  mxArray* output = mxCreateDoubleMatrix(m, n, mxREAL);
  ValueType* array = (ValueType*) malloc(sizeof(ValueType) * m * n);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      array[i + j*m] = X(i,j);
    }
  }
  memcpy(mxGetPr(output), array, (m*n) * sizeof(double));
  free(array);
  return output;
}

template <typename ValueType, typename DeviceType>
mxArray* saveVectorToMatlab(Kokkos::View<ValueType *, Kokkos::LayoutLeft, DeviceType> &X) {
  mwSize m = X.extent(0);

  mxArray* output = mxCreateDoubleMatrix(m, 1, mxREAL);
  ValueType* array = (ValueType*) malloc(sizeof(ValueType) * m);
  for (int i = 0; i < m; i++) {
    array[i] = X(i);
  }
  memcpy(mxGetPr(output), array, m * sizeof(double));
  free(array);
  return output;
}

}
#endif
#endif
