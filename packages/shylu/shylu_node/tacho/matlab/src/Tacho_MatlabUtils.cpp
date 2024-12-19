// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Tacho_MatlabUtils_def.hpp"

/* Stuff for MATLAB R2006b vs. previous versions */
#if (defined(MX_API_VER) && MX_API_VER >= 0x07030000)
#else
typedef int mwIndex;
#endif


namespace Tacho {

// explicit instantiations
template int loadDataFromMatlab<int>(const mxArray* mxa);
template std::string loadDataFromMatlab<std::string>(const mxArray* mxa);


//template <typename double, typename DeviceType>
//int loadDataFromMatlab(const mxArray* mxa, CrsMatrixBase<ValueType, DeviceType> &A)

}
