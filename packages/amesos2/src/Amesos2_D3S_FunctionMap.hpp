// @HEADER
// *****************************************************************************
//           Amesos2: Templated Direct Sparse Solver Package
//
// Copyright 2011 NTESS and the Amesos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/**
   \file   Amesos2_D3S_FunctionMap.hpp
   \author Siva Rajamanickam <srajama@sandia.gov>

   \brief  Provides a mechanism to map function calls to the correct Solver
           function based on the scalar type of Matrices and MultiVectors
*/

#ifndef AMESOS2_D3S_FUNCTIONMAP_HPP
#define AMESOS2_D3S_FUNCTIONMAP_HPP

// Note since Klu2 is templated we don't use function maps.
// Includes are still collected here which mirrors setup in other solvers.

#ifdef HAVE_TEUCHOS_COMPLEX
#include <complex>
#endif

#include "Amesos2_FunctionMap.hpp"
#include "Amesos2_D3S_TypeMap.hpp"

/* External definitions of the D3S functions
 */
#include "d3_solver.h"


namespace Amesos2 {

  // double
  template <>
  struct FunctionMap<D3S,double>
  {
    static double * convert_scalar(double * pData) {
      return pData; // no conversion necessary
    }
  };

  // float
  template <>
  struct FunctionMap<D3S,float>
  {
    static float * convert_scalar(float * pData) {
      return pData; // no conversion necessary
    }
  };

#ifdef HAVE_TEUCHOS_INST_COMPLEX_DOUBLE
  template <>
  struct FunctionMap<D3S, Kokkos::complex<double>>
  {
    static std::complex<double> * convert_scalar(Kokkos::complex<double> * pData) {
      return reinterpret_cast<std::complex<double> *>(pData);
    }
  };

  template <>
  struct FunctionMap<D3S, std::complex<double>>
  {
    static Kokkos::complex<double> * convert_scalar(std::complex<double> * pData) {
      return reinterpret_cast<Kokkos::complex<double> *>(pData);
    }
  };
#endif // HAVE_TEUCHOS_COMPLEX_DOUBLE

#ifdef HAVE_TEUCHOS_INST_COMPLEX_FLOAT
  template <>
  struct FunctionMap<D3S, Kokkos::complex<float>>
  {
    static std::complex<float> * convert_scalar(Kokkos::complex<float> * pData) {
      return reinterpret_cast<std::complex<float> *>(pData);
    }
  };

  template <>
  struct FunctionMap<D3S, std::complex<float>>
  {
    static Kokkos::complex<float> * convert_scalar(std::complex<float> * pData) {
      return reinterpret_cast<Kokkos::complex<float> *>(pData);
    }
  };
#endif // HAVE_TEUCHOS_COMPLEX_FLOAT

  // if not specialized, then assume generic conversion is fine
  template <typename scalar_t>
  struct FunctionMap<D3S,scalar_t>
  {
    static scalar_t * convert_scalar(scalar_t * pData) {
      return pData; // no conversion necessary
    }
  };
} // end namespace Amesos2

#endif  // AMESOS2_D3S_FUNCTIONMAP_HPP
