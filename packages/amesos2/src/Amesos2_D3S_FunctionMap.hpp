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

namespace Amesos2 {

#ifdef HAVE_TEUCHOS_COMPLEX
  template <>
  struct FunctionMap<D3S,Kokkos::complex<double>>
  {
    static std::complex<double> * convert_scalar(Kokkos::complex<double> * pData) {
      return reinterpret_cast<std::complex<double> *>(pData);
    }
  };

  // Note that Klu2 does not support complex float so it does not appear here.
#endif // HAVE_TEUCHOS_COMPLEX

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
