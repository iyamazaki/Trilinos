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

  template <>
  struct FunctionMap<D3S,double>
  {
    static double * convert_scalar(double * pData) {
      return pData; // no conversion necessary
    }

    static int factorize(const Teuchos::RCP<D3Solver> solver, const std::vector<double> & values) {
      return solver->factorize(values);
    }

    static int solve(const Teuchos::RCP<D3Solver> solver, const std::vector<double> & rhs,
                                                                std::vector<double> & sol) {
      return solver->solve(rhs, sol);
    }
  };

  // if not specialized, then assume generic conversion is fine
  template <typename scalar_t>
  struct FunctionMap<D3S,scalar_t>
  {
    static scalar_t * convert_scalar(scalar_t * pData) {
      return pData; // no conversion necessary
    }

    static int factorize(const Teuchos::RCP<D3Solver> solver, const std::vector<scalar_t> & values) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "D3S has implemented only for double.");
      return 0;
    }

    static int solve(const Teuchos::RCP<D3Solver> solver, const std::vector<scalar_t> & rhs,
                                                                std::vector<scalar_t> & sol) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "D3S has implemented only for double.");
      return 0;
    }
  };
} // end namespace Amesos2

#endif  // AMESOS2_D3S_FUNCTIONMAP_HPP
