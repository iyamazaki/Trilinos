// @HEADER
// *****************************************************************************
//           Amesos2: Templated Direct Sparse Solver Package
//
// Copyright 2011 NTESS and the Amesos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/**
   \file   Amesos2_D3S_TypeMap.hpp
   \author Siva Rajamanickam <srajama@sandia.gov>

   \brief Provides definition of D3S types as well as conversions and type
	  traits.

*/

#ifndef AMESOS2_D3S_TYPEMAP_HPP
#define AMESOS2_D3S_TYPEMAP_HPP

#include <functional>
#ifdef HAVE_TEUCHOS_COMPLEX
#include <complex>
#endif

#include <Teuchos_as.hpp>
#ifdef HAVE_TEUCHOS_COMPLEX
#include <Teuchos_SerializationTraits.hpp>
#endif

#include "Amesos2_TypeMap.hpp"

namespace D3S {

#include "klu2_ext.hpp"	// for Dtype_t declaration

} // end namespace KLU

namespace Amesos2 {

template <class, class> class D3S;

/* Specialize the Amesos2::TypeMap struct for D3S types
 * TODO: Mostly dummy assignments as D3S is templated. Remove if possible.
 *
 * \cond D3S_type_specializations
 */
template <>
struct TypeMap<D3S,float>
{
  typedef float dtype;
  typedef float type;
};

template <>
struct TypeMap<D3S,double>
{
  typedef double dtype;
  typedef double type;
};

#ifdef HAVE_TEUCHOS_COMPLEX

template <>
struct TypeMap<D3S,std::complex<float> >
{
  typedef std::complex<double> dtype;
  typedef Kokkos::complex<double> type;
};

template <>
struct TypeMap<D3S,std::complex<double> >
{
  typedef std::complex<double> dtype;
  typedef Kokkos::complex<double> type;
};

template <>
struct TypeMap<D3S,Kokkos::complex<float> >
{
  typedef std::complex<double> dtype;
  typedef Kokkos::complex<double> type;
};

template <>
struct TypeMap<D3S,Kokkos::complex<double> >
{
  typedef std::complex<double> dtype;
  typedef Kokkos::complex<double> type;
};

#endif  // HAVE_TEUCHOS_COMPLEX

/* \endcond D3S_type_specializations */


} // end namespace Amesos2

#endif  // AMESOS2_SUPERLU_TYPEMAP_HPP
