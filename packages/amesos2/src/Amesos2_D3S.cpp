// @HEADER
// *****************************************************************************
//           Amesos2: Templated Direct Sparse Solver Package
//
// Copyright 2011 NTESS and the Amesos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Amesos2_D3S_decl.hpp"

#ifdef HAVE_AMESOS2_EXPLICIT_INSTANTIATION

#include "Amesos2_D3S_def.hpp"
#include "Amesos2_ExplicitInstantiationHelpers.hpp"
#include "TpetraCore_ETIHelperMacros.h"

namespace Amesos2 {

  #define AMESOS2_D3S_LOCAL_INSTANT(S,LO,GO,N) \
  template class Amesos2::D3S<Tpetra::CrsMatrix<S, LO, GO, N>, \
                              Tpetra::MultiVector<S, LO, GO, N> >;

  TPETRA_ETI_MANGLING_TYPEDEFS()

  TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR(AMESOS2_D3S_LOCAL_INSTANT)

  #define AMESOS2_KOKKOS_IMPL_SOLVER_NAME D3S
  #include "Amesos2_Kokkos_Impl.hpp"
}

#endif  // HAVE_AMESOS2_EXPLICIT_INSTANTIATION
