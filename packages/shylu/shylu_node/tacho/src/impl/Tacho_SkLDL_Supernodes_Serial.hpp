// clang-format off
// @HEADER
// *****************************************************************************
//                            Tacho package
//
// Copyright 2022 NTESS and the Tacho contributors.
// SPDX-License-Identifier: BSD-2-Clause
// *****************************************************************************
// @HEADER
// clang-format on
#ifndef __TACHO_SK_SUPERNODES_SERIAL_HPP__
#define __TACHO_SK_SUPERNODES_SERIAL_HPP__

/// \file Tacho_SK_Supernodes.hpp
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tacho_CholSupernodes_Serial.hpp"

#include "Tacho_Symmetrize.hpp"
#include "Tacho_Symmetrize_Internal.hpp"

#include "Tacho_ApplyPivots.hpp"
#include "Tacho_ApplyPivots_Internal.hpp"

#include "Tacho_Copy.hpp"
#include "Tacho_Copy_Internal.hpp"

#include "Tacho_Scale2x2_BlockInverseDiagonals.hpp"
#include "Tacho_Scale2x2_BlockInverseDiagonals_Internal.hpp"

#include "Tacho_LDL.hpp"
#include "Tacho_LDL_External.hpp"
#include "Tacho_LDL_Internal.hpp"

#include "Tacho_SkLDL.hpp"
#include "Tacho_SkLDL_Internal.hpp"

#include "Tacho_GemmTriangular.hpp"
#include "Tacho_GemmTriangular_External.hpp"
#include "Tacho_GemmTriangular_Internal.hpp"

namespace Tacho {

template <> struct SkLDL_Supernodes<Algo::Workflow::Serial> {
  template <typename MemberType, typename SupernodeInfoType>
  KOKKOS_INLINE_FUNCTION static int
  factorize(MemberType &member, const SupernodeInfoType &info, const typename SupernodeInfoType::ordinal_type_array &P,
            const typename SupernodeInfoType::value_type_matrix &D,
            const typename SupernodeInfoType::value_type_array &W,
            const typename SupernodeInfoType::value_type_matrix &ABR, const ordinal_type sid,
            const bool pivot) {
    using supernode_info_type = SupernodeInfoType;
    using value_type = typename supernode_info_type::value_type;
    using value_type_matrix = typename supernode_info_type::value_type_matrix;
    using ordinal_type_array = typename supernode_info_type::ordinal_type_array;

    // algorithm choice
    using FactAlgoType = typename LDL_Algorithm::type;
    using TrsmAlgoType = typename TrsmAlgorithm::type;
    using GemmAlgoType = typename GemmAlgorithm::type;
    //printf( "\n SkLDL_Supernodes::factorize (sid = %d)\n",sid );

    // get current supernode
    const auto &s = info.supernodes(sid);

    // get panel pointer
    value_type *ptr = s.u_buf;

    // panel (s.m x s.n) is divided into ATL (m x m) and ATR (m x n)
    const ordinal_type m = s.m, n = s.n - s.m;

    // m and n are available, then factorize the supernode block
    int npivots = (pivot ? 0 : -1);
    if (m > 0) {
      /// LDL factorize ATL, extract diag, symmetrize ATL with unit diagonals
      UnmanagedViewType<value_type_matrix> ATL(ptr, m, m);
      ptr += m * m;

      SkSymmetrize<Uplo::Upper, Algo::Internal>::invoke(member, ATL);

      SkLDL<Uplo::Lower, Algo::Internal>::invoke(member, ATL, P, W, npivots);
      /*printf( " > L = [\n" );
      for (int i=0; i < m; i++) {
        for (int j=0; j < m; j++) printf( "%e ",ATL(i,j) );
        printf("\n");
      }
      printf( " ]\n" );*/
      SkLDL<Uplo::Lower, Algo::Internal>::modify(member, ATL, P, D);
      /*printf( " D = [\n" );
      for (int i=0; i < m; i++) printf( "%.16e %.16e\n",D(i,0),D(i,1) );
      printf( " ]\n" );
      printf( " * L = [\n" );
      for (int i=0; i < m; i++) {
        for (int j=0; j < m; j++) printf( "%.16e ",ATL(i,j) );
        printf("\n");
      }
      printf( " ]\n" );*/

      if (n > 0) {
        const value_type one(1), zero(0);
        UnmanagedViewType<value_type_matrix> ATR(ptr, m, n);
        ptr += m * n;
        UnmanagedViewType<value_type_matrix> STR(W.data(), m, n);
        /*printf( " > B = [\n" );
        for (int i=0; i < m; i++) {
          for (int j=0; j < n; j++) printf( "%e ",ATR(i,j) );
          printf("\n");
        }
        printf( " ]\n" );*/

        auto fpiv = ordinal_type_array(P.data() + m, m);
        ApplyPivots<PivotMode::Flame, Side::Left, Direct::Forward, Algo::Internal> /// row inter-change
            ::invoke(member, fpiv, ATR);
        Trsm<Side::Left, Uplo::Lower, Trans::NoTranspose, TrsmAlgoType>::invoke(member, Diag::Unit(), one, ATL, ATR);
        /*printf( " > Y = [\n" );
        for (int i=0; i < m; i++) {
          for (int j=0; j < n; j++) printf( "%e ",ATR(i,j) );
          printf("\n");
        }
        printf( " ]\n" );*/

        Copy<Algo::Internal>::invoke(member, STR, ATR);
        Scale2x2_BlockInverseDiagonals<Side::Left, Algo::Internal> /// row scaling
            ::invoke(member, P, D, ATR);
        /*printf( " > X = [\n" );
        for (int i=0; i < m; i++) {
          for (int j=0; j < n; j++) printf( "%e ",ATR(i,j) );
          printf("\n");
        }
        printf( " ]\n" );*/

        TACHO_TEST_FOR_ABORT(static_cast<ordinal_type>(ABR.extent(0)) != n ||
                                 static_cast<ordinal_type>(ABR.extent(1)) != n,
                             "ABR dimension does not match to supernodes");
        GemmTriangular<Trans::Transpose, Trans::NoTranspose, Uplo::Upper, GemmAlgoType>::invoke(member, -one, ATR, STR,
                                                                                                zero, ABR);
      }
    }
    s.npivots = npivots;

    return 0;
  }

  template <typename MemberType, typename SupernodeInfoType>
  KOKKOS_INLINE_FUNCTION static int
  factorize_recursive_serial(MemberType &member, const SupernodeInfoType &info, const ordinal_type sid,
                             const bool final, typename SupernodeInfoType::ordinal_type_array::pointer_type piv,
                             typename SupernodeInfoType::value_type_array::pointer_type diag,
                             typename SupernodeInfoType::value_type_array::pointer_type buf, const size_type bufsize,
                             const bool pivot) {
    using supernode_info_type = SupernodeInfoType;
    using value_type = typename supernode_info_type::value_type;
    using value_type_array = typename supernode_info_type::value_type_array;
    using value_type_matrix = typename supernode_info_type::value_type_matrix;
    using ordinal_type_array = typename supernode_info_type::ordinal_type_array;

    const auto &s = info.supernodes(sid);
    /*{
      const auto &t = info.supernodes(60154);
      value_type *ptr = t.u_buf;
      const ordinal_type m = t.m;
      printf( " check (sid = 60154) : m = %d\n",m );
      UnmanagedViewType<value_type_matrix> ATL(ptr, m, m);
      printf( "[\n" );
      for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) printf( " %e",ATL(i,j) );
        printf("\n");
      }
      printf( "];\n\n" );
    }*/
    if (final) {
      //printf( "   * factorize_recursive_serial (nchild = %d) *\n",s.nchildren );
      // serial recursion
      for (ordinal_type i = 0; i < s.nchildren; ++i)
        factorize_recursive_serial(member, info, s.children[i], final, piv, diag, buf, bufsize, pivot);
    }

    {
      //printf( "\n > factorize_recursive_serial (sid = %d) <\n",sid );
      const ordinal_type m = s.m;
      const ordinal_type rbeg = s.row_begin;
      UnmanagedViewType<ordinal_type_array> ipiv(piv + rbeg * 4, 4 * m);
      UnmanagedViewType<value_type_matrix> dblk(diag + rbeg * 2, m, 2);

      const ordinal_type n = s.n - s.m;

      const ordinal_type mm = m < 32 ? m : 32;
      const ordinal_type mn = mm > n ? mm : n;

      const size_type bufsize_required = (n * n + m * mn) * sizeof(value_type);
      TACHO_TEST_FOR_ABORT(bufsize < bufsize_required, "bufsize is smaller than required");
      value_type *bufptr = buf;
      UnmanagedViewType<value_type_matrix> ABR(bufptr, n, n);
      bufptr += ABR.span();
      UnmanagedViewType<value_type_array> w(bufptr, m * mn);
      bufptr += w.span();

      SkLDL_Supernodes<Algo::Workflow::Serial>::factorize(member, info, ipiv, dblk, w, ABR, sid, pivot);

      /// assembly is same
      CholSupernodes<Algo::Workflow::Serial>::update(member, info, ABR, sid, bufsize - ABR.span() * sizeof(value_type),
                                                     (void *)(w.data()));
    }
    /*{
      const auto &t = info.supernodes(60154);
      value_type *ptr = t.u_buf;
      const ordinal_type m = t.m;
      printf( " check (sid = 60154) : m = %d\n",m );
      UnmanagedViewType<value_type_matrix> ATL(ptr, m, m);
      printf( " -> [\n" );
      for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) printf( " %e",ATL(i,j) );
        printf("\n");
      }
      printf( "];\n\n" );
    }*/
    return 0;
  }

  template <typename MemberType, typename SupernodeInfoType>
  KOKKOS_INLINE_FUNCTION static int
  solve_lower_recursive_serial(MemberType &member, const SupernodeInfoType &info, const ordinal_type sid,
                               const bool final, typename SupernodeInfoType::ordinal_type_array::pointer_type piv,
                               typename SupernodeInfoType::value_type_array::pointer_type buf,
                               const size_type bufsize) {
    using supernode_info_type = SupernodeInfoType;

    using value_type = typename supernode_info_type::value_type;
    using value_type_matrix = typename supernode_info_type::value_type_matrix;
    using ordinal_type_array = typename supernode_info_type::ordinal_type_array;

    const auto &s = info.supernodes(sid);

    if (final) {
      // serial recursion
      for (ordinal_type i = 0; i < s.nchildren; ++i)
        solve_lower_recursive_serial(member, info, s.children[i], final, piv, buf, bufsize);
    }

    {
      const ordinal_type m = s.m;
      const ordinal_type rbeg = s.row_begin;
      UnmanagedViewType<ordinal_type_array> ipiv(piv + rbeg * 4, 4 * m);

      const ordinal_type n = s.n - s.m;
      const ordinal_type nrhs = info.x.extent(1);
      const size_type bufsize_required = n * nrhs * sizeof(value_type);

      TACHO_TEST_FOR_ABORT(bufsize < bufsize_required, "bufsize is smaller than required");

      UnmanagedViewType<value_type_matrix> xB((value_type *)buf, n, nrhs);

      LDL_Supernodes<Algo::Workflow::Serial>::solve_lower(member, info, ipiv, xB, sid);

      CholSupernodes<Algo::Workflow::Serial>::update_solve_lower(member, info, xB, sid);
    }
    return 0;
  }

  template <typename MemberType, typename SupernodeInfoType>
  KOKKOS_INLINE_FUNCTION static int
  solve_upper_recursive_serial(MemberType &member, const SupernodeInfoType &info, const ordinal_type sid,
                               const bool final, typename SupernodeInfoType::ordinal_type_array::pointer_type piv,
                               typename SupernodeInfoType::value_type_array::pointer_type diag,
                               typename SupernodeInfoType::value_type_array::pointer_type buf,
                               const ordinal_type bufsize) {
    using supernode_info_type = SupernodeInfoType;
    using value_type = typename supernode_info_type::value_type;
    using value_type_matrix = typename supernode_info_type::value_type_matrix;
    using ordinal_type_array = typename supernode_info_type::ordinal_type_array;

    const auto &s = info.supernodes(sid);
    {
      const ordinal_type m = s.m;
      const ordinal_type rbeg = s.row_begin;
      UnmanagedViewType<ordinal_type_array> ipiv(piv + rbeg * 4, 4 * m);
      UnmanagedViewType<value_type_matrix> dblk(diag + rbeg * 2, m, 2);

      const ordinal_type n = s.n - s.m;
      const ordinal_type nrhs = info.x.extent(1);
      const ordinal_type bufsize_required = n * nrhs * sizeof(value_type);

      TACHO_TEST_FOR_ABORT(bufsize < bufsize_required, "bufsize is smaller than required");

      UnmanagedViewType<value_type_matrix> xB((value_type *)buf, n, nrhs);

      CholSupernodes<Algo::Workflow::Serial>::update_solve_upper(member, info, xB, sid);

      LDL_Supernodes<Algo::Workflow::Serial>::solve_upper(member, info, ipiv, dblk, xB, sid);
    }

    if (final) {
      // serial recursion
      for (ordinal_type i = 0; i < s.nchildren; ++i)
        solve_upper_recursive_serial(member, info, s.children[i], final, piv, diag, buf, bufsize);
    }
    return 0;
  }

  template <typename MemberType, typename SupernodeInfoType>
  KOKKOS_INLINE_FUNCTION static int
  get_diag_recursive_serial(MemberType &member, const SupernodeInfoType &info,
                            typename SupernodeInfoType::value_type_array::pointer_type diag,
                            typename SupernodeInfoType::value_type_array::pointer_type D,
                            const ordinal_type sid, const bool final) {
    using supernode_info_type = SupernodeInfoType;
    using value_type = typename supernode_info_type::value_type;
    using value_type_matrix = typename supernode_info_type::value_type_matrix;
    using value_type_array = typename supernode_info_type::value_type_array;
    using ordinal_type_array = typename supernode_info_type::ordinal_type_array;

    const auto &s = info.supernodes(sid);
    if (final) {
      // serial recursion
      for (ordinal_type i = 0; i < s.nchildren; ++i)
        get_diag_recursive_serial(member, info, diag, D, s.children[i], final);
    }
    {
      const ordinal_type m = s.m;
      const ordinal_type rbeg = s.row_begin;

      UnmanagedViewType<value_type_matrix> dblk(diag + rbeg * 2, m, 2);
      UnmanagedViewType<value_type_array>  d(D + rbeg, m);

      for (ordinal_type i = 0; i < m; i++) {
        d(i) = dblk(i,0);
      }
    }
    return 0;
  }

  template <typename MemberType, typename SupernodeInfoType>
  KOKKOS_INLINE_FUNCTION static int
  get_pf_recursive_serial(MemberType &member, const SupernodeInfoType &info,
                          typename SupernodeInfoType::value_type_array::pointer_type diag,
                          const ordinal_type sid, const bool final) {
    using supernode_info_type = SupernodeInfoType;
    using value_type = typename supernode_info_type::value_type;
    using value_type_matrix = typename supernode_info_type::value_type_matrix;
    using value_type_array = typename supernode_info_type::value_type_array;
    using ordinal_type_array = typename supernode_info_type::ordinal_type_array;
    using arith_traits = Tacho::ArithTraits<value_type>;

    int pf = 1;
    const auto &s = info.supernodes(sid);
    if (final) {
      // serial recursion
      for (ordinal_type i = 0; i < s.nchildren; ++i) {
        int pf_i = get_pf_recursive_serial(member, info, diag, s.children[i], final);
        pf *= pf_i;
        //printf( " => %d (%d / %d)\n",pf,i,s.nchildren );
      }
    }
    {
      //printf( "\n > get_pf_recursive_serial (sid = %d, num_childs = %d) <\n",sid,s.nchildren );
      const ordinal_type m = s.m;
      const ordinal_type rbeg = s.row_begin;

      // from det(T)
      //printf( "  pf = %d ",pf );
      UnmanagedViewType<value_type_matrix> dblk(diag + rbeg * 2, m, 2); // input
      for (ordinal_type i = 1; i < m; i+=2) {
        if (arith_traits::real(dblk(i,0)) < 0.0) {
          pf *= -1;
        }
      }
      //printf(" -> %d ",pf );
      // from det(P)
      if (s.npivots >= 0) {
        pf *= pow(-1,s.npivots);
      }
      //printf(" x -1^(%d) -> %d (%d)\n",s.npivots,pf,sid );
    }
    return pf;
  }
};
} // namespace Tacho

#endif
