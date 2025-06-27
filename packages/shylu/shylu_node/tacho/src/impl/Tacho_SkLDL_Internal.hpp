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
#ifndef __TACHO_SKLDL_INTERNAL_HPP__
#define __TACHO_SKLDL_INTERNAL_HPP__

/// \file  Tacho_SkLDL_Internal.hpp
/// \brief Skewed LDL factorization
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tacho_Blas_External.hpp"
#include "Tacho_Lapack_External.hpp"

namespace Tacho {

/// Skewed LDL
/// ==========
template <> struct SkLDL<Uplo::Lower, Algo::Internal> {
  template <typename MemberType, typename ViewTypeA, typename ViewTypeP, typename ViewTypeW>
  KOKKOS_INLINE_FUNCTION static int invoke(MemberType &member, const ViewTypeA &A, const ViewTypeP &P,
                                           const ViewTypeW &W, int &npivots) {
    typedef typename ViewTypeA::non_const_value_type value_type;
    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
    const value_type one(1), minus_one(-1), zero(0);

    static_assert(ViewTypeA::rank == 2, "A is not rank 2 view.");
    static_assert(ViewTypeP::rank == 1, "P is not rank 1 view.");
    static_assert(ViewTypeW::rank == 1, "W is not rank 1 view.");

    TACHO_TEST_FOR_EXCEPTION(P.extent(0) < 4 * A.extent(0), std::logic_error, "P should be 4*A.extent(0) .");

    int r_val(0);
    const ordinal_type m = A.extent(0);

    bool pivot = (npivots == 0);
    if (m > 0) {
      /// factorize LDL
      // TODO: move this to symbolic init
      Kokkos::View<value_type **, Kokkos::LayoutLeft, typename ViewTypeA::execution_space> Wk("WK",m,2);

      //printf( "\n SkLDL_Internal(m = %d, %s)\n",m,(pivot ? "pivot" : "no pivot"));
      //if (m == 8) 
      /*{
        printf( "\n A=[\n" );
        for (int i=0; i<m; i++) {
          for (int j=0; j<m; j++) printf("%.16e ",A(i,j));
          printf("\n");
        }
        printf( "];\n" );
      }*/
      bool left_look = false;
      if (left_look) {
        // no easy way to pivot..
        for (ordinal_type j=0; j < m; j+=2) {
          // factorize j:j+1 th columns
          auto Aj = Kokkos::subview(A, range_type(j,m), range_type(j,j+2));   
          if (j > 0) {
            // -----------------------------------
            // 1) update using previous columns : A(j:end, j:j+1) - L(j:end, 0:j-1) * T(0:j-1, 0:j-1) * L(j:j+1, 0:j-1)'
            //  1.1) compute W = L(j:end, 0:j-1) * T(0:j-1, 0:j-1) * L(j:j+1, 0:j-1)'
            auto Li = Kokkos::subview(A, range_type(j,j+2), range_type(0,j));   
            for (ordinal_type k=0; k < j-1; k+= 2) {
              auto T = Kokkos::subview(A, range_type(k,k+2), range_type(k,k+2));
              Wk(k,0)   = -T(1,0) * Li(0,k+1); Wk(k,1)   = -T(1,0) * Li(1,k+1); 
              Wk(k+1,0) =  T(1,0) * Li(0,k);   Wk(k+1,1) =  T(1,0) * Li(1,k);
            }
            //  1.2) A(j:end, j:j+1) -= L(j:end,0:j) * W 
            auto Lp = Kokkos::subview(A, range_type(j,m), range_type(0,j));   
            Blas<value_type>::gemm('N','N',m-j, 2, j,
                                    minus_one, Lp.data(), Lp.stride_1(),
                                               Wk.data(), Wk.stride_1(),
                                          one, Aj.data(), Aj.stride_1());
          }

          // -----------------------------------
          // 2) pick 2-by-2 pivot
          // TODO: No piv for now
          value_type piv = Aj(1,0);
          P(j) = P(j+1) = -(j+2);
          if (piv == zero) {
            TACHO_TEST_FOR_EXCEPTION(true, std::logic_error, ">> zero pivot during Skewed LDLt.");
          }

          // -----------------------------------
          // 3) scale with 2-by-2 pivot
          // 3.1) jth column
          ordinal_type mj = m-j;
          for (ordinal_type i=2; i<mj; i++) {
            Wk(i,0) = - Aj(i,1) / piv;
          }
          // 3.2) j+1 th column
          for (ordinal_type i=2; i<mj; i++) {
            Aj(i,1) = Aj(i,0) / piv;
          }
          // 3.2) copy back jth column
          for (ordinal_type i=2; i<mj; i++) {
            Aj(i,0) = Wk(i,0);
          }
        }
      } else {
        Kokkos::View<value_type **, Kokkos::LayoutLeft, typename ViewTypeA::execution_space> Ak("Ak",m,2);
        for (ordinal_type j=0; j < m; j+=2) {
          // factorize j:j+1 th columns
          auto Aj = Kokkos::subview(A, range_type(j,m), range_type(j,j+2));   

          // -----------------------------------
          // 1) pick 2-by-2 pivot
          value_type piv = Aj(1,0);
          P(j) = P(j+1) = -(j+2);
          ordinal_type mj = m-j;
          //printf( "\n === j = %d (mj = %d) ===\n",j,mj );
          if (pivot) // on/off pivot
          {
            for (ordinal_type k=3; k<mj; k+=2) {
              if (abs(piv) < abs(Aj(k,0))) {
                piv = Aj(k,0);
                P(j) = P(j+1) = -(j+k+1);
              }
            }
          }
          // check if we should use the pivot
          typedef ArithTraits<value_type> arith_traits;
          const typename arith_traits::mag_type tol(0.01);
          if (tol*abs(piv) < abs(Aj(1,0))) {
            // don't use this pivot (not big enough)
            piv = Aj(1,0);
            P(j) = P(j+1) = -(j+2);
          }
          if (piv == zero) {
            if (false) {
              piv = Aj(1,0) = arith_traits::epsilon();
              P(j) = P(j+1) = -(j+2);
            } else {
	      TACHO_TEST_FOR_EXCEPTION(true, std::logic_error, ">> zero pivot during Skewed LDLt.");
            }
          }
          if (P(j) != -(j+2)) {
            // pivot id
            //printf( " pivot : %d -> %d (%e -> %e)\n",j+1,-P(j)-1,Aj(1,0),piv );
            ordinal_type j2 = -P(j)-1;

            // row-swap (only to this and remaining columns)
            for (ordinal_type k=j; k<m; k++) {
              value_type val = A(j+1, k);
              A(j+1, k) = A(j2, k);
              A(j2, k)  = val;
            }

            // col-swap (only to this and remaining rows)
            for (ordinal_type k=j; k<m; k++) {
              value_type val = A(k, j+1);
              A(k, j+1) = A(k, j2);
              A(k, j2)  = val;
            }
            npivots ++;
          }
          //printf( " > PIVOT = %e\n",piv );

          // -----------------------------------
          // 2) scale with 2-by-2 pivot
          // 2.1) jth column
          for (ordinal_type i=2; i<mj; i++) {
            Wk(i,0) = - Aj(i,1) / piv;
          }
          // 2.2) j+1 th column
          for (ordinal_type i=2; i<mj; i++) {
            Aj(i,1) = Aj(i,0) / piv;
          }
          // 2.3) copy back jth column
          for (ordinal_type i=2; i<mj; i++) {
            Aj(i,0) = Wk(i,0);
          }

          if (j < m-2) {
            // -----------------------------------
            // 3) update using previous columns : A(j+2:end, j+1:end) - L(j+2:end, j:j+2) * T(j:j+2, j:j+2) * L(j+2:end, j:j+2)'
            auto Up = Kokkos::subview(A, range_type(j,j+2), range_type(j+2,m));   
            auto Lp = Kokkos::subview(A, range_type(j+2,m), range_type(j,j+2));   
            auto Ap = Kokkos::subview(A, range_type(j+2,m), range_type(j+2,m));   
            Blas<value_type>::gemm('N','N',mj-2, mj-2, 2,
                                    minus_one, Lp.data(), Lp.stride_1(),
                                               Up.data(), Up.stride_1(),
                                          one, Ap.data(), Ap.stride_1());
          }

          // ------------------------------------------------------------
          // expand to upper (after original used for right-look update)
          for (ordinal_type k=j+1; k<m; k++) {
            A(j, k) = -A(k, j);
          }
          for (ordinal_type k=j+2; k<m; k++) {
            A(j+1, k) = -A(k, j+1);
          }
        }
      }
    }
    return r_val;
  }

  template <typename MemberType, typename ViewTypeA, typename ViewTypeP, typename ViewTypeD>
  inline static int modify(MemberType &member, const ViewTypeA &A, const ViewTypeP &P, const ViewTypeD &D) {
    int r_val = 0;

    static constexpr bool runOnHost = run_tacho_on_host_v<typename ViewTypeA::execution_space>;

    if constexpr(runOnHost) {
      typedef typename ViewTypeA::non_const_value_type value_type;

      static_assert(ViewTypeA::rank == 2, "A is not rank 2 view.");
      static_assert(ViewTypeP::rank == 1, "P is not rank 1 view.");
      static_assert(ViewTypeD::rank == 2, "D is not rank 2 view.");

      TACHO_TEST_FOR_EXCEPTION(D.extent(0) < A.extent(0), std::runtime_error, "D extent(0) is smaller than A extent(0).");
      TACHO_TEST_FOR_EXCEPTION(D.extent(1) != 2, std::runtime_error, "D is supposed to store 2x2 blocks .");
      TACHO_TEST_FOR_EXCEPTION(P.extent(0) < 4 * A.extent(0), std::runtime_error, "P should be 4*A.extent(0) .");

      const ordinal_type m = A.extent(0);
      if (m > 0) {
        value_type *KOKKOS_RESTRICT Aptr = A.data();
        ordinal_type *KOKKOS_RESTRICT ipiv = P.data(), *KOKKOS_RESTRICT fpiv = ipiv + m, *KOKKOS_RESTRICT perm = fpiv + m,
                                   *KOKKOS_RESTRICT peri = perm + m;

        const value_type one(1), zero(0);
        for (ordinal_type i = 0; i < m; ++i)
          perm[i] = i;
        for (ordinal_type i = 0; i < m; ++i) {
          if (ipiv[i] < 0) {
            // symm pivots have been already applied to this and remaining part
            // * just extract diagonal !!
            // * apply pivoting to previous columns, if needed
            {
              // first pivot
              ipiv[i] = 0; /// invalidate this pivot
              fpiv[i] = 0;

              D(i, 0) = zero; //A(i, i);
              D(i, 1) = -A(i + 1, i); /// skew symmetric
              A(i, i) = one;
            }
            {
              // second pivot
              i++;
              const ordinal_type fla_pivot = -ipiv[i] - i - 1;
              fpiv[i] = fla_pivot;
              if (fla_pivot) {
                // apply the row-swap to the previous columns
                value_type *KOKKOS_RESTRICT src = Aptr + i;
                value_type *KOKKOS_RESTRICT tgt = src + fla_pivot;
                for (ordinal_type j = 0; j < (i - 1); ++j) {
                  const ordinal_type idx = j * m;
                  swap(src[idx], tgt[idx]);
                }
              }

              D(i, 0) = A(i, i - 1);
              D(i, 1) = zero; //A(i, i);
              A(i, i - 1) = zero;
              A(i, i) = one;
            }
          } else {
            const ordinal_type fla_pivot = ipiv[i] - i - 1;
            fpiv[i] = fla_pivot;
            if (fla_pivot) {
              value_type *src = Aptr + i;
              value_type *tgt = src + fla_pivot;
              for (ordinal_type j = 0; j < i; ++j) {
                const ordinal_type idx = j * m;
                swap(src[idx], tgt[idx]);
              }
            }
            D(i, 0) = A(i, i);
            A(i, i) = one;
          }

          /// apply pivots to perm vector
          if (fpiv[i]) {
            const ordinal_type pidx = i + fpiv[i];
            swap(perm[i], perm[pidx]);
          }
        }
        for (ordinal_type i = 0; i < m; ++i)
          peri[perm[i]] = i;
        /*{
          bool pivoted = false;
          printf("perm=[\n");
          for (ordinal_type i = 0; i < m; ++i) { printf("%d %d\n",perm[i],peri[i]); if (perm[i] != i) pivoted = true; }
          printf("];\n");
          if (pivoted) printf( " WARNING PIVOTED\n" );
        }*/
      }
    } else {
      TACHO_TEST_FOR_EXCEPTION(true, std::logic_error, ">> This function is only allowed in host space.");
    }
    return r_val;
  }
};

} // namespace Tacho

#endif
