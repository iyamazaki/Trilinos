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
                                           const ViewTypeW &W) {
    typedef typename ViewTypeA::non_const_value_type value_type;
    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
    const value_type one(1), minus_one(-1), zero(0);

    static_assert(ViewTypeA::rank == 2, "A is not rank 2 view.");
    static_assert(ViewTypeP::rank == 1, "P is not rank 1 view.");
    static_assert(ViewTypeW::rank == 1, "W is not rank 1 view.");

    TACHO_TEST_FOR_ABORT(P.extent(0) < 4 * A.extent(0), "P should be 4*A.extent(0) .");

    int r_val(0);
    const ordinal_type m = A.extent(0);

    if (m > 0) {
      /// factorize LDL
      // TODO: move this to symbolic init
      Kokkos::View<value_type **, Kokkos::LayoutLeft, typename ViewTypeA::execution_space> Wk("WRK",m,2);
      //printf( "\ m = %d => (%d,%d)\n",m,Wk.stride_0(),Wk.stride_1() );
      #if 1
      /*printf("A=[\n");
      for (ordinal_type i=0; i < m; i++) {
        for (ordinal_type j=0; j < m; j++) {
          printf( "%e ",A(i,j) );
        }
        printf("\n");
      }
      printf("];\n");*/
      for (ordinal_type j=0; j < m; j+=2) {
        //printf( " == j = %d/%d ==\n",j,m );
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
            //printf( " %e * %e, %e * %e = %e, %e\n",-T(0,1),Li(0,k+1),-T(0,1),Li(1,k+1),Wk(k,0),Wk(k,1) );
            //printf( " %e * %e, %e * %e = %e, %e\n\n",T(1,0),Li(0,k),T(1,0),Li(1,k),Wk(k+1,0),Wk(k+1,1) );
          }
          //  1.2) A(j:end, j:j+1) -= L(j:end, 
          auto Lp = Kokkos::subview(A, range_type(j,m), range_type(0,j));   
          /*printf( " GEMM(%d,%d,%d : %d,%d,%d\n",m-j,2,j, Lp.stride_1(),Wk.stride_1(),Aj.stride_1());
          printf(" Aj=[\n" );
          for (int i=0; i<m-j; i++) {
            printf("%e %e\n",Aj(i,0),Aj(i,1));
          }
          printf("];\n" );
          printf(" Lp=[\n" );
          for (int i=0; i<m-j; i++) {
            for (int k=0; k<j; k++) {
              printf("%e ",Lp(i,k));
            }
            printf("\n");
          }
          printf("];\n" );
          printf(" Wk=[\n" );
          for (int i=0; i<j; i++) {
            printf("%e %e\n",Wk(i,0),Wk(i,1));
          }
          printf("];\n" );*/
          Blas<value_type>::gemm('N','N',m-j, 2, j,
                                  minus_one, Lp.data(), Lp.stride_1(),
                                             Wk.data(), Wk.stride_1(),
                                        one, Aj.data(), Aj.stride_1());
          /*printf(" => [\n" );
          for (int i=0; i<m-j; i++) {
            printf("%e %e\n",Aj(i,0),Aj(i,1));
          }
          printf("];\n" );*/
        }

        // -----------------------------------
        // 2) pick 2-by-2 pivot
        // TODO: No piv for now
        value_type piv = Aj(1,0);
        P(j) = P(j+1) = -(j+2);
        if (piv == zero) {
          TACHO_TEST_FOR_ABORT(true, ">> zero pivot during Skewed LDLt.");
        }

        // -----------------------------------
        // 3) scale with 2-by-2 pivot
        // 3.1) jth column
        ordinal_type mj = m-j;
        for (ordinal_type i=2; i<mj; i++) {
          Wk(i,0) = - Aj(i,1) / piv;
          //printf( "A(%d,%d) = %e / %e = %e\n",i,j,Aj(i,1),-piv,Wk(i,0) );
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
      /*printf("L=[\n");
      for (ordinal_type i=0; i < m; i++) {
        for (ordinal_type j=0; j < m; j++) {
          printf( "%e ",A(i,j) );
        }
        printf("\n");
      }
      printf("];\n");*/
      #else
      //Lapack<value_type>::sytrf('L', m, A.data(), A.stride_1(), P.data(), W.data(), W.extent(0), &r_val);
      #endif
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
            {
              // first pivot
              ipiv[i] = 0; /// invalidate this pivot
              fpiv[i] = 0;

              D(i, 0) = zero; //A(i, i);
              D(i, 1) = -A(i + 1, i); /// skew symmetric
              A(i, i) = one;
              //printf( " A(%d,%d) = %e\n",i,i,A(i,i) );
            }
            {
              // second pivot
              i++;
              const ordinal_type fla_pivot = -ipiv[i] - i - 1;
              fpiv[i] = fla_pivot;
              //printf( " ipiv[%d] = %d -> %d\n",i,ipiv[i],fla_pivot );
              if (fla_pivot) {
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
              //printf( " A(%d,%d) = %e\n",i,i,A(i,i) );
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
      }
    } else {
      TACHO_TEST_FOR_ABORT(true, ">> This function is only allowed in host space.");
    }
    return r_val;
  }
};

} // namespace Tacho

#endif
