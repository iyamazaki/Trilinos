//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"

/// KokkosKernels headers
#include "KokkosBatched_Util.hpp"

#include "Kokkos_ArithTraits.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Copy_Impl.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Team_Impl.hpp"

#include "KokkosBlas2_serial_gemv_impl.hpp"

/// cuda profile
#if defined(KOKKOS_ENABLE_CUDA)
#include "cuda_profiler_api.h"
#endif

struct KokkosSerialTag {};
struct KokkosBlasTag   {};
struct KokkosBatchTag  {};

using exec_space_type   = Kokkos::DefaultExecutionSpace;
using memory_space_type = typename exec_space_type::memory_space;
using host_space        = Kokkos::DefaultHostExecutionSpace;
using range_type        = Kokkos::pair<int, int>;

//using val_type    = double;
using val_type    = float;
using ATS = Kokkos::ArithTraits<val_type>;
using policy_type = Kokkos::TeamPolicy<exec_space_type>;
using serial_policy_type = Kokkos::RangePolicy<exec_space_type, KokkosSerialTag>;
using batch_policy_type  = Kokkos::TeamPolicy<exec_space_type,  KokkosBatchTag>;
using blas_policy_type   = Kokkos::TeamPolicy<exec_space_type,  KokkosBlasTag>;

using member_type  = typename policy_type::member_type;
using blas_member_type  = typename blas_policy_type::member_type;
using batch_member_type = typename batch_policy_type::member_type;

template <typename ManyMatrixType, typename ManyVectorType>
val_type computeResidual(const ManyMatrixType &A, const ManyVectorType &x, const ManyVectorType &b,
                         const ManyVectorType &t, const ManyVectorType &r, const int numRuns) {
  /// compute residual, t = b + (numRuns)*Ax and r = b on input
  val_type residual(0);
  {
    //policy_type policy(A.extent(0), Kokkos::AUTO());
    policy_type policy(A.extent(0), 1);
    Kokkos::parallel_reduce(
        "compute-residual", policy,
        KOKKOS_LAMBDA(const member_type &member, val_type &update) {
          const int i = member.league_rank();
          auto AA = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          auto xx = Kokkos::subview(x, i, Kokkos::ALL());
          auto tt = Kokkos::subview(t, i, Kokkos::ALL());
          auto rr = Kokkos::subview(r, i, Kokkos::ALL());
          auto bb = Kokkos::subview(b, i, Kokkos::ALL());

          val_type anrm(0);
          val_type xnrm(0);
          val_type bnrm(0);
          val_type rnrm(0);
          //for (int k=0; k<numRuns; k++) {
          //  TeamGemv<member_type, Trans::NoTranspose, Algo::Level2::Unblocked>::invoke(member, one, AA, xx, one, rr);
          //}
          if (member.team_rank() == 0)
          {
            const int m = AA.extent(0);
            const int n = AA.extent(1);
            for (int k=0; k<numRuns; k++) {
              //printf("%d:\n",k );
              for (int row=0; row<m; row++) {
                for (int col=0; col<n; col++) {
                  //if (row == 0) printf("rr(%d) = %e + %e * %e",row,rr(row),AA(row,col),xx(col));
                  rr(row) += AA(row,col) * xx(col);
                  //if (row == 0) printf(" = %e\n",rr(row));
                }
                //if (row == 0) printf("\n");
              }
            }
            for (int row=0; row<m; row++) {
              for (int col=0; col<n; col++) {
                anrm += AA(row,col)*AA(row,col);
              }
            }
            for (int col=0; col<n; col++) {
              xnrm += xx(col)*xx(col);
            }
            for (int row=0; row<m; row++) {
              bnrm += bb(row)*bb(row);
            }
            anrm = ATS::sqrt(anrm);
            xnrm = ATS::sqrt(xnrm);
            bnrm = ATS::sqrt(bnrm);
          }
          /*if (member.team_rank() == 0) {
            const int m = AA.extent(0);
            const int n = AA.extent(1);
            printf("A=[\n");
            for (int row=0; row<m; row++) {
              for (int col=0; col<n; col++) {
                printf("%.16e ",AA(row,col));
              }
              printf("\n");
            }
            printf("];\n");
            printf("xbtr=[\n");
            for (int row=0; row<m; row++) printf("%d %d %.16e %.16e %e %e\n",i,row,xx(row),bb(row),tt(row),rr(row));
            printf("];\n");
          }*/

          member.team_barrier();
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(member, rr.extent(0)),
              [&](const int &k, val_type &lsum) { lsum += (tt(k)-rr(k))*(tt(k)-rr(k)); }, rnrm);
          member.team_barrier();
          rnrm = ATS::sqrt(rnrm) / (numRuns*anrm*xnrm + bnrm);

          Kokkos::single(Kokkos::PerTeam(member), [&]() { update += rnrm; });
          //if (member.team_rank() == 0) update += sum;
          member.team_barrier();
        },
        residual);
  }
  return residual;
}


template <class VTA, class VTX, class VTB, class member_type>
struct Gemv_Task1 {
 private:
  int __n;
  VTA __A;
  VTX __x;
  VTB __b;

 public:
  Gemv_Task1(int n, VTA A, VTX x, VTB b) : __n(n), __A(A), __x(x), __b(b) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const KokkosSerialTag &, const int rank) const {
    const val_type one(1);
    for (int i = rank*__n; i < (rank+1)*__n; i++) {
      auto AA = Kokkos::subview(__A, i, Kokkos::ALL(), Kokkos::ALL());
      auto xx = Kokkos::subview(__x, i, Kokkos::ALL());
      auto bb = Kokkos::subview(__b, i, Kokkos::ALL());
      KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Level2::Unblocked>::invoke(one, AA, xx, one, bb);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const KokkosBlasTag &, const blas_member_type &member) const {
    const int rank = member.league_rank();
    const val_type one(1);
    for (int i = rank*__n; i < (rank+1)*__n; i++) {
      auto AA = Kokkos::subview(__A, i, Kokkos::ALL(), Kokkos::ALL());
      auto xx = Kokkos::subview(__x, i, Kokkos::ALL());
      auto bb = Kokkos::subview(__b, i, Kokkos::ALL());
      KokkosBlas::TeamGemv<blas_member_type, KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Level2::Unblocked>::invoke(member, one, AA, xx, one, bb);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const KokkosBatchTag &, const blas_member_type &member) const {
    const int i = member.league_rank();
    const val_type one(1);
    auto range_i = range_type(i*__n, (i+1)*__n);
    auto AA = Kokkos::subview(__A, range_i, Kokkos::ALL(), Kokkos::ALL());
    auto xx = Kokkos::subview(__x, range_i, Kokkos::ALL());
    auto bb = Kokkos::subview(__b, range_i, Kokkos::ALL());
    KokkosBatched::TeamGemv<member_type, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Level2::Unblocked>::invoke(member, one, AA, xx, one, bb);
  }
};


template <class VTA, class VTX, class VTB, class member_type>
struct Gemv_Task2 {
 private:
  int __n;
  int __numRuns;
  VTA __A;
  VTX __x;
  VTB __b;

 public:
  Gemv_Task2(int n, int numRuns, VTA A, VTX x, VTB b) : __n(n), __numRuns(numRuns), __A(A), __x(x), __b(b) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const KokkosBlasTag &, const member_type &member) const {
    const int rank = member.league_rank();
    const val_type one(1);
    for (int iter=0; iter<__numRuns; iter++) {
      for (int i = rank*__n; i < (rank+1)*__n; i++) {
        auto AA = Kokkos::subview(__A, i, Kokkos::ALL(), Kokkos::ALL());
        auto xx = Kokkos::subview(__x, i, Kokkos::ALL());
        auto bb = Kokkos::subview(__b, i, Kokkos::ALL());
        KokkosBlas::TeamGemv<member_type, KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Level2::Unblocked>::invoke(member, one, AA, xx, one, bb);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const KokkosBatchTag &, const member_type &member) const {
    const int i = member.league_rank();
    const val_type one(1);
    auto range_i = range_type(i*__n, (i+1)*__n);
    auto AA = Kokkos::subview(__A, range_i, Kokkos::ALL(), Kokkos::ALL());
    auto xx = Kokkos::subview(__x, range_i, Kokkos::ALL());
    auto bb = Kokkos::subview(__b, range_i, Kokkos::ALL());
    for (int iter=0; iter<__numRuns; iter++) {
      KokkosBatched::TeamGemv<member_type, KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Level2::Unblocked>::invoke(member, one, AA, xx, one, bb);
    }
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
#if 0 //defined(KOKKOS_ENABLE_CUDA)
    cudaProfilerStop();
#endif
    //Kokkos::print_configuration(std::cout);
    Kokkos::Timer timer;

    ///
    /// input arguments parsing
    ///
    bool check = false;
    int type = 1;          /// type of kernels
    int N1   = 128 * 128;  /// # of problems (batch size)
    int N2   = 128 * 128;  /// # of problems (batch size)
    int Ninc = 1;
    int nT = 1;            /// # of problems / team
    int Blk = 5;           /// block dimension
    int numRuns = 1000;
    for (int i = 1; i < argc; ++i) {
      const std::string &token = argv[i];
      if (token == std::string("-type"))  type = std::atoi(argv[++i]);
      if (token == std::string("-N1"))    N1   = std::atoi(argv[++i]);
      if (token == std::string("-N2"))    N2   = std::atoi(argv[++i]);
      if (token == std::string("-Ninc"))  Ninc = std::atoi(argv[++i]);
      if (token == std::string("-nT"))    nT   = std::atoi(argv[++i]);
      if (token == std::string("-B"))     Blk  = std::atoi(argv[++i]);
      if (token == std::string("-runs"))  numRuns = std::atoi(argv[++i]);
      if (token == std::string("-check")) { check = true; }
    }
    ///
    /// Problem container: rank-3 array
    ///
    /// A - multiple block matrices representing block diagonals
    /// T - temporal block matrices to store its LU factors
    /// x - solution vector
    /// b - right hand side vector
    ///
    for( int N=N1; N<=N2; N+=Ninc) {
      printf("\n :::: Testing Type %d (N = %d (# of PEs), nT = %d (# of blocks / PE), Blk = %d (size of each block)) for %d runs\n", type, N, nT, Blk,numRuns);
      Kokkos::View<val_type ***, Kokkos::LayoutRight, exec_space_type> A("block diagonals", nT*N, Blk, Blk);
      Kokkos::View<val_type **, Kokkos::LayoutRight, exec_space_type> x("x", nT*N, Blk);
      Kokkos::View<val_type **, Kokkos::LayoutRight, exec_space_type> b("b", nT*N, Blk);
      Kokkos::View<val_type **, Kokkos::LayoutRight, exec_space_type> t("r", nT*N, Blk);
      Kokkos::View<val_type **, Kokkos::LayoutRight, exec_space_type> r("r", nT*N, Blk);

      /// The block diagonal matrices are assumed to be extracted from a block
      /// sparse matrix. Here we set the blocks with random values
      Kokkos::Random_XorShift64_Pool<exec_space_type> random(13245);
      Kokkos::fill_random(A, random, val_type(1.0));
      Kokkos::fill_random(b, random, val_type(1.0));
      Kokkos::fill_random(x, random, val_type(1.0));

      /// Task 1. Use the so-called standard batch interface
      {
#if 0//defined(KOKKOS_ENABLE_CUDA)
        cudaProfilerStart();
#endif
        Kokkos::deep_copy(t, b);
        /// warm up (call once)
        if (type == 0) {
          serial_policy_type policy(0, N);
          Kokkos::parallel_for("KokkosSerial::Gemv(task1)", policy,
              Gemv_Task1<decltype(A), decltype(x), decltype(t), blas_member_type>(nT, A, x, t));
	} else if (type == 1) {
          blas_policy_type policy(N, Kokkos::AUTO());
          Kokkos::parallel_for("KokkosBlas::Gemv(task1)", policy,
              Gemv_Task1<decltype(A), decltype(x), decltype(t), blas_member_type>(nT, A, x, t));
        } else {
          batch_policy_type policy(N, Kokkos::AUTO());
          Kokkos::parallel_for("KokkosBatched::Gemv(task1)", policy,
              Gemv_Task1<decltype(A), decltype(x), decltype(t), batch_member_type>(nT, A, x, t));
        }

        /// benchmark
        Kokkos::fence();
        timer.reset();
        if (type == 0) {
          serial_policy_type policy(0, N);
          for (int k=0; k<numRuns; k++) {
            Kokkos::parallel_for("KokkosSerial::Gemv(task1)", policy,
                Gemv_Task1<decltype(A), decltype(x), decltype(t), blas_member_type>(nT, A, x, t));
          }
	} else if (type == 1) {
          blas_policy_type policy(N, Kokkos::AUTO());
          for (int k=0; k<numRuns; k++) {
            Kokkos::parallel_for("KokkosBlas::Gemv(task1)", policy,
                Gemv_Task1<decltype(A), decltype(x), decltype(t), blas_member_type>(nT, A, x, t));
          }
        } else {
          batch_policy_type policy(N, Kokkos::AUTO());
          for (int k=0; k<numRuns; k++) {
            Kokkos::parallel_for("KokkosBatched::Gemv(task1)", policy,
                Gemv_Task1<decltype(A), decltype(x), decltype(t), batch_member_type>(nT, A, x, t));
          }
        }
        Kokkos::fence();
        const double toc = timer.seconds();
        printf("\n Task 1: time = %e\n", toc/numRuns);

        /// check residual
        if (check) {
          Kokkos::deep_copy(r, b);
	  // 1+numRuns because of warmup
          const double residual = computeResidual(A, x, b, t, r, 1+numRuns);
          printf(" > task 1: residual = %e\n\n", residual);
        }
#if 0//defined(KOKKOS_ENABLE_CUDA)
        cudaProfilerStop();
#endif
      }

      /// Task 2. Compose a new batch function using kokkos batched team-level
      {
#if 0//defined(KOKKOS_ENABLE_CUDA)
        cudaProfilerStart();
#endif
        Kokkos::deep_copy(t, b);
        /// warm up (called numRuns times)
        if (type == 1) {
          blas_policy_type policy(N, Kokkos::AUTO());
          Kokkos::parallel_for("KokkosBlas::Gemv(task2)", policy,
              Gemv_Task2<decltype(A), decltype(x), decltype(t), blas_member_type>(nT, numRuns, A, x, t));
        } else {
          batch_policy_type policy(N, Kokkos::AUTO());
          Kokkos::parallel_for("KokkosBatch:Gemv(task2)", policy,
              Gemv_Task2<decltype(A), decltype(x), decltype(t), batch_member_type>(nT, numRuns, A, x, t));
        }

        /// benchmark
        Kokkos::fence();
        timer.reset();
        if (nT == 1) {
          blas_policy_type policy(N, Kokkos::AUTO());
          Kokkos::parallel_for("KokkosBlas::Gemv(task2)", policy,
              Gemv_Task2<decltype(A), decltype(x), decltype(t), blas_member_type>(nT, numRuns, A, x, t));
        } else {
          batch_policy_type policy(N, Kokkos::AUTO());
          Kokkos::parallel_for("KokkosBatch::Gemv(task2)", policy,
              Gemv_Task2<decltype(A), decltype(x), decltype(t), batch_member_type>(nT, numRuns, A, x, t));
        }
        Kokkos::fence();
        const double toc = timer.seconds();
        printf(" Task 2: time = %e\n", toc/numRuns);

        /// check residual
        if (check) {
          Kokkos::deep_copy(r, b);
          const double residual = computeResidual(A, x, b, t, r, 2*numRuns);
          printf(" > task 2: residual = %e\n\n", residual);
        }
#if 0//defined(KOKKOS_ENABLE_CUDA)
        cudaProfilerStop();
#endif
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
