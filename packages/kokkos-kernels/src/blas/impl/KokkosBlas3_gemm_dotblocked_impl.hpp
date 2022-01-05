/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_BLAS3_GEMM_DOTBLOCKED_IMPL_HPP_
#define KOKKOS_BLAS3_GEMM_DOTBLOCKED_IMPL_HPP_

namespace KokkosBlas {
namespace Impl {


// DotBasedGEMM implements the optimization for C = beta*C + alpha*A^TB 
// with A and B matrices both being tall and skinny. C matrix is assumably 
// small, so, each entry of C is computed by performing the dot product of 
// respective columns of A and B matrices. Note that the dot products are
// performed on very long vectors, so, each dot product is distributed among
// numTeamsPerBlk teams.     

// NOTE: block size defined like this??
#define BlockedGEMM_MB 1
#define BlockedGEMM_NB 3


namespace BlockedGEMM_op {

// 1-by-2 case
template<class scalar_type>
struct reduce_mb1_nb2 {

  // NOTE: block allocated like this?
  scalar_type C[2];

  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  reduce_mb1_nb2 () {
    C[0] = scalar_type (0.0);
    C[1] = scalar_type (0.0);
  }

  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  reduce_mb1_nb2 (const reduce_mb1_nb2 & src) {
    C[0] = src.C[0];
    C[1] = src.C[1];
  }

  // operator +=
  KOKKOS_INLINE_FUNCTION   // add operator
  reduce_mb1_nb2& operator += (const reduce_mb1_nb2& src) {
    C[0] += src.C[0];
    C[1] += src.C[1];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator 
  void operator += (const volatile reduce_mb1_nb2& src) volatile {
    C[0] += src.C[0];
    C[1] += src.C[1];
  }

  // operator +
  KOKKOS_INLINE_FUNCTION   // add operator
  reduce_mb1_nb2& operator + (const reduce_mb1_nb2& src) {
    C[0] += src.C[0];
    C[1] += src.C[1];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator 
  void operator + (const volatile reduce_mb1_nb2& src) volatile {
    C[0] += src.C[0];
    C[1] += src.C[1];
  }

  // operator =
  KOKKOS_INLINE_FUNCTION
  void operator = (const volatile reduce_mb1_nb2 &src) volatile {
    C[0] = src.C[0];
    C[1] = src.C[1];
  }
  KOKKOS_INLINE_FUNCTION
  reduce_mb1_nb2& operator = (const reduce_mb1_nb2 &src) {
    C[0] = src.C[0];
    C[1] = src.C[1];
    return *this;
  }
};


// 1-by-3 case
template<class scalar_type>
struct reduce_mb1_nb3 {

  // NOTE: block allocated like this?
  scalar_type C[3];

  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  reduce_mb1_nb3 () {
    C[0] = scalar_type (0.0);
    C[1] = scalar_type (0.0);
    C[2] = scalar_type (0.0);
  }

  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  reduce_mb1_nb3 (const reduce_mb1_nb3 & src) {
    C[0] = src.C[0];
    C[1] = src.C[1];
    C[2] = src.C[2];
  }

  // operator +=
  KOKKOS_INLINE_FUNCTION   // add operator
  reduce_mb1_nb3& operator += (const reduce_mb1_nb3& src) {
    C[0] += src.C[0];
    C[1] += src.C[1];
    C[2] += src.C[2];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator 
  void operator += (const volatile reduce_mb1_nb3& src) volatile {
    C[0] += src.C[0];
    C[1] += src.C[1];
    C[2] += src.C[2];
  }

  // operator +
  KOKKOS_INLINE_FUNCTION   // add operator
  reduce_mb1_nb3& operator + (const reduce_mb1_nb3& src) {
    C[0] += src.C[0];
    C[1] += src.C[1];
    C[2] += src.C[2];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator 
  void operator + (const volatile reduce_mb1_nb3& src) volatile {
    C[0] += src.C[0];
    C[1] += src.C[1];
    C[2] += src.C[2];
  }

  // operator =
  KOKKOS_INLINE_FUNCTION
  void operator = (const volatile reduce_mb1_nb3 &src) volatile {
    C[0] = src.C[0];
    C[1] = src.C[1];
    C[2] = src.C[2];
  }
  KOKKOS_INLINE_FUNCTION
  reduce_mb1_nb3& operator = (const reduce_mb1_nb3 &src) {
    C[0] = src.C[0];
    C[1] = src.C[1];
    C[2] = src.C[2];
    return *this;
  }
};


// general case
template<class scalar_type>
struct reduce {

  // NOTE: block allocated like this?
  scalar_type C[BlockedGEMM_MB][BlockedGEMM_NB];

  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  reduce () {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] = scalar_type (0.0);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  reduce (const reduce & src) {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] = src.C[i][j];
      }
    }
  }

  // operator +=
  KOKKOS_INLINE_FUNCTION   // add operator
  reduce& operator += (const reduce& src) {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] += src.C[i][j];
      }
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator 
  void operator += (const volatile reduce& src) volatile {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] += src.C[i][j];
      }
    }
  }

  // operator +
  KOKKOS_INLINE_FUNCTION   // add operator
  reduce& operator + (const reduce& src) {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] += src.C[i][j];
      }
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator 
  void operator + (const volatile reduce& src) volatile {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] += src.C[i][j];
      }
    }
  }

  // operator =
  KOKKOS_INLINE_FUNCTION
  void operator = (const volatile reduce &src) volatile {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] = src.C[i][j];
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  reduce& operator = (const reduce &src) {
    for (int i = 0; i < BlockedGEMM_MB; i++) {
      for (int j = 0; j < BlockedGEMM_NB; j++) {
        C[i][j] = src.C[i][j];
      }
    }
    return *this;
  }
};

} // BlockedGEMM_op


struct TabBlkZero{};   // The init tag for beta=0 
struct TabBlkInit{};   // The init tag for beta!=0 and beta !=1 
struct TabBlkMult{};   // The multiplication tag for transposed A
struct TabBlkMultCT{};   // The multiplication tag for conjugate-transposed A 
template<class ExecSpace, class AV, class BV, class CV>
struct DotBlockedGEMM{

  const AV A;
  const BV B;
  CV C;

  using scalar_A = typename AV::non_const_value_type;
  using size_A = typename AV::size_type;
  using scalar_C = typename CV::non_const_value_type;
  using size_C = typename CV::size_type;
  using AVT = Kokkos::Details::ArithTraits<scalar_A>;
  using CVT = Kokkos::Details::ArithTraits<scalar_C>;
  using range_type = Kokkos::pair<int, int>;

  const scalar_A alpha;
  const scalar_C beta;

  const bool symmetric;
  const size_C M;           
  const size_C N;

  size_C mx; // block size
  size_C nx; // block size
  size_C mb; // num of blocks in row of C
  size_C nb; // num of blocks in col of C

  size_C numTeamsPerBlk;   // number of teams collectively performing a dot product
  size_C numTeams;       // total number of teams
  
  const size_A K;  // the length of the vectors in the dot products
  size_A chunkSize;      // the local length of each team's share on the dot product  
  

  DotBlockedGEMM(const scalar_A& alpha_, const AV& A_, const BV& B_, const scalar_C& beta_, const CV& C_) :
  A(A_), B(B_), C(C_), alpha(alpha_), beta(beta_), symmetric(false),
  M(C.extent(0)), N(C.extent(1)), K(A.extent(0))
  { 
    mx = BlockedGEMM_MB; // block size
    nx = BlockedGEMM_NB; // block size
    mb = (M+mx-1)/mx; // num of blocks in row of C
    nb = (N+nx-1)/nx; // num of blocks in col of C
  }

  DotBlockedGEMM(const scalar_A& alpha_, const AV& A_, const BV& B_, const scalar_C& beta_, const CV& C_, const bool symmetric_) :
  A(A_), B(B_), C(C_), alpha(alpha_), beta(beta_), symmetric(symmetric_),
  M(C.extent(0)), N(C.extent(1)), K(A.extent(0))
  { 
    mx = BlockedGEMM_MB; // block size
    nx = BlockedGEMM_NB; // block size
    mb = (M+mx-1)/mx; // num of blocks in row of C
    nb = (N+nx-1)/nx; // num of blocks in col of C
  }

  void run(bool conjugateTranspose) {

    // NOTE: these workPerTeam and approxNumTeams were used for TPL CUBLAS,
    //       and may need to be retuned for other architectures
    constexpr size_C workPerTeam = 4096;     // Amount of work per team
    const size_C nblks = mb*nb;              // Total number of blocks in C
    #if 1
    size_C appxNumTeams = (K * M * N) / workPerTeam; // Estimation for appxNumTeams
    #else
    size_C appxNumTeams = (K * nblks) / workPerTeam; // Estimation for appxNumTeams
    appxNumTeams *= (mx*nx);
    #endif

    // Adjust appxNumTeams in case it is too small or too large
    if(appxNumTeams < 1)
      appxNumTeams = 1;
    if(appxNumTeams > 1024)
      appxNumTeams = 1024;

    #if 1
    // If there are more dot products than the number of teams,
    // then set the number of teams to be number of dot products
    // and each team will perform only one dot product.
    // We don't want a team to perform more than one dot product.
    if(nblks >= appxNumTeams) {
      numTeams = nblks;
      numTeamsPerBlk = 1;
    }
    // If there are more teams than dot products, each dot product can
    // potentially be performed by multiple teams. First, compute 
    // numTeamsPerBlk as an integer (take the floor, not ceiling), then,
    // compute actual number of teams by using this factor.
    else {
      numTeamsPerBlk = appxNumTeams / nblks; // teams / blk
      numTeams = nblks * numTeamsPerBlk;     // update num of teams
    }
    #else
    numTeams = nblks;
    numTeamsPerBlk = 1;
    #endif

    // Determine the local length for the dot product
    chunkSize = K / numTeamsPerBlk;
    if(numTeamsPerBlk > 1)
      chunkSize++;

    // Initialize C matrix if beta != 1
    if(beta == CVT::zero()) {
      Kokkos::MDRangePolicy<TabBlkZero, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {M, N});
      Kokkos::parallel_for("Initialize C for Dot Product Based GEMM", policyInit, *this);
    }
    else if(beta != CVT::one()) {
      Kokkos::MDRangePolicy<TabBlkInit, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {M, N});
      Kokkos::parallel_for("Initialize C for Dot Product Based GEMM", policyInit, *this);
    }
    
    // Multiply alpha*A^TB and add it to beta*C
    if(conjugateTranspose) {
      Kokkos::TeamPolicy<TabBlkMultCT, ExecSpace> policyMult(numTeams, Kokkos::AUTO);
      Kokkos::parallel_for("Perform Block Based GEMM", policyMult, *this);
    }
    else{
      Kokkos::TeamPolicy<TabBlkMult, ExecSpace> policyMult(numTeams, Kokkos::AUTO);
      Kokkos::parallel_for("Perform Block Based GEMM", policyMult, *this);
    }
    /*auto hostC = Kokkos::create_mirror_view(C);
    std::cout << std::endl << "C = [" << std::endl;
    for (int ii=0; ii < (int)M; ii++) {
      std::cout << " ";
      for (int jj=0; jj < (int)N; jj++) std::cout << hostC(ii,jj) << " ";
      std::cout << std::endl;
    }
    std::cout << "];" << std::endl << std::endl;*/
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TabBlkZero&, const size_C &rowId, const size_C &colId ) const {
    C(rowId, colId) = CVT::zero(); 
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TabBlkInit&, const size_C &rowId, const size_C &colId ) const {
    C(rowId, colId) = beta * C(rowId, colId);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TabBlkMult&, const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) const {
    // NOTE: numTeams = nblks * numTeamsPerBlk, where nblks = mb * nb
    const size_C globalRank = teamMember.league_rank();
    const size_C localRank = globalRank % numTeamsPerBlk; // which chunk ?
    const size_C id        = globalRank / numTeamsPerBlk; // which team within the chunk ?
    const size_C i = id / nb; // block row id
    const size_C j = id % nb; // block col id

    const size_A i1 = i*mx;                     // start row of Cij
    const size_A i2 = (i1+mx > M ? M : i1+mx);  // last  row of Cij

    const size_A j1 = (symmetric ? i1 : j*nx);  // start col of Cij
    const size_A j2 = (j1+nx > N ? N : j1+nx);  // last  col of Cij

    // partial dots with the team
    if (i2 - i1 == 1 && j2 - j1 == 1) {
      // == 1-by-1 case ==
      scalar_C result = scalar_C(0.0);
      const size_A baseInd = chunkSize*localRank; // offset into A & B
      Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, scalar_C &update ) {
        if(baseInd + k < K) {
          update += alpha * A(baseInd+k, i1) * B(baseInd+k, j1);
        }
      }, result );

      // atomic add among the teams
      Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
        Kokkos::atomic_add(&C(i1, j1), result);
      });
    } else if (i2 - i1 == 1 && j2 - j1 == 2) {
      // == 1-by-2 case ==
      using reducer_mb1_nb2_type = typename BlockedGEMM_op::reduce_mb1_nb2<scalar_C>;
      reducer_mb1_nb2_type result;
      const size_A baseInd = chunkSize*localRank; // offset into A & B
      Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, reducer_mb1_nb2_type &update ) {
        if(baseInd + k < K) {
          update.C[0] += alpha * A(baseInd+k, i1) * B(baseInd+k, j1);
          update.C[1] += alpha * A(baseInd+k, i1) * B(baseInd+k, j1+1);
        }
      }, Kokkos::Sum<reducer_mb1_nb2_type> (result) );

      // atomic add among the teams
      Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
        Kokkos::atomic_add(&C(i1, j1),   result.C[0]);
        Kokkos::atomic_add(&C(i1, j1+1), result.C[1]);
      });
    } else if (i2 - i1 == 1 && j2 - j1 == 3) {
      // == 1-by-3 case ==
      using reducer_mb1_nb3_type = typename BlockedGEMM_op::reduce_mb1_nb3<scalar_C>;
      reducer_mb1_nb3_type result;
      const size_A baseInd = chunkSize*localRank; // offset into A & B
      Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, reducer_mb1_nb3_type &update ) {
        if(baseInd + k < K) {
          update.C[0] += alpha * A(baseInd+k, i1) * B(baseInd+k, j1);
          update.C[1] += alpha * A(baseInd+k, i1) * B(baseInd+k, j1+1);
          update.C[2] += alpha * A(baseInd+k, i1) * B(baseInd+k, j1+2);
        }
      }, Kokkos::Sum<reducer_mb1_nb3_type> (result) );

      // atomic add among the teams
      #if 1
      using execution_space = typename CV::execution_space;
      using memory_space = typename execution_space::memory_space;
      #if 0
       Kokkos::View<scalar_C**, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic> > Catomic(C.data(), M, N);
       Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
         Catomic(i1, j1)   += result.C[0];
         Catomic(i1, j1+1) += result.C[1];
         Catomic(i1, j1+2) += result.C[2];
       });
      #else
       Kokkos::View<scalar_C*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic> > c11(&C(i1, j1),   1);
       Kokkos::View<scalar_C*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic> > c12(&C(i1, j1+1), 1);
       Kokkos::View<scalar_C*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic> > c13(&C(i1, j1+2), 1);
       Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
         c11(0) += result.C[0];
         c12(0) += result.C[1];
         c13(0) += result.C[2];
       });
      #endif
      #else
      Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
        Kokkos::atomic_add(&C(i1, j1),   result.C[0]);
        Kokkos::atomic_add(&C(i1, j1+1), result.C[1]);
        Kokkos::atomic_add(&C(i1, j1+2), result.C[2]);
      });
      #endif
    } else {
      using reducer_type = typename BlockedGEMM_op::reduce<scalar_C>;
      reducer_type result;
      const size_A baseInd = chunkSize*localRank; // offset into A & B
      Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, reducer_type &update ) {
        if(baseInd + k < K) {
          for (size_C ii = i1; ii < i2; ii++) {
            for (size_C jj = j1; jj < j2; jj++) {
              update.C[ii-i1][jj-j1] += alpha * A(baseInd+k, ii) * B(baseInd+k, jj);
            }
          }
        }
      }, Kokkos::Sum<reducer_type> (result) );

      // atomic add among the teams
      if (i2 > i1 && j2 > j1) {
        Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
          for (size_C ii = i1; ii < i2; ii++) {
            for (size_C jj = j1; jj < j2; jj++) {
              Kokkos::atomic_add(&C(ii, jj), result.C[ii-i1][jj-j1]);
            }
          }
        });
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TabBlkMultCT&, const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) const {

    // NOTE: numTeams = nblks * numTeamsPerBlk, where nblks = mb * nb
    const size_C globalRank = teamMember.league_rank();
    const size_C localRank = globalRank % numTeamsPerBlk; // which chunk ?
    const size_C id        = globalRank / numTeamsPerBlk; // which team within the chunk ?
    const size_C i = id / nb; // block row id
    const size_C j = id % nb; // block col id
    
    const size_A i1 = i*mx;                     // start row of Cij
    const size_A i2 = (i1+mx > M ? M : i1+mx);  // last  row of Cij

    const size_A j1 = j*nx;                     // start col of Cij
    const size_A j2 = (j1+nx > N ? N : j1+nx);  // last  col of Cij

    // partial dots with the team
    using reducer_type = BlockedGEMM_op::reduce<scalar_C>;
    reducer_type result;
    const size_A baseInd = chunkSize*localRank; // offset into A & B
    Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, reducer_type &update ) {
      if(baseInd + k < K) {
        for (size_C ii = i1; ii < i2; ii++) {
          for (size_C jj = j1; jj < j2; jj++) {
            update.C[ii-i1][jj-j1] += alpha * AVT::conj(A(baseInd+k, ii)) * B(baseInd+k, jj);
          }
        }
      }
    }, Kokkos::Sum<reducer_type> (result) );

    // atomic add among the teams
    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
      for (size_C ii = i1; ii < i2; ii++) {
        for (size_C jj = j1; jj < j2; jj++) {
          Kokkos::atomic_add(&C(ii, jj), result.C[ii-i1][jj-j1]);
        }
      }
    });
  }

};

}
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<class scalar_type>
   struct reduction_identity< KokkosBlas::Impl::BlockedGEMM_op::reduce_mb1_nb2<scalar_type> > {
      KOKKOS_FORCEINLINE_FUNCTION static KokkosBlas::Impl::BlockedGEMM_op::reduce_mb1_nb2<scalar_type> sum() {
         return KokkosBlas::Impl::BlockedGEMM_op::reduce_mb1_nb2<scalar_type> ();
      }
   };
   template<class scalar_type>
   struct reduction_identity< KokkosBlas::Impl::BlockedGEMM_op::reduce_mb1_nb3<scalar_type> > {
      KOKKOS_FORCEINLINE_FUNCTION static KokkosBlas::Impl::BlockedGEMM_op::reduce_mb1_nb3<scalar_type> sum() {
         return KokkosBlas::Impl::BlockedGEMM_op::reduce_mb1_nb3<scalar_type> ();
      }
   };
   template<class scalar_type>
   struct reduction_identity< KokkosBlas::Impl::BlockedGEMM_op::reduce<scalar_type> > {
      KOKKOS_FORCEINLINE_FUNCTION static KokkosBlas::Impl::BlockedGEMM_op::reduce<scalar_type> sum() {
         return KokkosBlas::Impl::BlockedGEMM_op::reduce<scalar_type> ();
      }
   };
}

#endif
