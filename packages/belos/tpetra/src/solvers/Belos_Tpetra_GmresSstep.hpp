//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//@HEADER

#ifndef BELOS_TPETRA_GMRES_SSTEP_HPP
#define BELOS_TPETRA_GMRES_SSTEP_HPP

#include "Belos_Tpetra_Gmres.hpp"
#include "Belos_Tpetra_UpdateNewton.hpp"

#include "KokkosBlas3_trsm.hpp"

namespace BelosTpetra {
namespace Impl {

//
// Cholesky-QR
template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<>,
         class OP = Tpetra::Operator<> >
class CholQR {
private:
  using LO = typename MV::local_ordinal_type;
  using blas_type = Teuchos::BLAS<LO, SC>;
  using lapack_type = Teuchos::LAPACK<LO, SC>;

  bool useSVQR;
  int numReFacto;

public:
  /// \typedef FactorOutput
  /// \brief Return value of \c factor().
  ///
  /// Here, FactorOutput is just a minimal object whose value is
  /// irrelevant, so that this class' interface looks like that of
  /// \c CholQR.
  using FactorOutput = int;
  using STS = Teuchos::ScalarTraits<SC>;
  using MVT = Belos::MultiVecTraits<SC, MV>;
  using mag_type = typename STS::magnitudeType;
  using STM = Teuchos::ScalarTraits<mag_type>;
  using dense_matrix_type = Teuchos::SerialDenseMatrix<LO, SC>;
  using dense_vector_type = Teuchos::SerialDenseVector<LO, SC>;

  /// \brief Default constructor
  ///
  CholQR () :
  numReFacto (0),
  useSVQR (false)
  {}

  /// \brief Constructor
  ///
  CholQR (bool useSVQR_) :
  numReFacto (0),
  useSVQR (useSVQR_)
  {}

  /// \brief Return number of reecursive CholQR
  int
  getNumReFacto () {
    return numReFacto;
  }

  /// \brief Compute the QR factorization of the matrix A.
  ///
  /// Compute the QR factorization of the nrows by ncols matrix A,
  /// with nrows >= ncols, stored either in column-major order (the
  /// default) or as contiguous column-major cache blocks, with
  /// leading dimension lda >= nrows.
  FactorOutput
  factor (Teuchos::FancyOStream* outPtr, MV& A, dense_matrix_type& R)
  {
    Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("CholQR::factor");
    Teuchos::TimeMonitor LocalTimer (*factorTimer);

    blas_type blas;
    lapack_type lapack;

    const SC zero = STS::zero ();
    const SC one  = STS::one ();

    // quick return
    size_t ncols = A.getNumVectors ();
    size_t nrows = A.getLocalLength ();
    if (ncols == 0 || nrows == 0) {
      return 0;
    }

    // Compute R := A^T * A, using a single BLAS call.
    // MV with "static" memory (e.g., Tpetra manages the static GPU memory pool)
    MV R_mv = makeStaticLocalMultiVector (A, ncols, ncols);
    //R_mv.putScalar (STS::zero ());

    // compute R := A^T * A
    R_mv.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, one, A, A, zero);

    // Compute the Cholesky factorization of R in place, so that
    // A^T * A = R^T * R, where R is ncols by ncols upper
    // triangular.
    int info = 0;
    {
      auto R_h = R_mv.getLocalViewHost (Tpetra::Access::ReadWrite);
      int ldr = int (R_h.extent (0));
      SC *Rdata = reinterpret_cast<SC*> (R_h.data ());
      if (useSVQR) {
        dense_vector_type S (ncols);
        dense_matrix_type VT (ncols, ncols);
        SC *Sdata = S.values();
        SC *VTdata = VT.values();

        // figure out workspace size
        SC temp;
        int ione = 1;
        int ldvt = int (VT.stride());
        lapack.GESVD('N', 'A', ncols, ncols, Rdata, ldr, Sdata, &temp, 1, 
                     VTdata, ldvt, &temp, -ione, &temp, &info);
        int lwork = int (temp);
        lapack.GEQRF(ncols, ncols, Rdata, ldr, &temp, &temp, -ione, &info);
        lwork = (lwork > int (temp) ? lwork : int (temp));

        // diagonal scaling
        dense_vector_type D (ncols);
        bool diag_scale = true;
        if (diag_scale) {
          for (size_t i=0; i<ncols; i++) {
            D(i) = std::sqrt(R_h(i,i));
            for (size_t j=0; j<ncols; j++) {
              R_h(i,j) /= D(i);
              R_h(j,i) /= D(i);
            }
          }
        }

        // compute SVD
        dense_vector_type W (lwork);
        SC *Wdata = W.values();
        lapack.GESVD('N', 'A', ncols, ncols, Rdata, ldr, Sdata, &temp, 1,
                     VTdata, ldvt, Wdata, lwork, &temp, &info);
        if (info != 0) std::cout << " ERROR : GESVD returned info = " << info << std::endl;
        for (size_t j=0; j<ncols; j++) {
          for (size_t i=0; i<ncols; i++) {
            R_h(i,j) = std::sqrt(S(i)) * VT(i, j);
          }
        }
        // compute QR
        dense_vector_type TAU (ncols);
        SC *TAUdata = TAU.values ();
        lapack.GEQRF(ncols, ncols, Rdata, ldr, TAUdata, Wdata, lwork, &info);
        // zero-out lower-triangular part, and make sure positive diagonals
        for (size_t i=0; i<ncols; i++) {
          // zero-out lower-triangular part
          for (size_t j=0; j<i; j++) {
            R_h(i, j) = zero;
          }
          // make sure positive diagonals
          if (R_h(i,i) < zero) {
            for (size_t j=i; j<ncols; j++) {
              R_h(i,j) = -R_h(i, j);
            }
          }
          // apply-back diagonal scaling
          if (diag_scale) {
            for (size_t j=i; j<ncols; j++) {
              R_h(i,j) *= D(j);
            }
          }
        }
      } else {
        lapack.POTRF ('U', ncols, Rdata, ldr, &info);
      }
      if (info > 0) {
        // FIXME (mfh 17 Sep 2018) Don't throw; report an error code.
        //ncols = info;
        //throw std::runtime_error("Cholesky factorization failed");
        *outPtr << "  >  POTRF( " << ncols << " ) failed with info = " << info << std::endl;
        for (size_t i=info-1; i<ncols; i++) {
          R_h(i, i) = one;
          for (size_t j=i+1; j<ncols; j++) {
            R_h(i, j) = zero;
          }
        }
      }
    }
    // Copy to the output R
    Tpetra::deep_copy (R, R_mv);
    // TODO: replace with a routine to zero out lower-triangular
    //     : not needed, but done for testing
    for (size_t i=0; i<ncols; i++) {
      for (size_t j=0; j<i; j++) {
        R(i, j) = zero;
      }
    }

    // Compute A := A * R^{-1}.  We do this in place in A, using
    // BLAS' TRSM with the R factor (form POTRF) stored in the upper
    // triangle of R.

    // Compute A_cur / R (Matlab notation for A_cur * R^{-1}) in place.
    auto A_d = A.getLocalViewDevice (Tpetra::Access::ReadWrite);
    auto R_d = R_mv.getLocalViewDevice (Tpetra::Access::ReadOnly);
    KokkosBlas::trsm ("R", "U", "N", "N",
                      one, R_d, A_d);
    return (info > 0 ? info-1 : ncols);
  }

  // recursive call to factor
  FactorOutput
  reFactor (Teuchos::FancyOStream* outPtr, MV& A, dense_matrix_type& R)
  {
    int ncols = int (A.getNumVectors ());
    int rank = 0;
    int old_rank = -1;

    // recursively call factor while cols remaining and has made progress
    // note: When Chol fails, CholQR puts identity on the remaining submatrix
    //                        and performs TRSM. Hence, the remaining columns
    //                        are orthogonalized against the columns, for which
    //                        Chol succeeded.
    // TODO: one more sweep to orthogonalize remaining against the success?
    numReFacto = 0;
    while (rank < ncols && old_rank != rank) {
      Teuchos::Range1D next_index(rank, ncols-1);
      MV nextA = * (A.subView(next_index));

      dense_matrix_type nextR (Teuchos::View, R, ncols-rank, ncols-rank, rank, rank);
      old_rank = rank;
      auto new_rank = factor (outPtr, nextA, nextR);
      if (outPtr != nullptr) {
        if (rank > 0) {
          numReFacto ++;
          *outPtr << "  ++ reCholQR(";
        } else {
          *outPtr << "  >>   CholQR(";
        }
        *outPtr << rank << ":" << ncols-1 << "), new_rank = " << new_rank << std::endl;
      }
      rank += new_rank;
    }
    return rank;
  }
};


//
// Low-synch block Gram-Schmidt
#if 1
template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<>,
         class OP = Tpetra::Operator<> >
class LowSynchBGS {
private:
  using LO = typename MV::local_ordinal_type;
  using blas_type = Teuchos::BLAS<LO, SC>;
  using lapack_type = Teuchos::LAPACK<LO, SC>;

public:
  using FactorOutput = int;
  using STS = Teuchos::ScalarTraits<SC>;
  using MVT = Belos::MultiVecTraits<SC, MV>;
  using mag_type = typename STS::magnitudeType;
  using STM = Teuchos::ScalarTraits<mag_type>;
  using dense_matrix_type = Teuchos::SerialDenseMatrix<LO, SC>;
  using dense_vector_type = Teuchos::SerialDenseVector<LO, SC>;

  /// \brief Default constructor
  ///
  LowSynchBGS () :
  useCholQR2_ (false),
  orthoType_("MGS LowSynch")
  {}

  /// \brief Constructor
  ///
  LowSynchBGS (bool useCholQR2, std::string orthoType) :
  useCholQR2_ (useCholQR2),
  orthoType_(orthoType)
  {}

  /// \brief Compute the QR factorization of the matrix A.
  ///
  /// Compute the QR factorization of the nrows by ncols matrix A,
  /// with nrows >= ncols, stored either in column-major order (the
  /// default) or as contiguous column-major cache blocks, with
  /// leading dimension lda >= nrows.
  FactorOutput
  factor (Teuchos::FancyOStream* outPtr, MV& Q,
          dense_matrix_type& C,
          dense_matrix_type& G,
          dense_matrix_type& G2,
          int iiter, int step, int prevStep, int numIters)
  {
    blas_type blas;
    lapack_type lapack;

    const SC zero = STS::zero ();
    const SC one  = STS::one ();

    int delayed_rank = -1;
    int current_rank = -1;
    if (iiter > 0) {
      //
      // dot-product for re-normalization, and single-reduce orthogonalization
      // vector to be orthogonalized, and the vectors to be lagged-normalized
      int prevIter = iiter-prevStep;
      Teuchos::Range1D index_next (prevIter, iiter+step);
      MV Qnext = * (Q.subView (index_next));

      // vectors to be orthogonalized against
      Teuchos::Range1D index(0, iiter+step);
      Teuchos::RCP< const MV > Qi = MVT::CloneView (Q, index);

      // compute coefficient, C(:,iiter-stepSize:iiter+step) = Q(:,0:iiter+step)'*Q(iiter-stepSize:iiter+step)
      Teuchos::RCP< dense_matrix_type > c
        = Teuchos::rcp (new dense_matrix_type (Teuchos::View, C, iiter+step+1, prevStep+step+1, 0, prevIter));
      {
        MVT::MvTransMv (one, *Qi, Qnext, *c);
      }

      // re-normalize the previous s-step set of vectors (lagged)
      if (useCholQR2_) {
        delayed_rank = reNormalizeCholQR2 (prevIter, prevStep, step, Q, C, G, G2);
        // save G for convregence check
        for (int j = 0; j <= prevStep; j++) {
          for (int i = 0; i <= prevIter+prevStep; i++) {
            G2(i,j) = G(i,j);
          }
        }
      }

      // orthogonalize the new vectors against the previous columns
      {
        current_rank = projectAndNormalizeCholQR2 (numIters,
                                                   iiter, prevStep, step, Q, C, G);
      }
    } else {
      // orthogonalize
      //rank = normalizeCholQR (iiter, step, Q, G);
    }
    return (1 ? delayed_rank : current_rank);
  }

  int
  projectAndNormalizeCholQR2 (const int numIters,
                              const int iter,
                              const int stepSize,
                              const int step,
                              MV& Q,
                              dense_matrix_type& C, // store aggregated coefficient, results of block dot-products
                              dense_matrix_type& G) // Hessenburg matrix
  {
    const SC one  = STS::one  ();
    const SC zero = STS::zero ();

    Teuchos::BLAS<LO, SC> blas;
    Teuchos::LAPACK<LO, SC> lapack;

    int rank = 0;
    int prevIter = iter-stepSize;

    Teuchos::RCP< Teuchos::Time >  BOrthTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep Bort chol2");
    Teuchos::RCP< Teuchos::Time >   TsqrTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep TSQR chol2");
    Teuchos::RCP< Teuchos::Time >   trsmTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep trsm chol2");
    Teuchos::RCP< Teuchos::Time > updateTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep gemm chol2");
    if (iter > 0) {
      // extract new coefficients (note: C(:,iter) is used for T(:,iter) and G(:,0))
      for (int i = 0; i < iter+step+1; i++) {
        for (int j = 0; j < step+1; j++) {
          G(i, j) = C(i, iter+j);
        }
      }

      dense_matrix_type Gnew (Teuchos::View, G, iter, step+1, 0, 0);
      dense_matrix_type Cnew (iter, step+1, true);
      {
        // making a local copy (original)
        // C(:,iter) is used for T(:,iter) and G(:,0)
        for (int i = 0; i < iter; i++) {
          for (int j = 0; j < step+1; j++) {
            Cnew(i, j) = C(i, iter+j);
          }
        }

        // T(n, n) /= T(n, n);
        // T(n, n) -= one; // T = Q'*Q - I
        #if 0
        for (int i=0; i < stepSize+1; i++) {
          for (int j=0; j < stepSize+1; j++) {
            C(prevIter+i, prevIter+j) = zero;
          }
        }
        #else
        for (int i=0; i < stepSize; i++) {
          //C(prevIter+i, prevIter+i) = zero;
          C(prevIter+i, prevIter+i) -= one;
        }
        #endif

        // expand T
        for (int j=prevIter; j <= iter; j++) {
          for (int i=0; i < prevIter; i++) C(j, i) = C(i, j);
        }

        #if 0
        mag_type maxT = std::abs(C(0, 0));
        for (int i = 0; i < iter; i++) {
          for (int j = 0; j < iter; j++) {
            //printf( "%.2e ",C(i,j));
            maxT = std::max(maxT, std::abs(C(i, j)));
          }
          //printf( "\n" );
        }
        std::cout << "max:" << numIters << " " << maxT << std::endl;
        #endif

        // update H
        if (orthoType_ == "MGS LowSynch") {
            // H := (I+L)^(-1)H
            blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI,
                       Teuchos::NO_TRANS, Teuchos::UNIT_DIAG,
                       iter, step+1,
                       one, C.values(), C.stride(),
                            Gnew.values(), Gnew.stride());
        } else if (orthoType_ == "CGS2 LowSynch" ) {
            // H := (I-T)H
            blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                      iter, step+1, iter,
                     -one, C.values(), C.stride(),
                           Cnew.values(), Cnew.stride(),
                      one, Gnew.values(), Gnew.stride());
        }
      }

      // ----------------------------------------------------------
      // orthogonalize the new vectors against the previous columns
      // ----------------------------------------------------------
      Teuchos::Range1D index_new(iter, iter+step);
      MV Qnew = * (Q.subView(index_new));

      Teuchos::Range1D index_prev(0, iter-1);
      Teuchos::RCP< const MV > Qprev = MVT::CloneView(Q, index_prev);
      {
        Teuchos::TimeMonitor LocalTimer (*BOrthTimer);
        MVT::MvTimesMatAddMv(-one, *Qprev, Gnew, one, Qnew);
      }

      // the scaling factor
      dense_matrix_type Rnew (Teuchos::View, G, step+1, step+1, iter, 0);
      {
        // fix the coefficients
        // H-=R*P+P*R-P*(T+I)*P
        dense_matrix_type Ctmp (iter, step+1, true);
        // P+T*P
        for (int i = 0; i < iter; i++) {
          for (int j = 0; j < step+1; j++) {
            Ctmp(i, j) = Gnew(i, j);
          }
        }
        blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                  iter, step+1, iter,
                  one, C.values(), C.stride(),
                       Gnew.values(), Gnew.stride(),
                  one, Ctmp.values(), Ctmp.stride());
        // H = H+P'*(T*P)
        blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                  step+1, step+1, iter,
                  one, Gnew.values(), Gnew.stride(),
                       Ctmp.values(), Ctmp.stride(),
                  one, Rnew.values(), Rnew.stride());
        // H = H - R*P - P*R
        #if 0
        const SC two  = one+one;
        blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                  step+1, step+1, iter,
                 -two, Gnew.values(), Gnew.stride(),
                       Gnew.values(), Gnew.stride(),
                  one, Rnew.values(), Rnew.stride());
        #else
        blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                  step+1, step+1, iter,
                 -one, Gnew.values(), Gnew.stride(),
                       Cnew.values(), Cnew.stride(),
                  one, Rnew.values(), Rnew.stride());
        blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                  step+1, step+1, iter,
                 -one, Cnew.values(), Cnew.stride(),
                       Gnew.values(), Gnew.stride(),
                  one, Rnew.values(), Rnew.stride());
        #endif
      }

      // -------------------------
      // normalize the new vectors
      // -------------------------
      #if 0
      if (! STS::isComplex) {
        using real_type = typename STS::magnitudeType;
        using real_vector_type = Teuchos::SerialDenseVector<LO, real_type>;

        int ione = 1;
        int M = step+1;
        dense_matrix_type X (M, M, true);
        real_vector_type S (M, true);
        for (int i=0; i<M; i++) {
          X(i, i) = Rnew(i, i);
          for (int j=i+1; j<M; j++) {
            X(i, j) = Rnew(i, j);
            X(j, i) = Rnew(i, j);
          }
        }
        /*for (int i=0; i<M; i++) {
          for (int j=0; j<M; j++) {
            printf("%.16e ",X(i, j));
          }
          printf("\n");
        }*/
        int INFO, LWORK;
        SC U, VT, TEMP;
        real_type RWORK;
        LWORK = -1;
        lapack.GESVD('N', 'N', M, M, X.values (), M, S.values (), &U, ione, &VT, ione,
                     &TEMP, LWORK, &RWORK, &INFO);
        LWORK = Teuchos::as<LO> (STS::real (TEMP));
        dense_vector_type WORK (LWORK, true);
        lapack.GESVD('N', 'N', M, M, X.values (), M, S.values (), &U, ione, &VT, ione,
                     WORK.values (), LWORK, &RWORK, &INFO);
        std::cout << "cond(G):" << numIters << ", sqrt(" << S(0) << "/" << S(M-1) << ")="
                  << "sqrt(" << S(0)/S(M-1) << ")="
                  << std::sqrt(S(0)/S(M-1)) << std::endl;
      }
      #endif

      // Compute the Cholesky factorization of R in place
      {
        Teuchos::TimeMonitor LocalTimer (*TsqrTimer);

        int info = 0;
        int ncols = step+1;
        const LO LDR = Rnew.stride();
        lapack.POTRF ('U', ncols, Rnew.values (), LDR, &info);
        if (info < 0) {
          for (int i=info-1; i<ncols; i++) {
            Rnew(i, i) = one;
            for (int j=i+1; j<ncols; j++) {
              Rnew(i, j) = zero;
            }
          }
        }
        // zero-out, in case re-orthogonalization update with trmm
        const SC zero = STS::zero ();
        for (int i=0; i<step+1; i++) {
          for (int j=0; j<i; j++) {
            Rnew(i, j) = zero;
          }
        }

        // Compute A_cur / R (Matlab notation for A_cur * R^{-1}) in place.
        {
          MV R_mv = makeStaticLocalMultiVector (Q, ncols, ncols);
          Tpetra::deep_copy (R_mv, Rnew);
          auto Q_d = Qnew.getLocalViewDevice (Tpetra::Access::ReadWrite);
          auto R_d = R_mv.getLocalViewDevice (Tpetra::Access::ReadOnly);
          KokkosBlas::trsm ("R", "U", "N", "N",
                            one, R_d, Q_d);
        }
        rank = ncols;
      }
    } else {
      // should not be called with itre <= 0
      //rank = normalizeCholQR (iter, step, Q, G);
    }

    return rank;
  }

  int
  reNormalizeCholQR2 (const int prevIter, // starting index of columns for re-orthogonalization 
                      const int stepSize, // number of columns for re-orthogonalization
                      const int step,     // number of new columns, need to be re-scaled
                      MV& Q,
                      dense_matrix_type& C,  // store aggregated coefficient, results of block dot-products
                      dense_matrix_type& G,  // Hessenburg matrix
                      dense_matrix_type& G2) // workspace
  {
    const SC one  = STS::one  ();
    const SC zero = STS::zero ();

    Teuchos::BLAS<LO, SC> blas;
    Teuchos::LAPACK<LO, SC> lapack;

    int rank = 0;
    int iter = prevIter+stepSize;

    // re-normalize the previous s-step set of vectors (lagged)
    // making a copy of C(prevIter,prevIter):
    // G still contains coeff from previous step for convergence check
    //dense_matrix_type Rfix (stepSize, stepSize, true);
    dense_matrix_type Rfix (Teuchos::View, G2, stepSize, stepSize, 0, 0);
    for (int i=0; i < stepSize; i++) {
      for (int j=i; j < stepSize; j++) {
        Rfix(i, j) = C(prevIter+i, prevIter+j);
      }
    }

    // Compute the Cholesky factorization of R in place
    int info = 0;
    const LO LDR = Rfix.stride();
    lapack.POTRF ('U', stepSize, Rfix.values (), LDR, &info);
    if (info < 0) {
      for (int i=info-1; i<stepSize; i++) {
        Rfix(i, i) = one;
        for (int j=i+1; j<stepSize; j++) {
          Rfix(i, j) = zero;
        }
      }
      rank = info;
      throw std::runtime_error("second Cholesky factorization failed");
    } else {
      rank = stepSize;
    }

    // Compute A_cur / R (Matlab notation for A_cur * R^{-1}) in place.
    Teuchos::Range1D index_old(prevIter, iter-1);
    MV Qold = * (Q.subView(index_old));

    // rescale the previous vector, MVT::MvScale (Qn, one / tnn);
    {
      MV R_mv = makeStaticLocalMultiVector (Q, stepSize, stepSize);
      Tpetra::deep_copy (R_mv, Rfix);
      auto Q_d = Qold.getLocalViewDevice (Tpetra::Access::ReadWrite);
      auto R_d = R_mv.getLocalViewDevice (Tpetra::Access::ReadOnly);
      KokkosBlas::trsm ("R", "U", "N", "N",
                        one, R_d, Q_d);
    }

    // merge two R
    // H(n, n-1) *= tnn;
    dense_matrix_type Rold (Teuchos::View, G, stepSize, stepSize, prevIter, 0);
    blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
               stepSize, stepSize,
               one, Rfix.values(), Rfix.stride(),
                    Rold.values(), Rold.stride());

    // update coefficients
    // for (int i = 0; i < n; i++) T(i, n) /= tnn;
    dense_matrix_type c1(Teuchos::View, C, iter, stepSize, 0, prevIter);
    blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
               iter+1, stepSize,
               one, Rfix.values(), Rfix.stride(),
                    c1.values(), c1.stride());
    // H(n, n) /= tnn;
    dense_matrix_type c2 (Teuchos::View, C, stepSize, step, prevIter, prevIter);
    blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::TRANS, Teuchos::NON_UNIT_DIAG,
               stepSize, stepSize+step+1,
               one, Rfix.values(), Rfix.stride(),
                    c2.values(), c2.stride());

    return rank;
  }

private:
  bool useCholQR2_;
  std::string orthoType_;
};
#endif

//
// s-step GMRES
template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<SC>,
         class OP = Tpetra::Operator<SC>>
class GmresSstep : public Gmres<SC, MV, OP>  {
private:
  using base_type = Gmres<SC, MV, OP>;
  using MVT = Belos::MultiVecTraits<SC, MV>;
  using LO = typename MV::local_ordinal_type;
  using STS = Teuchos::ScalarTraits<SC>;
  using mag_type = typename STS::magnitudeType;
  using STM = Teuchos::ScalarTraits<mag_type>;
  using complex_type = std::complex<mag_type>;
  using dense_matrix_type = Teuchos::SerialDenseMatrix<LO, SC>;
  using dense_vector_type = Teuchos::SerialDenseVector<LO, SC>;
  using device_type = typename MV::device_type;
  using dot_type = typename MV::dot_type;
  using vec_type = typename base_type::vec_type;
  using ortho_type = typename base_type::ortho_type;

public:
  GmresSstep () :
    base_type::Gmres (),
    numOrthoSteps_ (0),
    useTSQR_1st_step_ (false),
    lowSynch_ (false),
    bgs_ (Teuchos::null),
    cholqr_ (Teuchos::null),
    tsqr_ (Teuchos::null)
  {}

  GmresSstep (const Teuchos::RCP<const OP>& A) :
    base_type::Gmres (A),
    numOrthoSteps_ (0),
    useTSQR_1st_step_ (false),
    lowSynch_ (false),
    bgs_ (Teuchos::null),
    cholqr_ (Teuchos::null),
    tsqr_ (Teuchos::null)
  {}

  virtual ~GmresSstep () = default;

  virtual void
  getParameters (Teuchos::ParameterList& params,
                 const bool defaultValues) const
  {
    base_type::getParameters (params, defaultValues);

    const int stepSize = defaultValues ? 5 : this->input_.stepSize;
    params.set ("Step Size", stepSize );
  }

  virtual void
  setParameters (Teuchos::ParameterList& params) {
    base_type::setParameters (params);
    int stepSize = params.get<int> ("Step Size", this->input_.stepSize);
    this->input_.stepSize = stepSize;

    bool computeRitzValuesOnFly 
      = params.get<bool> ("Compute Ritz Values on Fly", this->input_.computeRitzValuesOnFly);
    this->input_.computeRitzValuesOnFly = computeRitzValuesOnFly;

    // ortho option
    std::string orthoType = this->input_.orthoType;
    constexpr bool useCholQR2_default = false;
    bool useCholQR2 = params.get<bool> ("CholeskyQR2", useCholQR2_default);

    if (orthoType == "CGS2 LowSynch" || orthoType == "MGS LowSynch") {
      bgs_ = Teuchos::rcp (new LowSynchBGS<SC, MV, OP> (useCholQR2, this->input_.orthoType));

      tsqr_ = Teuchos::null;
      cholqr_ = Teuchos::null;
      lowSynch_ = true;
    } //else 
    {
      // intra block ortho option (CholQR is default)
      // (low-synch uses standard intra-block ortho on the first block)
      constexpr bool useCholQR_default = true;
      bool useCholQR = params.get<bool> ("CholeskyQR", useCholQR_default);
      if (useCholQR2) {
        useCholQR = true;
        numOrthoSteps_ = 2;
      }

      constexpr bool useSVQR_default = false;
      bool useSVQR = params.get<bool> ("SVQR", useSVQR_default);

      constexpr bool useSVQR2_default = false;
      bool useSVQR2 = params.get<bool> ("SVQR2", useSVQR2_default);
      if (useSVQR2) {
        useSVQR = true;
        numOrthoSteps_ = 2;
      }
      if (useSVQR || useSVQR2) {
        useCholQR = false;
        useCholQR2 = false;
        if (!cholqr_.is_null ()) {
          cholqr_ = Teuchos::null;
        }
      }

      bool useTSQR_1st_step = params.get<bool> ("TSQR for initial step", useTSQR_1st_step_);

      std::string tsqrType
        = params.get<std::string> ("TSQR", this->input_.tsqrType);
      if (tsqrType != "none") {
        useCholQR = false;
        useCholQR2 = false;

        useSVQR = false;
        useSVQR2 = false;
        this->setOrthogonalizer (tsqrType, tsqr_);
      } else {
        if (useTSQR_1st_step) {
          this->setOrthogonalizer ("TSQR", tsqr_);
        } else if (!tsqr_.is_null ()) {
          tsqr_ = Teuchos::null;
        }
      }
      useTSQR_1st_step_ = useTSQR_1st_step;

      if (!useCholQR && !useSVQR) {
        if (!cholqr_.is_null ()) {
          cholqr_ = Teuchos::null;
        }
      } else if (useCholQR || useSVQR) {
        if (cholqr_.is_null ()) {
          cholqr_ = Teuchos::rcp (new CholQR<SC, MV, OP> (useSVQR));
        }
      }

      this->input_.tsqrType = tsqrType;
    }
  }

private:
  SolverOutput<SC>
  solveOneVec (Teuchos::FancyOStream* outPtr,
               vec_type& X, // in X/out X
               vec_type& B, // in B/out R (not left-preconditioned)
               const OP& A,
               const OP& M,
               const SolverInput<SC>& input)
  {
    using std::endl;
    int stepSize = input.stepSize;
    int restart = input.resCycle;
    int step = stepSize;
    const SC zero = STS::zero ();
    const SC one  = STS::one ();

    // timers
    Teuchos::RCP< Teuchos::Time > spmvTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::matrix-apply");
    Teuchos::RCP< Teuchos::Time > bortTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::BOrtho");
    Teuchos::RCP< Teuchos::Time > tsqrTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::TSQR");

    // initialize output parameters
    SolverOutput<SC> output {};
    output.converged = false;
    output.numRests = 0;
    output.numIters = 0;

    if (outPtr != nullptr) {
      *outPtr << "GmresSstep" << endl;
      Indent indent1 (outPtr);
      *outPtr << "Solver input:" << endl;
      Indent indentInner (outPtr);
      *outPtr << input;
    }

    Teuchos::BLAS<LO, SC> blas;
    Teuchos::LAPACK<LO, SC> lapack;
    dense_matrix_type  H (restart+1, restart,   true);  // Hessenburg matrix
    dense_matrix_type  Rh(restart+1, restart,   true);  // H reduced to upper-triangular matrix
    dense_matrix_type  G (restart+1, step+1,    true);  // Upper-triangular matrix from ortho process
    dense_matrix_type  G2(restart+1, restart,   true);  // A copy of Hessenburg matrix for computing Ritz values
    dense_matrix_type  C (restart+1, restart+1, true);  // Aggregated dot-products for low-synch Gram-Schmidtt
    dense_vector_type  y (restart+1, true);
    dense_matrix_type  h (restart+1, 1, true); // used for reorthogonalization
    std::vector<mag_type> cs (restart);
    std::vector<SC> sn (restart);

    #ifdef HAVE_TPETRA_DEBUG
    dense_matrix_type H2 (restart+1, restart,   true);
    dense_matrix_type H3 (restart+1, restart,   true);
    #endif

    mag_type b_norm;  // initial residual norm
    mag_type b0_norm; // initial residual norm, not left preconditioned
    mag_type r_norm;
    mag_type r_norm_imp;
    mag_type metric;

    bool zeroOut = false; // Kokkos::View:init can take a long time on GPU?
    vec_type R (B.getMap (), zeroOut);
    vec_type Y (B.getMap (), zeroOut);
    vec_type MP (B.getMap (), zeroOut);
    MV  Q (B.getMap (), restart+1, zeroOut);
    vec_type P0 = * (Q.getVectorNonConst (0));

    // Compute initial residual (making sure R = B - Ax)
    {
      Teuchos::TimeMonitor LocalTimer (*spmvTimer);
      A.apply (X, R);
    }
    R.update (one, B, -one);
    b0_norm = R.norm2 (); // initial residual norm, not preconditioned
    if (input.precoSide == "left") {
      M.apply (R, P0);
      r_norm = P0.norm2 (); // initial residual norm, left-preconditioned
    } else {
      r_norm = b0_norm;
    }
    b_norm = r_norm;

    metric = this->getConvergenceMetric (b0_norm, b0_norm, input);
    if (metric <= input.tol) {
      if (outPtr != nullptr) {
        *outPtr << "Initial guess' residual norm " << b_norm
                << " meets tolerance " << input.tol << endl;
      }
      output.absResid = r_norm;
      output.relResid = r_norm / b0_norm;
      output.converged = true;
      // Return residual norm as B
      Tpetra::deep_copy (B, P0);
      return output;
    } else if (STM::isnaninf (metric)) {
      if (outPtr != nullptr) {
        *outPtr << "Initial guess' residual norm " << b_norm
                << " is nan " << endl;
      }
      output.absResid = r_norm;
      output.relResid = r_norm / b0_norm;
      output.converged = false;
      // Return residual norm as B
      Tpetra::deep_copy (B, P0);
      return output;
    } else if (input.computeRitzValues && !input.computeRitzValuesOnFly) {
      // Invoke standard Gmres for the first restart cycle, to compute
      // Ritz values for use as Newton shifts
      if (outPtr != nullptr) {
        *outPtr << "Run standard GMRES for first restart cycle" << endl;
      }
      SolverInput<SC> input_gmres = input;
      input_gmres.maxNumIters = input.resCycle;
      input_gmres.maxNumIters = std::min(input.resCycle, input.maxNumIters);
      input_gmres.computeRitzValues = true;

      Tpetra::deep_copy (R, B);
      output = Gmres<SC, MV, OP>::solveOneVec (outPtr, X, R, A, M,
                                               input_gmres);
      if (output.converged) {
        return output; // standard GMRES converged
      }

      if (input.precoSide == "left") {
        M.apply (R, P0);
        r_norm = P0.norm2 (); // residual norm
      }
      else {
        r_norm = output.absResid;
      }
      output.numRests++;
    }

    // Initialize starting vector
    if (input.precoSide != "left") {
      Tpetra::deep_copy (P0, R);
    }
    P0.scale (one / r_norm);
    y[0] = r_norm;

    // Main loop
    int iter = 0;
    while (output.numIters < input.maxNumIters && ! output.converged) {
      if (outPtr != nullptr) {
        *outPtr << "Restart cycle " << output.numRests << ":" << endl;
        Indent indent2 (outPtr);
        *outPtr << output;
      }

      if (input.maxNumIters < output.numIters+restart) {
        restart = input.maxNumIters-output.numIters;
      }

      // Restart cycle
      for (; iter < restart && metric > input.tol; iter+=step) {
        if (outPtr != nullptr) {
          *outPtr << "Current iteration: iter=" << iter
                  << ", restart=" << restart
                  << ", step=" << step
                  << ", metric=" << metric << endl;
          Indent indent3 (outPtr);
        }

        // Compute matrix powers
        int prevStep = stepSize;
        if (input.computeRitzValuesOnFly && output.numIters < input.stepSize) {
          stepSize = 1;
        } else {
          stepSize = input.stepSize;
        }
        for (step=0; step < stepSize && iter+step < restart; step++) {
          // AP = A*P
          vec_type P  = * (Q.getVectorNonConst (iter+step));
          vec_type AP = * (Q.getVectorNonConst (iter+step+1));
          if (input.precoSide == "none") {
            Teuchos::TimeMonitor LocalTimer (*spmvTimer);
            A.apply (P, AP);
          }
          else if (input.precoSide == "right") {
            M.apply (P, MP);
            {
              Teuchos::TimeMonitor LocalTimer (*spmvTimer);
              A.apply (MP, AP);
            }
          }
          else {
            {
              Teuchos::TimeMonitor LocalTimer (*spmvTimer);
              A.apply (P, MP);
            }
            M.apply (MP, AP);
          }
          // Shift for Newton basis
          if ( int (output.ritzValues.size()) > step) {
            //AP.update (-output.ritzValues(step), P, one);
            const complex_type theta = output.ritzValues[step];
            UpdateNewton<SC, MV>::updateNewtonV(iter+step, Q, theta);
          }
          #ifdef HAVE_TPETRA_DEBUG
          if (outPtr != nullptr) {
            // vector to be orthogonalized
            Teuchos::Range1D index_prev(iter, iter+step+1);
            MV Qnew = * (Q.subView(index_prev));

            dense_matrix_type T2 (step+2, step+2, true);
            MVT::MvTransMv(one, Qnew, Qnew, T2);
            SC condV = STM::squareroot (this->computeCondNum(T2));
            *outPtr << " > condNum( V(" << iter << ":" << iter+step+1 << ") ) = " << condV << endl;
          }
          #endif
          output.numIters++;
        }

        // Orthogonalization
        int rank = 0;
        if (lowSynch_ && iter > 0) {
          // return delayed rank
          rank = bgs_->factor (outPtr, Q, C, G, G2, iter, step, prevStep, output.numIters);
        } else {
          {
            Teuchos::TimeMonitor LocalTimer (*bortTimer);
            this->projectBelosOrthoManager (iter, step, Q, G);
          }
          {
            Teuchos::TimeMonitor LocalTimer (*tsqrTimer);
            rank = recursiveCholQR (outPtr, iter, step, Q, G, output.numIters);
            if (numOrthoSteps_ > 1 && (iter > 0 || !useTSQR_1st_step_)) {
              rank = recursiveCholQR (outPtr, iter, step, Q, G2, output.numIters);
              // merge R 
              dense_matrix_type Rfix (Teuchos::View, G2, step+1, step+1, iter, 0);
              dense_matrix_type Rold (Teuchos::View, G,  step+1, step+1, iter, 0);
              blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                         Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                         step+1, step+1,
                         one, Rfix.values(), Rfix.stride(),
                              Rold.values(), Rold.stride());
            }
          }
          if (rank == 0) {
            // FIXME: Don't throw; report an error code.
            throw std::runtime_error("orthogonalization failed with rank = 0");
          }
        }
        bool delayedNorm = lowSynch_;
        bool doneCheck = (delayedNorm && iter == 0);
        while (!doneCheck) {
          int Step = (delayedNorm ? prevStep      : step);
          int Iter = (delayedNorm ? iter-prevStep : iter);
          int Rank = (delayedNorm ? rank+1        : rank); // delayed-norm leaves the starting vector for the current block
          if (delayedNorm) {
            updateHessenburg (Iter, Step, output.ritzValues, H, G2);
          } else {
            updateHessenburg (Iter, Step, output.ritzValues, H, G);
          }
          // Convergence check
          if (Rank == Step+1 && H(Iter+Step, Iter+Step-1) != zero) {
            // Copy H to R and apply Givens rotations to new columns of T and y
            for (int iiter = 0; iiter < Step; iiter++) {
              // Check negative norm
              TEUCHOS_TEST_FOR_EXCEPTION
                (STS::real (H(Iter+iiter+1, Iter+iiter)) < STM::zero (),
                 std::runtime_error, "At iteration " << output.numIters << ", H("
                 << Iter+iiter+1 << ", " << Iter+iiter << ") = "
                 << H(Iter+iiter+1, Iter+iiter) << " < 0.");

              for (int i = 0; i <= Iter+iiter+1; i++) {
                Rh(i, Iter+iiter) = H(i, Iter+iiter);
              }
              #ifdef HAVE_TPETRA_DEBUG
              this->checkNumerics (outPtr, Iter+iiter, Iter+iiter, A, M, Q, X, B, y,
                                   H, H2, H3, cs, sn, input);
              #endif
              this->reduceHessenburgToTriangular(Iter+iiter, Rh, cs, sn, y);
              metric = this->getConvergenceMetric (STS::magnitude (y(Iter+iiter+1)), b_norm, input);
              if (outPtr != nullptr) {
                *outPtr << " > implicit residual norm=(" << Iter+iiter+1 << ")="
                        << STS::magnitude (y(Iter+iiter+1))
                        << " metric=" << metric << endl;
              }
              if (STM::isnaninf (metric) || metric <= input.tol) {
                if (outPtr != nullptr) {
                  *outPtr << " > break at step = " << iiter+1 << " (" << step << ")" << endl;
                }
                step = iiter+1;
                break;
              }
            }
            if (STM::isnaninf (metric)) {
              // metric is nan
              break;
            }
          }
          else {
            if (outPtr != nullptr) {
              *outPtr << " >  H(" << Iter+Step << ", " << Iter+Step-1 << ") = " <<  H(Iter+Step, Iter+Step-1)
                      << " -> zero (rank = " << rank << ", step = " << Step << ")" << endl;
            }
            metric = STM::zero ();
          }
          if (delayedNorm && iter+step >= restart) {
            // clean up
            rank = step+1; // full-rank..
            delayedNorm = false;
          } else {
            // done check
            doneCheck = true;
          }
        }

        // Optionally, compute Ritz values for generating Newton basis
        if (input.computeRitzValuesOnFly && int (output.ritzValues.size()) == 0
            && output.numIters >= input.stepSize) {
          for (int i = 0; i < input.stepSize; i++) {
            for (int iiter = 0; iiter < input.stepSize; iiter++) {
              G2(i, iiter) = H(i, iiter);
            }
          }
          computeRitzValues (input.stepSize, G2, output.ritzValues);
          sortRitzValues <LO, SC> (input.stepSize, output.ritzValues);
          if (outPtr != nullptr) {
            *outPtr << " > ComputeRitzValues: " << endl;
            for (int i = 0; i < input.stepSize; i++) {
              *outPtr << " > ritzValues[ " << i << " ] = " << output.ritzValues[i] << endl;
            }
          }
        }
      } // End of restart cycle

      // Update solution
      if (iter < restart) {
        // save the old solution, just in case explicit residual norm failed the convergence test
        Tpetra::deep_copy (Y, X);
        blas.COPY (1+iter, y.values(), 1, h.values(), 1);
      }
      r_norm_imp = STS::magnitude (y (iter)); // save implicit residual norm
      blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                 Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                 iter, 1, one,
                 Rh.values(), Rh.stride(), y.values(), y.stride());
      Teuchos::Range1D cols(0, iter-1);
      Teuchos::RCP<const MV> Qj = Q.subView(cols);
      dense_vector_type y_iter (Teuchos::View, y.values (), iter);
      if (input.precoSide == "right") {
        MVT::MvTimesMatAddMv (one, *Qj, y_iter, zero, R);
        M.apply (R, MP);
        X.update (one, MP, one);
      }
      else {
        MVT::MvTimesMatAddMv (one, *Qj, y_iter, one, X);
      }
      // Compute real residual (not-preconditioned)
      {
        Teuchos::TimeMonitor LocalTimer (*spmvTimer);
        A.apply (X, R);
      }
      R.update (one, B, -one);
      r_norm = R.norm2 (); // residual norm
      output.absResid = r_norm;
      output.relResid = r_norm / b0_norm;
      if (outPtr != nullptr) {
        *outPtr << "Implicit and explicit residual norms at restart: " << r_norm_imp << ", " << r_norm << endl;
      }

      metric = this->getConvergenceMetric (r_norm, b0_norm, input);
      if (metric <= input.tol) {
        // update solution
        output.converged = true;
      }
      else if (STM::isnaninf (metric)) {
        // failed with nan
        // Return residual norm as B
        Tpetra::deep_copy (B, R);
        return output;
      }
      else if (output.numIters < input.maxNumIters) {
        // Restart, only if max inner-iteration was reached.
        // Otherwise continue the inner-iteration.
        if (iter >= restart) {
          // Restart: Initialize starting vector for restart
          iter = 0;
          P0 = * (Q.getVectorNonConst (0));
          if (input.precoSide == "left") { // left-precond'd residual norm
            M.apply (R, P0);
            r_norm = P0.norm2 ();
          }
          else {
            // set the starting vector
            Tpetra::deep_copy (P0, R);
          }
          P0.scale (one / r_norm);
          y[0] = SC {r_norm};
          for (int i=1; i < restart+1; ++i) {
            y[i] = STS::zero ();
          }
          output.numRests++;
        }
        else {
          // reset to the old solution
          Tpetra::deep_copy (X, Y);
          blas.COPY (1+iter, h.values(), 1, y.values(), 1);
        }
      }
    }

    // Return residual norm as B
    Tpetra::deep_copy (B, R);

    if (outPtr != nullptr) {
      *outPtr << "At end of solve:" << endl;
      Indent indentInner (outPtr);
      *outPtr << output;
    }
    return output;
  }

protected:
  virtual void
  setOrthogonalizer (const std::string& orthoType)
  {
    if (!lowSynch_) {
      base_type::setOrthogonalizer (orthoType);
    } else {
      this->input_.orthoType = orthoType;
    }
  }

  //! Create Belos::OrthoManager instance.
  virtual void
  setOrthogonalizer (const std::string& orthoType, Teuchos::RCP<ortho_type> &ortho)
  {
    base_type::setOrthogonalizer (orthoType, ortho);
  }

  void
  updateHessenburg (const int n,
                    const int s,
                    std::vector<complex_type>& S,
                    dense_matrix_type& H,
                    dense_matrix_type& R) const
  {
    const SC one  = STS::one ();
    const SC zero = STS::zero ();

    // 1) multiply H with R(1:n+s+1, 1:n+s+1) from left
    //    where R(1:n, 1:n) = I and 
    //          H(n+1:n+s+1, n+1:n+s) = 0, except h(n+j+1,n+j)=1 for j=1,..,s
    // 1.1) copy: H(j:n-1, j:n-1) = R(j:n-1, j:n-1), i.e., H = R*B
    for (int j = 0; j < s; j++ ) {
      for (int i = 0; i <= n+j+1; i++) {
        H(i, n+j) = R(i, j+1);
        if (int (S.size ()) > j && i <= n+j) {
          //H(i, n+j) += S[j].real * R(i, j);
          H(i, n+j) += UpdateNewton<SC, MV>::updateNewtonH (i, j, R, S[j]);
        }
      }
      for(int i = n+j+2; i <= n+s; i++) {
        H(i, n+j) = zero;
      }
    }

    // submatrices
    dense_matrix_type r_diag (Teuchos::View, R, s,   s, n, 0);
    dense_matrix_type h_diag (Teuchos::View, H, s+1, s, n, n);
    Teuchos::BLAS<LO, SC> blas;

    if (n == 0) { // >> first matrix-power iteration <<
      // 2) multiply H with R(1:s, 1:s)^{-1} from right
      // diagonal block
      blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS,
                 Teuchos::NON_UNIT_DIAG, s+1, s, one,
                 r_diag.values(), r_diag.stride(),
                 h_diag.values(), h_diag.stride());
    } else  { // >> rest of iterations <<
      // 1.2) update the starting vector
      for (int i = 0; i < n; i++ ) {
        H(i, n-1) += H(n, n-1) * R(i, 0);
      }
      H(n, n-1) *= R(n, 0);

      // 2) multiply H with R(1:n+s, 1:n+s)^{-1} from right,
      //    where R(1:n, 1:n) = I
      // 2.1) diagonal block
      for (int j = 0; j < s; j++ ) {
        H(n, n+j) -= H(n, n-1) * R(n-1, j);
      }
      // diagonal block
      blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS,
                 Teuchos::NON_UNIT_DIAG, s, s, one,
                 r_diag.values(), r_diag.stride(),
                 h_diag.values(), h_diag.stride());
      H(n+s, n+s-1) /= R(n+s-1, s-1);

      // 2.2) upper off-diagonal block: H(0:j-1, j:j+n-2)
      dense_matrix_type r_off (Teuchos::View, R, n, s, 0, 0);
      dense_matrix_type h_off (Teuchos::View, H, n, s, 0, n);
      blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                n, s, n,
               -one, H.values(),      H.stride(),
                     r_off.values(), r_off.stride(),
                one, h_off.values(), h_off.stride());

      blas.TRSM(Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS,
                Teuchos::NON_UNIT_DIAG, n, s, one,
                r_diag.values(), r_diag.stride(),
                h_off.values(),  h_off.stride() );
    }
  }

  //! Apply the orthogonalization using Belos' OrthoManager
  int
  normalizeCholQR (Teuchos::FancyOStream* outPtr,
                   const int n,
                   const int s,
                   MV& Q,
                   dense_matrix_type& R)
  {
    // vector to be orthogonalized
    Teuchos::Range1D index_prev(n, n+s);
    MV Qnew = * (Q.subView(index_prev));

    dense_matrix_type r_new (Teuchos::View, R, s+1, s+1, n, 0);

    int rank = 0;
    if (cholqr_ != Teuchos::null) {
      rank = cholqr_->factor (outPtr, Qnew, r_new);
    }
    else {
      rank = this->normalizeBelosOrthoManager (Qnew, r_new);
    }
    return rank;
  }

  //! Apply the orthogonalization using Belos' OrthoManager
  int
  recursiveCholQR (Teuchos::FancyOStream* outPtr,
                   const int n,
                   const int s,
                   MV& Q,
                   dense_matrix_type& R,
                   const int iters)
  {
    // vector to be orthogonalized
    Teuchos::Range1D index_prev(n, n+s);
    MV Qnew = * (Q.subView(index_prev));

    dense_matrix_type r_new (Teuchos::View, R, s+1, s+1, n, 0);

    int rank = 0;
    if (cholqr_ != Teuchos::null && (n > 0 || !useTSQR_1st_step_)) {
      rank = cholqr_->reFactor (outPtr, Qnew, r_new);
      if (outPtr != nullptr) {
        *outPtr << " ** CholQR (iter = " << iters << ") ** " << std::endl;
        int numReFacto = cholqr_->getNumReFacto();
        if (numReFacto > 0) {
          *outPtr << " x CholQR(" << iters << "): numReFacto = " << numReFacto << std::endl;
        }
      }
    }
    else {
      *outPtr << " ** TSQR (iter = " << iters << ") ** " << std::endl;
      rank = this->normalizeBelosOrthoManager (Qnew, r_new);
    }
    return rank;
  }

  virtual int
  normalizeBelosOrthoManager (MV& Q, dense_matrix_type& R)
  {
    if (this->input_.tsqrType != "none") {
      if (tsqr_.get () == nullptr) {
        this->setOrthogonalizer (this->input_.tsqrType, tsqr_);
      }

      Teuchos::RCP<dense_matrix_type> R_ptr = Teuchos::rcpFromRef (R);
      return tsqr_->normalize (Q, R_ptr);
    }

    return base_type::normalizeBelosOrthoManager (Q, R);
  }

private:
  int numOrthoSteps_;
  bool useTSQR_1st_step_;
  bool lowSynch_;
  Teuchos::RCP<LowSynchBGS<SC, MV, OP> > bgs_;
  Teuchos::RCP<CholQR<SC, MV, OP> > cholqr_;
  Teuchos::RCP<ortho_type> tsqr_;
};


template<class SC, class MV, class OP,
         template<class, class, class> class KrylovSubclassType>
class SolverManager;

// This is the Belos::SolverManager subclass that gets registered with
// Belos::SolverFactory.
template<class SC, class MV, class OP>
using GmresSstepSolverManager = SolverManager<SC, MV, OP, GmresSstep>;

/// \brief Register GmresSstepSolverManager for all enabled Tpetra
///   template parameter combinations.
void register_GmresSstep (const bool verbose);

} // namespace Impl
} // namespace BelosTpetra

#endif // BELOS_TPETRA_GMRES_SSTEP_HPP
