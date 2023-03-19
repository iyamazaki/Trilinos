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

template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<>,
         class OP = Tpetra::Operator<> >
class CholQR {
private:
  using LO = typename MV::local_ordinal_type;
  using blas_type = Teuchos::BLAS<LO, SC>;
  using lapack_type = Teuchos::LAPACK<LO, SC>;

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

  /// \brief Constructor
  ///
  CholQR () = default;

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

    /*if (computeCondNum) {
      const LO ione = 1;

      LO m = G.numRows ();
      dense_matrix_type C (m, m, true); 
      for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) C(i,j) = G(i,j);
      }

      LO INFO, LWORK;
      SC  U, VT, TEMP;
      real_type RWORK;
      real_vector_type S (m, true);
      LWORK = -1;
      Teuchos::LAPACK<LO ,SC> lapack;
      lapack.GESVD('N', 'N', m, m, C.values (), C.stride (),
                   S.values (), &U, ione, &VT, ione,
                   &TEMP, LWORK, &RWORK, &INFO);
      LWORK = Teuchos::as<LO> (STS::real (TEMP));
      dense_vector_type WORK (LWORK, true);
      lapack.GESVD('N', 'N', m, m, C.values (), C.stride (),
                   S.values (), &U, ione, &VT, ione,
                   WORK.values (), LWORK, &RWORK, &INFO);
      condNum = S(0) / S(m-1);
    }*/
    // Compute the Cholesky factorization of R in place, so that
    // A^T * A = R^T * R, where R is ncols by ncols upper
    // triangular.
    int info = 0;
    {
      auto R_h = R_mv.getLocalViewHost (Tpetra::Access::ReadWrite);
      int ldr = int (R_h.extent (0));
      SC *Rdata = reinterpret_cast<SC*> (R_h.data ());
      lapack.POTRF ('U', ncols, Rdata, ldr, &info);
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
    {
      auto A_d = A.getLocalViewDevice (Tpetra::Access::ReadWrite);
      auto R_d = R_mv.getLocalViewDevice (Tpetra::Access::ReadOnly);
/*std::cout << "Q(A) = [" <<std::endl;
for (int i=0; i<A_d.extent(0); i++) {
  for (int j=0; j<A_d.extent(1); j++) std::cout << A_d(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;*/
      KokkosBlas::trsm ("R", "U", "N", "N",
                        one, R_d, A_d);
/*std::cout << " => [" <<std::endl;
for (int i=0; i<A_d.extent(0); i++) {
  for (int j=0; j<A_d.extent(1); j++) std::cout << A_d(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;*/
    }
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
    while (rank < ncols && old_rank != rank) {
      Teuchos::Range1D next_index(rank, ncols-1);
      MV nextA = * (A.subView(next_index));

      dense_matrix_type nextR (Teuchos::View, R, ncols-rank, ncols-rank, rank, rank);
      old_rank = rank;
      auto new_rank = factor (outPtr, nextA, nextR);
      if (outPtr != nullptr) {
        if (rank > 0) {
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
  using vec_type = typename Krylov<SC, MV, OP>::vec_type;
  using device_type = typename MV::device_type;
  using dot_type = typename MV::dot_type;

public:
  GmresSstep () :
    base_type::Gmres (),
    useRandomQR_ (false),
    useRandomCGS2_ (false),
    useCholQR2_ (false),
    cholqr_ (Teuchos::null)
  {}

  GmresSstep (const Teuchos::RCP<const OP>& A) :
    base_type::Gmres (A),
    cholqr_ (Teuchos::null)
  {}

  virtual ~GmresSstep () = default;

  virtual void
  getParameters (Teuchos::ParameterList& params,
                 const bool defaultValues) const
  {
    base_type::getParameters (params, defaultValues);

    const int stepSize = defaultValues ? 5 : this->input_.stepSize;
    params.set ("Step Size", stepSize );

    const int sketchDom = defaultValues ? this->input_.resCycle : this->input_.sketchDom;
    params.set ("Sketch Domain", sketchDom );
  }

  virtual void
  setParameters (Teuchos::ParameterList& params) {
    base_type::setParameters (params);
    int stepSize = params.get<int> ("Step Size", this->input_.stepSize);
    this->input_.stepSize = stepSize;

    int sketchDom = params.get<int> ("Sketch Domain", this->input_.resCycle);
    this->input_.sketchDom = sketchDom;

    bool computeRitzValuesOnFly 
      = params.get<bool> ("Compute Ritz Values on Fly", this->input_.computeRitzValuesOnFly);
    this->input_.computeRitzValuesOnFly = computeRitzValuesOnFly;

    constexpr bool useCholQR_default = true;
    bool useCholQR = params.get<bool> ("CholeskyQR", useCholQR_default);

    bool useRandomQR = params.get<bool> ("RandomQR", useRandomQR_);
    useRandomQR_ = useRandomQR;

    bool useRandomCGS2 = params.get<bool> ("RandomCGS2", useRandomCGS2_);
    useRandomCGS2_ = useRandomCGS2;

    bool useCholQR2 = params.get<bool> ("CholeskyQR2", useCholQR2_);
    useCholQR2_ = useCholQR2;

    if ((!useCholQR && !useCholQR2) && !cholqr_.is_null ()) {
      cholqr_ = Teuchos::null;
    } else if ((useCholQR || useCholQR2) && cholqr_.is_null ()) {
      cholqr_ = Teuchos::rcp (new CholQR<SC, MV, OP> ());
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
    const SC  one  =  STS::one ();
    const SC mone  = -STS::one ();

    // timers
    Teuchos::RCP< Teuchos::Time > totalTimer = Teuchos::TimeMonitor::getNewCounter("GmresSstep::Total");

    Teuchos::RCP< Teuchos::Time > spmvTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::Matrix-apply");
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
    dense_matrix_type  H (restart+1, restart, true); // Hessenburg matrix
    dense_matrix_type  T (restart+1, restart, true); // H reduced to upper-triangular matrix
    dense_matrix_type  G (restart+1, restart+1, true);  // Upper-triangular matrix from ortho process
    dense_matrix_type  G2(restart+1, restart+1, true); // a copy of Hessenburg matrix for computing Ritz values
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

    // random Gaussian vectors for random sketching
    int sketchDom = input.sketchDom;
    int sketchSize = 2*(1+sketchDom);
    dense_matrix_type  Qhat (sketchSize, restart+1, true); // projected matrix
    MV  W (B.getMap (), sketchSize, zeroOut);
    W.randomize ();

{
    Teuchos::TimeMonitor GmresTimer (*totalTimer);

    // Compute initial residual (making sure R = B - Ax)
    {
      Teuchos::TimeMonitor LocalTimer (*spmvTimer);
      A.apply (X, R);
    }
    R.update (one, B, mone);
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
    int currentSketchSize = 0;
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
      H.putScalar (zero);
      G.putScalar (zero); // re-initialize R for randomCGS2
      for (int i = 0; i < restart+1; i++) G(i, i) = one;
      for (; iter < restart && metric > input.tol; iter+=step) {
        if (outPtr != nullptr) {
          *outPtr << "Current s-step iteration: iter=" << iter
                  << ", restart=" << restart
                  << ", step=" << step
                  << ", metric=" << metric << endl;
          Indent indent3 (outPtr);
        }

        // Compute matrix powers
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
          output.numIters++;
        }

        // Orthogonalization
        int rank = 0;
        {
          Teuchos::TimeMonitor LocalTimer (*bortTimer);
          rank = this->projectBelosBlockOrthoManager (outPtr, iter, step, Q, sketchSize, W, Qhat, G, G2);
          currentSketchSize += step;
        }
        #ifdef HAVE_TPETRA_DEBUG
        for (int iiter = 0; iiter < step; iiter++) {
          Teuchos::Range1D cols(iter, iter+iiter+1);
          Teuchos::RCP<const MV> Qj = Q.subView(cols);
          *outPtr << " > condNum( " << iter+iiter << " ) = " << this->computeCondNum(*Qj) << std::endl;
        }
        #endif
        {
          Teuchos::TimeMonitor LocalTimer (*tsqrTimer);
          // first panel QR
          if (useRandomQR_ || useRandomCGS2_) {
            if (!useRandomCGS2_) {
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << " > calling RandQR" << std::endl;
#endif
              rank = randomQR (outPtr, iter, step, W, Q, Qhat, G);
            }
          } else {
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << " > calling CholQR" << std::endl;
#endif
            rank = recursiveCholQR (outPtr, iter, step, Q, G);
          }
          if (useCholQR2_ && !useRandomCGS2_) {
            // second panel QR
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << " > calling CholQR2" << std::endl;
#endif
            rank = recursiveCholQR (outPtr, iter, step, Q, G2);
            // merge R 
            dense_matrix_type Rfix (Teuchos::View, G2, step+1, step+1, iter, 0);
            dense_matrix_type Rold (Teuchos::View, G,  step+1, step+1, iter, 0);
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- Rold = [" <<std::endl;
for (int i=0; i<step+1; i++) {
  for (int j=0; j<step+1; j++) std::cout << Rold(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
std::cout << "- Rfix = [" <<std::endl;
for (int i=0; i<step+1; i++) {
  for (int j=0; j<step+1; j++) std::cout << Rfix(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
            blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                       Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                       step+1, step+1,
                       one, Rfix.values(), Rfix.stride(),
                            Rold.values(), Rold.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- R = [" <<std::endl;
for (int i=0; i<step+1; i++) {
  for (int j=0; j<step+1; j++) std::cout << Rold(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
          }
          if (rank == 0) {
            // FIXME: Don't throw; report an error code.
            throw std::runtime_error("orthogonalization failed with rank = 0");
          }
        }

        #ifndef FORM_GLOBAL_R
        updateHessenburg (iter, step, output.ritzValues, H, G);
        #endif

        if (useRandomCGS2_ && (iter+step == restart || (iter+step)%sketchDom == 0)) {
          int start_col = (iter+step)-currentSketchSize;
          int end_col = iter+step;

#ifdef SKETCH_SSTEP_GMRES_DEBUG
{
  std::cout << " > calling last CholQR2(" << start_col << ":" << end_col << ", iter=" << iter << ", step=" << step << ", restart=" << restart << ", sketch=" << sketchSize << "x" << sketchDom << ")" << std::endl;
  auto Q_h = Q.getLocalViewHost (Tpetra::Access::ReadOnly);
  std::cout << "- V_chol = [" <<std::endl;
  for (int i=0; i<Q_h.extent(0); i++) {
    for (int j=0; j<=end_col; j++) printf("%.16e ", Q_h(i,j));
    std::cout << std::endl;
  }
  std::cout << "];" << std::endl;

  std::cout << "- H = [" <<std::endl;
  for (int i=0; i<1+end_col; i++) {
    for (int j=0; j<end_col; j++) printf("%.16e ",H(i,j));
    std::cout << std::endl;
  }
  std::cout << "];" << std::endl;
}
#endif
          if (start_col > 0) {
            Teuchos::RCP< Teuchos::Time > reorthTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::ReOrtho");
            Teuchos::TimeMonitor LocalTimer (*reorthTimer);
            // call block CGS2
            // vectors to be orthogonalized against
            Teuchos::Range1D index_old (0, start_col-1);
            MV Qold = *(Q.subViewNonConst (index_old));

            // vectors to be orthogonalized
            Teuchos::Range1D index(start_col, start_col+currentSketchSize);
            MV Qnew = *(Q.subViewNonConst (index));

            dense_matrix_type R_new (Teuchos::View, G2, start_col, currentSketchSize+1, 0, 0);
            MVT::MvTransMv(one, Qold, Qnew, R_new);
            MVT::MvTimesMatAddMv(mone, Qold, R_new, one, Qnew);
          }

          // call CholQR
          rank = recursiveCholQR (outPtr, start_col, currentSketchSize, Q, G2);

          // merge R 
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- Rfix = [" <<std::endl;
for (int i=0; i<start_col+currentSketchSize+1; i++) {
  for (int j=0; j<currentSketchSize+1; j++) printf("%.16e ", G2(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
          #ifdef FORM_GLOBAL_R
          dense_matrix_type Rold (Teuchos::View, G,  currentSketchSize+1, currentSketchSize+1, 0, 0);
          blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                     Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                     currentSketchSize+1, currentSketchSize+1,
                     one, Rfix.values(), Rfix.stride(),
                          Rold.values(), Rold.stride());
          #else
          if (start_col > 0) {
              // upper off-diagonals
              {
                  // H = H*R^{-1}
                  dense_matrix_type H11 (Teuchos::View, H,  start_col+1, start_col, 0, 0);
                  dense_matrix_type H12 (Teuchos::View, H,  start_col+1, currentSketchSize, 0, start_col);
                  dense_matrix_type R12_(Teuchos::View, G2, start_col,   currentSketchSize, 0, 0);
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H11 = [" <<std::endl;
for (int i=0; i<start_col+1; i++) {
  for (int j=0; j<start_col; j++) printf("%.16e ", H11(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
std::cout << "- R12 = [" <<std::endl;
for (int i=0; i<start_col; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", R12_(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
std::cout << "- H12 = [" <<std::endl;
for (int i=0; i<start_col+1; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H12(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
                  blas.GEMM (Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                             start_col+1, currentSketchSize, start_col,
                            mone, H11.values(),  H11.stride(),
                                  R12_.values(), R12_.stride(),
                             one, H12.values(),  H12.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H_off = [" <<std::endl;
for (int i=0; i<start_col+1; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H12(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
                  dense_matrix_type R22 (Teuchos::View, G2, currentSketchSize, currentSketchSize, start_col, 0);
                  blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
                             Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                             end_col+1, currentSketchSize,
                             one, R22.values(), R22.stride(),
                                  H12.values(), H12.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- R22 = [" <<std::endl;
for (int i=0; i<1+currentSketchSize; i++) {
  for (int j=0; j<1+currentSketchSize; j++) printf("%.16e ", R22(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H_off = [" <<std::endl;
for (int i=0; i<end_col+1; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H12(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
                  std::cout << "H12 = H(0:" << start_col-1 << ", " << start_col-1 << ":" << start_col+currentSketchSize-1 << " " << start_col << "x" << currentSketchSize+1 << ")" << std::endl;
#endif

                  // H = R*H
                  dense_matrix_type _H12 (Teuchos::View, H,  start_col,   currentSketchSize+1, 0,         start_col-1);
                  dense_matrix_type _H22 (Teuchos::View, H,  currentSketchSize+1, currentSketchSize+1, start_col, start_col-1);
                  dense_matrix_type  R12 (Teuchos::View, G2, start_col,   currentSketchSize+1, 0, 0);
                  blas.GEMM (Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                             start_col, currentSketchSize+1, currentSketchSize+1,
                             one,  R12.values(),  R12.stride(),
                                  _H22.values(), _H22.stride(),
                             one, _H12.values(), _H12.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H_off = [" <<std::endl;
for (int i=0; i<start_col; i++) {
  for (int j=0; j<currentSketchSize+1; j++) printf("%.16e ", _H12(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
              }

              // "Diagonal" part
              {
                  // H = R*H
                  dense_matrix_type H22_(Teuchos::View, H,  currentSketchSize+1, currentSketchSize,   start_col, start_col);
                  dense_matrix_type R22_(Teuchos::View, G2, currentSketchSize+1, currentSketchSize+1, start_col, 0);
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << std::endl << " DIAG " << std::endl;
std::cout << "- H22_ = [" <<std::endl;
for (int i=0; i<currentSketchSize; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H22_(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
std::cout << "- R22_ = [" <<std::endl;
for (int i=0; i<currentSketchSize; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", R22_(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
                  blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                             Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                             currentSketchSize+1, currentSketchSize,
                             one, R22_.values(), R22_.stride(),
                                  H22_.values(),  H22_.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H22_ = [" <<std::endl;
for (int i=0; i<currentSketchSize; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H22_(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
                  #if 0
                  // H = H*R^{-1}
                  //dense_matrix_type H22 (Teuchos::View, H,  sketchDom,   sketchDom-1, start_col+1, start_col+2);
                  dense_matrix_type R22 (Teuchos::View, G2, sketchDom, sketchDom, start_col,   0);
std::cout << "- R22 = [" <<std::endl;
for (int i=0; i<sketchDom; i++) {
  for (int j=0; j<sketchDom; j++) printf("%.16e ", R22(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
                  blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
                             Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                             sketchDom, sketchDom,
                             one, R22.values(), R22.stride(),
                                  H22_.values(), H22_.stride());
std::cout << "- H22_ = [" <<std::endl;
for (int i=0; i<sketchDom; i++) {
  for (int j=0; j<sketchDom; j++) printf("%.16e ", H22_(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
                  #endif
              }
          } else {
              // H = R*H
              dense_matrix_type H22 (Teuchos::View, H,  currentSketchSize+1, currentSketchSize,   start_col, start_col);
              dense_matrix_type R22_(Teuchos::View, G2, currentSketchSize+1, currentSketchSize+1, start_col, 0);
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H22 = [" <<std::endl;
for (int i=0; i<currentSketchSize+1; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H22(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
              blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                         Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                         currentSketchSize+1, currentSketchSize,
                         one, R22_.values(), R22_.stride(),
                              H22.values(), H22.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H22 = [" <<std::endl;
for (int i=0; i<currentSketchSize+1; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H22(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif

              // H = H*R^{-1}
              dense_matrix_type R22 (Teuchos::View, G2, currentSketchSize, currentSketchSize, start_col, 0);
              blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
                         Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                         currentSketchSize+1, currentSketchSize,
                         one, R22.values(), R22.stride(),
                              H22.values(), H22.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- H22 = [" <<std::endl;
for (int i=0; i<currentSketchSize+1; i++) {
  for (int j=0; j<currentSketchSize; j++) printf("%.16e ", H22(i,j));
  std::cout << std::endl;
}
#endif
          }
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- Hnew = [" <<std::endl;
for (int i=0; i<end_col+1; i++) {
  for (int j=0; j<end_col; j++) printf("%.16e ", H(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
{
  auto Q_h = Q.getLocalViewHost (Tpetra::Access::ReadOnly);
  std::cout << "- Q_chol = [" <<std::endl;
  for (int i=0; i<Q_h.extent(0); i++) {
    for (int j=0; j<=end_col; j++) printf("%.16e ", Q_h(i,j));
    std::cout << std::endl;
  }
  std::cout << "];" << std::endl;
}
#endif
          #endif
        } // end of ortho after sketching

        if (!useRandomCGS2_ || (iter+step == restart || (iter+step)%sketchDom == 0)) {
          int start_index = iter;
          int step_size = step;
          if (useRandomCGS2_) {
            start_index = (iter+step)-currentSketchSize;
            step_size = currentSketchSize;
          }

          #ifdef FORM_GLOBAL_R
          updateHessenburg (start_index, step_size, output.ritzValues, H, G);
          #endif

          // Convergence check
          if (rank == step_size+1 && H(start_index+step_size, start_index+step_size-1) != zero) {
            // Copy H to T and apply Givens rotations to new columns of T and y
            for (int iiter = 0; iiter < step_size; iiter++) {
              // Check negative norm
              TEUCHOS_TEST_FOR_EXCEPTION
                (STS::real (H(start_index+iiter+1, start_index+iiter)) < STM::zero (),
                 std::runtime_error, "At iteration " << output.numIters << ", H("
                 << start_index+iiter+1 << ", " << start_index+iiter << ") = "
                 << H(start_index+iiter+1, start_index+iiter) << " < 0.");

              for (int i = 0; i <= start_index+iiter+1; i++) {
                T(i, start_index+iiter) = H(i, start_index+iiter);
              }
              #ifdef HAVE_TPETRA_DEBUG
              this->checkNumerics (outPtr, start_index+iiter, start_index+iiter, A, M, Q, X, B, y,
                                   H, H2, H3, cs, sn, input);
              #endif
              this->reduceHessenburgToTriangular(start_index+iiter, T, cs, sn, y);
              metric = this->getConvergenceMetric (STS::magnitude (y(start_index+iiter+1)), b_norm, input);
              if (outPtr != nullptr) {
                *outPtr << " > implicit residual norm=(" << start_index+iiter+1 << ")="
                        << STS::magnitude (y(start_index+iiter+1))
                        << " metric=" << metric << endl;
              }
              if (STM::isnaninf (metric) || metric <= input.tol) {
                if (outPtr != nullptr) {
                  *outPtr << " > break at step = " << iiter+1 << " (" << step_size << ")" << endl;
                }
                step = (iiter+1) - (iter-start_index);
                break;
              }
            }
            if (STM::isnaninf (metric)) {
              // metric is nan
              break;
            }
          }
          else {
            metric = STM::zero ();
          }
          currentSketchSize = 0;
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
                 T.values(), T.stride(), y.values(), y.stride());
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
        *outPtr << "Implicit and explicit residual norms at restart: " << r_norm_imp << ", " << r_norm
                << " (iter = " << iter << ", " << output.numIters << ")" << endl;
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

  // ! Apply the orthogonalization
  int
  projectBelosBlockOrthoManager(Teuchos::FancyOStream* outPtr,
                                const int n,
                                const int s,
                                MV& Q,
                                const int sketchSize,
                                MV& W,
                                dense_matrix_type &Qhat,
                                dense_matrix_type &R,
                                dense_matrix_type &R2)
  {
    int rank = 0;
    Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho");
    Teuchos::TimeMonitor LocalTimer (*factorTimer);

    if (this->input_.orthoType == "CGS2") {
      Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::RandomCGS2");
      Teuchos::TimeMonitor LocalTimer (*factorTimer);
      const SC one  = STS::one ();
      const SC zero = STS::zero ();

      // vectors to be orthogonalized
      Teuchos::Range1D index(n, n+s);
      MV Qnew = *(Q.subViewNonConst (index));

      // vectors to be orthogonalized against
      Teuchos::Range1D index_old (0, n-1);
      MV Qold = *(Q.subViewNonConst (index_old));

      // sketched version of vectors
      MV O_mv = makeStaticLocalMultiVector (Qnew, sketchSize, s+1);

      // random sketch Qhat := W^T * Q
      if (useRandomCGS2_) {
        Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::Sketch");
        Teuchos::TimeMonitor LocalTimer (*factorTimer);

        Teuchos::Range1D index_sketch(0, sketchSize-1);
        MV Ws = *(W.subViewNonConst (index_sketch));
        O_mv.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, one, Ws, Qnew, zero);
        {
          // Qhat on host
          dense_matrix_type Qhat_new (Teuchos::View, Qhat, sketchSize, s+1, 0, n);
          auto O_h = O_mv.getLocalViewHost (Tpetra::Access::ReadOnly);
          for (int j = 0; j < s+1; j++) {
            for (int i = 0; i < sketchSize; i++) {
              Qhat_new(i,j) = O_h(i,j);
            }
          }
        }
#ifdef SKETCH_SSTEP_GMRES_DEBUG
{
  auto Q_h = Qnew.getLocalViewHost (Tpetra::Access::ReadOnly);
  std::cout << "- Q_new = [" <<std::endl;
  for (int i=0; i<Q_h.extent(0); i++) {
    for (int j=0; j<Q_h.extent(1); j++) printf("%.16e ", Q_h(i,j));
    std::cout << std::endl;
  }
}
std::cout << "];" << std::endl << std::endl;
#endif
      }

      if (n > 0) {
        if (useRandomCGS2_) {
          Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::CGS-1");
          Teuchos::TimeMonitor LocalTimer (*factorTimer);

          dense_matrix_type Qhat_new (Teuchos::View, Qhat, sketchSize, s+1, 0, n);
          dense_matrix_type Qhat_old (Teuchos::View, Qhat, sketchSize, n,   0, 0);
          dense_matrix_type R2_new (Teuchos::View, R2, n, s+1, 0, 0);
          #ifdef FORM_GLOBAL_R
          dense_matrix_type R_new  (Teuchos::View, R,  n, s+1, 0, n);
          #else
          dense_matrix_type R_new  (Teuchos::View, R,  n, s+1, 0, 0);
          #endif
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- Qhat_old = [" <<std::endl;
for (int i=0; i<sketchSize; i++) {
  for (int j=0; j<n; j++) printf("%.16e ", Qhat(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;
std::cout << "- Qhat_new = [" <<std::endl;
for (int i=0; i<sketchSize; i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", Qhat_new(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;
#endif

          // R = Qhat_old^T * Qhat_new
          Teuchos::BLAS<LO, SC> blas;
          blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                    n, s+1, sketchSize,
                    one,  Qhat_old.values(), Qhat_old.stride(),
                          Qhat_new.values(), Qhat_new.stride(),
                    zero, R2_new.values(),   R2_new.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- R_new = [" <<std::endl;
for (int i=0; i<n; i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", R2_new(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;
#endif

          // Qhat_new = Qhat_new - Qhat_old*R
          blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                    sketchSize, s+1, n,
                   -one, Qhat_old.values(), Qhat_old.stride(),
                         R2_new.values(),   R2_new.stride(),
                    one, Qhat_new.values(), Qhat_new.stride());

          #ifdef FORM_GLOBAL_R
          // ------------------------------------------------------------------------ //
          // NOTE: updated for each R2 since to accumulate and update,
          // we need additional buffer to store new update
          //
          // generate the "well-conditioned" basis vectors (original, not sketched)
          //  > orthogonalize with inner-product defined by sketch matrix
          MVT::MvTimesMatAddMv(-one, Qold, R2_new, one, Qnew);
          // ------------------------------------------------------------------------ //
          #endif

          // accumulate coefficients (since the first column is from the previous ortho)
          #ifdef FORM_GLOBAL_R
          for (int i = 0; i < n; i++) {
            R_new(i,0) += R2_new(i,0)*R(n,n);
          }
          for (int j = 1; j < s+1; j++) {
            for (int i = 0; i < n; i++) {
              R_new(i,j) += R2_new(i,j);
            }
          }
          #else
          for (int j = 0; j < s+1; j++) {
            for (int i = 0; i < n; i++) {
              R_new(i,j) = R2_new(i,j);
            }
          }
          #endif
/*std::cout << "- Qhat_out = [" <<std::endl;
for (int i=0; i<2*(s+1); i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", Qhat_new(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;*/
        } // end of randomCGS2
        else {
          dense_matrix_type R_new (Teuchos::View, R, n, s+1, 0, 0);
          MVT::MvTransMv(one, Qold, Qnew, R_new);
          MVT::MvTimesMatAddMv(-one, Qold, R_new, one, Qnew);
        }
#ifdef SKETCH_SSTEP_GMRES_DEBUG
{
  auto Q_h = Qnew.getLocalViewHost (Tpetra::Access::ReadOnly);
  std::cout << " R(" << n << ", " << n << ")=" << R(n,n) << std::endl;
  std::cout << "- Q_cgs = [" <<std::endl;
  for (int i=0; i<Q_h.extent(0); i++) {
    for (int j=0; j<Q_h.extent(1); j++) printf("%.16e ", Q_h(i,j));
    std::cout << std::endl;
  }
}
std::cout << "];" << std::endl << std::endl;
std::cout << " current R = [" << std::endl;
#ifdef FORM_GLOBAL_R
for (int i=0; i<n+s+1; i++) {
  for (int j=0; j<n+s+1; j++) printf("%.16e ", R(i,j));
  std::cout << std::endl;
}
#else
for (int i=0; i<n; i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", R(i,j));
  std::cout << std::endl;
}
#endif
std::cout << "];" << std::endl << std::endl;
#endif
        // second ortho
        // > orthogonalization coefficients
        dense_matrix_type R2_new (Teuchos::View, R2, n, s+1, 0, 0);
        if (useRandomCGS2_) {

          // projected matrix
          dense_matrix_type Qhat_old (Teuchos::View, Qhat, sketchSize, n,   0, 0);
          dense_matrix_type Qhat_new (Teuchos::View, Qhat, sketchSize, s+1, 0, n);

          #if 0
          // panel, to bring down norm of new vectors
          {
            dense_matrix_type Qhat_new (Teuchos::View, Qhat, 2*(s+1), s+1, 0, n);
            auto O_h = O_mv.getLocalViewHost (Tpetra::Access::ReadWrite);
            int ld = int (O_h.extent (0));
            SC *Odata = reinterpret_cast<SC*> (O_h.data ());
            for (int i=0; i<2*(s+1); i++) {
              for (int j=0; j<s+1; j++) {
                Odata[i + j*ld] = Qhat_new(i, j);
              }
            }
          }
          rank = randomQR_ (outPtr, n, s, O_mv, Q, Qhat, R);
          #endif

          const mag_type eps = STS::eps();
          //const mag_type tol = std::sqrt(eps);
          const mag_type tol = 100.0*eps;
          const int max_iters = 5;
          bool converged = false;
          int iters = 0;
          dense_matrix_type r_nrms (s+1, 1);
          dense_matrix_type q_nrms (s+1, 1);

          while (!converged && iters < max_iters) {
            Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::CGS-2");
            Teuchos::TimeMonitor LocalTimer (*factorTimer);

            for (int j=0; j<s+1; j++) {
              q_nrms(j,0) = zero;
              for (int i=0; i<sketchSize; i++) q_nrms(j,0) += Qhat_new(i,j)*Qhat_new(i,j);
              q_nrms(j,0) = std::sqrt(q_nrms(j,0));
            }
/*std::cout << std::endl << " -- iter = " << iters << " --" << std::endl;
std::cout << "- Qhat_new_in = [" <<std::endl;
for (int i=0; i<2*(s+1); i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", Qhat_new(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;*/

            // R = Qhat_old^T * Qhat_new
            Teuchos::BLAS<LO, SC> blas;
            blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                      n, s+1, sketchSize,
                      one,  Qhat_old.values(), Qhat_old.stride(),
                            Qhat_new.values(), Qhat_new.stride(),
                      zero, R2_new.values(),   R2_new.stride());
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- R2_new_out = [" <<std::endl;
for (int i=0; i<n; i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", R2_new(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;
#endif

            // Qhat_new = Qhat_new - Qhat_old*R
            blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                      sketchSize, s+1, n,
                     -one, Qhat_old.values(), Qhat_old.stride(),
                           R2_new.values(),   R2_new.stride(),
                      one, Qhat_new.values(), Qhat_new.stride());

            #ifdef FORM_GLOBAL_R
            // ------------------------------------------------------------------------ //
            // NOTE: updated for each R2 since to accumulate and update,
            // we need additional buffer to store new update
            //
            // generate the "well-conditioned" basis vectors (original, not sketched)
            //  > orthogonalize with inner-product defined by sketch matrix
            MVT::MvTimesMatAddMv(-one, Qold, R2_new, one, Qnew);
            // ------------------------------------------------------------------------ //
            #endif

            // accumulate coefficients
            #ifdef FORM_GLOBAL_R
            dense_matrix_type R_new (Teuchos::View, R, n, s+1, 0, n);
            for (int i = 0; i < n; i++) {
              R_new(i,0) += R2_new(i,0)*R(n,n);
            }
            for (int j = 1; j < s+1; j++) {
              for (int i = 0; i < n; i++) {
                R_new(i,j) += R2_new(i,j);
              }
            }
            #else
            dense_matrix_type R_new (Teuchos::View, R, n, s+1, 0, 0);
            for (int j = 0; j < s+1; j++) {
              for (int i = 0; i < n; i++) {
                R_new(i,j) += R2_new(i,j);
              }
            }
            #endif
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "- R = [" <<std::endl;
for (int i=0; i<n; i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", R_new(i,j));
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;
#endif

            iters ++;
            converged = true;
            for (int j=0; j<s+1; j++) {
              r_nrms(j,0) = zero;
              for (int i=0; i<n; i++) r_nrms(j,0) += R2_new(i,j)*R2_new(i,j);
              r_nrms(j,0) = std::sqrt(r_nrms(j,0));
              #ifdef SKETCH_SSTEP_GMRES_DEBUG
              printf(" check %d: %.16e/%.16e = %.16e vs %.16e\n",j,r_nrms(j,0),q_nrms(j,0),r_nrms(j,0)/q_nrms(j,0),tol );
              #endif
              if (r_nrms(j,0) > tol*q_nrms(j,0)) converged = false;
            }
            #ifdef SKETCH_SSTEP_GMRES_DEBUG
            printf("%s\n\n",(converged ? "pass" : "faile"));
            #endif
          } // end of iterative CGS

          #ifndef FORM_GLOBAL_R
          {
            Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::Project");
            Teuchos::TimeMonitor LocalTimer (*factorTimer);
            // ------------------------------------------------------------------------ //
            // generate the "well-conditioned" basis vectors (original, not sketched)
            //  > orthogonalize with inner-product defined by sketch matrix
            dense_matrix_type R_new (Teuchos::View, R, n, s+1, 0, 0);
            MVT::MvTimesMatAddMv(-one, Qold, R_new, one, Qnew);
            // ------------------------------------------------------------------------ //
	  }
          #endif
          #ifdef SKETCH_SSTEP_GMRES_DEBUG
          std::cout << std::endl << " projectBelosBlockOrthoManager : done with BCGS2 " << std::endl;
          #endif
        } // end of randomCGS2
        else {
          // reortho for CGS2
          MVT::MvTransMv(one, Qold, Qnew, R2_new);
          MVT::MvTimesMatAddMv(-one, Qold, R2_new, one, Qnew);
          // accumulate coefficients
          dense_matrix_type R_new (Teuchos::View, R, n, s+1, 0, 0);
          for (int j = 0; j < s+1; j++) {
            for (int i = 0; i < n; i++) {
              R_new(i,j) += R2_new(i,j);
            }
          }
        }
      }
#ifdef SKETCH_SSTEP_GMRES_DEBUG
{
  auto Q_h = Qnew.getLocalViewHost (Tpetra::Access::ReadOnly);
  std::cout << "- Q_cgs = [" <<std::endl;
  for (int i=0; i<Q_h.extent(0); i++) {
    for (int j=0; j<Q_h.extent(1); j++) printf("%.16e ", Q_h(i,j));
    std::cout << std::endl;
  }
}
std::cout << "];" << std::endl << std::endl;
std::cout << " current R = [" << std::endl;
#ifdef FORM_GLOBAL_R
for (int i=0; i<n+s+1; i++) {
  for (int j=0; j<n+s+1; j++) printf("%.16e ", R(i,j));
  std::cout << std::endl;
}
#else
for (int i=0; i<n; i++) {
  for (int j=0; j<s+1; j++) printf("%.16e ", R(i,j));
  std::cout << std::endl;
}
#endif
std::cout << "];" << std::endl << std::endl;
#endif

      if (useRandomCGS2_) {
        Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::InterOrtho::CGS2::RandomQR");
        Teuchos::TimeMonitor LocalTimer (*factorTimer);

        // randomQR with Qhat_new before re-ortho
        {
          dense_matrix_type Qhat_new (Teuchos::View, Qhat, sketchSize, s+1, 0, n);
          auto O_h = O_mv.getLocalViewHost (Tpetra::Access::ReadWrite);
          int ld = int (O_h.extent (0));
          SC *Odata = reinterpret_cast<SC*> (O_h.data ());
          for (int i=0; i<sketchSize; i++) {
            for (int j=0; j<s+1; j++) {
              Odata[i + j*ld] = Qhat_new(i, j);
            }
          }
        }
        rank = randomQR_ (outPtr, sketchSize, n, s, O_mv, Q, Qhat, R2);

        #ifdef SKETCH_SSTEP_GMRES_DEBUG
        {
            // check ortho error
            dense_matrix_type g_all(n+s+1, n+s+1);
            dense_matrix_type q_all (Teuchos::View, Qhat, sketchSize, n+s+1, 0, 0);
            Teuchos::BLAS<LO, SC> blas;
            blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                      n+s+1, n+s+1, sketchSize,
                      one,  q_all.values(), q_all.stride(),
                            q_all.values(), q_all.stride(),
                      zero, g_all.values(), g_all.stride());
            //std::cout << "T=[" << std::endl;
            #if 1
            mag_type ortho_error(0.0);
            for (int i=0; i<n+s+1; i++) {
              for (int j=0; j<n+s+1; j++) {
                mag_type tij = std::abs(i==j ? g_all(i,j)-one : g_all(i,j));
                ortho_error = (ortho_error > tij ? ortho_error : tij);
                //std::cout << (i==j ? g_all(i,j)-one : g_all(i,j)) << " ";
              }
              //std::cout << std::endl;
            }
            std::cout << " Ortho error (projected, after rand cgs & qr) = " << ortho_error << std::endl;
            #endif
            //std::cout << "];" << std::endl;
        }
        #endif
        {
          Teuchos::BLAS<LO, SC> blas;
          dense_matrix_type Rfix (Teuchos::View, R2, s+1, s+1, n, 0);
          #ifdef FORM_GLOBAL_R
          dense_matrix_type Rold (Teuchos::View, R,  s+1, s+1, n, n);
          blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                     Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                     s+1, s+1,
                     one, Rfix.values(), Rfix.stride(),
                          Rold.values(), Rold.stride());
          #else
          dense_matrix_type Rold (Teuchos::View, R,  s+1, s+1, n, 0);
          for (int i=0; i<s+1; i++) {
            for (int j=0; j<s+1; j++) Rold(i,j) = Rfix(i,j);
          }
          #endif
        }
#ifdef SKETCH_SSTEP_GMRES_DEBUG
{
  auto Q_h = Qnew.getLocalViewHost (Tpetra::Access::ReadOnly);
  std::cout << "- Q_new = [" <<std::endl;
  for (int i=0; i<Q_h.extent(0); i++) {
    for (int j=0; j<Q_h.extent(1); j++) printf("%.16e ", Q_h(i,j));
    std::cout << std::endl;
  }
}
std::cout << "];" << std::endl << std::endl;
        std::cout << " projectBelosBlock rank = " << rank << std::endl;
std::cout << " current R = [" << std::endl;
for (int i=0; i<n+s+1; i++) {
#ifdef FORM_GLOBAL_R
  for (int j=0; j<n+s+1; j++) printf("%.16e ", R(i,j));
#else
  for (int j=0; j<s+1; j++) printf("%.16e ", R(i,j));
#endif
  std::cout << std::endl;
}
std::cout << "];" << std::endl << std::endl;
#endif
      }
    }
    else {
      this->projectBelosOrthoManager(n, s, Q, R);
    }
    #ifdef SKETCH_SSTEP_GMRES_DEBUG
    std::cout << std::endl << " projectBelosBlockOrthoManager : done " << std::endl << std::endl;
    #endif

    return rank;
  }

protected:
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
    Teuchos::Range1D index(n, n+s);
    MV Qnew = * (Q.subView(index));

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
                   dense_matrix_type& R)
  {
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << " * in recursiveCholQR(" << n << ":" << n+s << ")" << std::endl;
#endif
    // vector to be orthogonalized
    Teuchos::Range1D index(n, n+s);
    MV Qnew = * (Q.subView(index));

    dense_matrix_type r_new (Teuchos::View, R, s+1, s+1, n, 0);

    int rank = 0;
    if (cholqr_ != Teuchos::null) {
      rank = cholqr_->reFactor (outPtr, Qnew, r_new);
    }
    else {
      rank = this->normalizeBelosOrthoManager (Qnew, r_new);
    }
    return rank;
  }

  //! Apply the orthogonalization using Belos' OrthoManager
  int
  randomQR (Teuchos::FancyOStream* outPtr,
            const int n,
            const int s,
            MV& W,
            MV& Q,
            dense_matrix_type& Qhat,
            dense_matrix_type& R)
  {
    Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::RandomQR");
    Teuchos::TimeMonitor LocalTimer (*factorTimer);

    const SC one  = STS::one ();
    const SC zero = STS::zero ();
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << " * in RandQR(n=" << n << ", s=" << s << ")" << std::endl;
#endif

    // random sketching vectors
    Teuchos::Range1D index_sketch(0, 2*(s+1)-1);
    MV Ws = * (W.subView(index_sketch));

    // vector to be orthogonalized
    Teuchos::Range1D index(n, n+s);
    MV Qnew = * (Q.subView(index));

    // random sketch Qhat := W^T * Q
    MV O_mv = makeStaticLocalMultiVector (Qnew, 2*(s+1), s+1);
    {
      Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::RandomQR::Sketch");
      Teuchos::TimeMonitor LocalTimer (*factorTimer);
      O_mv.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, one, Ws, Qnew, zero);
    }
    return randomQR_ (outPtr, 2*(s+1), n, s, O_mv, Q, Qhat, R);
  }

  int
  randomQR_ (Teuchos::FancyOStream* outPtr,
             const int m, // sketch size
             const int n, // offset
             const int s, // step size
             MV& O_mv,
             MV& Q,
             dense_matrix_type& Qhat,
             dense_matrix_type& R)
  {
    const SC one  = STS::one ();
    const SC zero = STS::zero ();

    // compute QR of O
    int info;
    int rank = 0;
    Teuchos::LAPACK<LO, SC> lapack;
    //dense_matrix_type r_new (Teuchos::View, R, s+1,  s+1, n, n);
    //dense_matrix_type r_new (Teuchos::View, R, s+1,  s+1, n, 0);
    dense_matrix_type r_new (Teuchos::View, R, s+1,  s+1, n, 0);

    int lwork = -1;
    dense_vector_type tau (s+1, true);
    {
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << std::endl << " In randomQR_ (n=" << n << ", s=" << s << ")" << std::endl;
#endif
      // get workspace size
      SC TEMP;
      lapack.GEQRF (m, s+1, &TEMP, m,
                    tau.values (), &TEMP, lwork, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error, "Belos::GmresSstep::randomQR:"
        " LAPACK's _GEQRF failed to compute a workspace size.");
      int lwork_geqrf = Teuchos::as<LO> (STS::real (TEMP));

      lapack.ORGQR (m, s+1, s+1, &TEMP, m,
                    tau.values (), &TEMP, lwork, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error, "Belos::GmresSstep::randomQR:"
        " LAPACK's _GORGQR failed to compute a workspace size.");
      int lwork_orgqr = Teuchos::as<LO> (STS::real (TEMP));
      lwork = (lwork_geqrf > lwork_orgqr ? lwork_geqrf : lwork_orgqr);

      // allocate workspace and call QR
#ifdef SKETCH_SSTEP_GMRES_DEBUG
      std::cout << " lwork = " << lwork << std::endl << std::flush;
#endif
    }

    dense_vector_type WORK (lwork, true);
    {
      auto Q_hat = O_mv.getLocalViewHost (Tpetra::Access::ReadWrite);
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "A = [" <<std::endl;
for (int i=0; i<m; i++) {
  for (int j=0; j<s+1; j++) std::cout << Q_hat(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif

      // compute QR
      lapack.GEQRF (m, s+1, Q_hat.data(), Q_hat.extent(0),
                    tau.values (), WORK.values (), lwork, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error, "Belos::GmresSstep::randomQR:"
        " LAPACK's _GEQRF failed to compute QR factorization.");

      // extract R
      rank = s+1;
      for (int i=0; i<rank; i++) {
        // zero-out lower-triangular part
        // keep original negative diagonals
        for (int j=0; j<i; j++) {
          r_new(i, j) = zero;
        }
        for (int j=i; j<rank; j++) {
          r_new(i, j) = Q_hat(i, j);
        }

        // make sure positive diagonals
        if (STS::real (Q_hat(i, i)) < STM::zero ()) {
#ifdef SKETCH_SSTEP_GMRES_DEBUG
 std::cout << " negative R(" << i << ",:)" << std::endl;
#endif
          for (int j=i; j<rank; j++) {
            Q_hat(i, j) = -Q_hat(i, j);
          }
        }
      }
    }
    {
      Teuchos::RCP< Teuchos::Time > factorTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep::RandomQR::Normalize");
      Teuchos::TimeMonitor LocalTimer (*factorTimer);

      using range_type = Kokkos::pair<int, int>;
      Teuchos::Range1D index(n, n+s);
      MV Qnew = * (Q.subView(index));
      auto Q_d = Qnew.getLocalViewDevice (Tpetra::Access::ReadWrite);
      auto O_d = O_mv.getLocalViewDevice (Tpetra::Access::ReadOnly);
      auto R_d = Kokkos::subview(O_d, range_type(0, s+1), Kokkos::ALL());
      KokkosBlas::trsm ("R", "U", "N", "N",
                        one, R_d, Q_d);
    }
    {
      // generate Qhat
      dense_matrix_type q_hat (Teuchos::View, Qhat, m, s+1, 0, n);
      auto Q_hat = O_mv.getLocalViewHost (Tpetra::Access::ReadWrite);
      lapack.ORGQR (m, s+1, s+1, Q_hat.data(), Q_hat.extent(0),
                    tau.values (), WORK.values (), lwork, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error, "Belos::GmresSstep::randomQR:"
        " LAPACK's _GEQRF failed to compute QR factorization.");

      // extract Qhat
      for (int i=0; i<s+1; i++) {
        if (STS::real (r_new(i, i)) < STM::zero ()) {
          // R(i,:) = -R(i,:) 
          for (int j=i; j<s+1; j++) {
            r_new(i, j) = -r_new(i, j);
          }
          // Q(:,i) = -Q(:,i) 
          for (int k=0; k<m; k++) {
            q_hat(k, i) = -Q_hat(k, i);
          }
        } else {
          // Q(:,i) = Q(:,i) 
          for (int k=0; k<m; k++) {
            q_hat(k, i) = Q_hat(k, i);
          }
        }
      }
#ifdef SKETCH_SSTEP_GMRES_DEBUG
std::cout << "Qhat = [" <<std::endl;
for (int i=0; i<m; i++) {
  for (int j=0; j<s+1; j++) std::cout << q_hat(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
std::cout << "R_new = [" <<std::endl;
for (int i=0; i<s+1; i++) {
  for (int j=0; j<s+1; j++) std::cout << r_new(i,j) << " ";
  std::cout << std::endl;
}
std::cout << "];" << std::endl;
#endif
    }

#ifdef SKETCH_SSTEP_GMRES_DEBUG
    {
      // checking ortho error
      dense_matrix_type g_all(n+s+1, n+s+1);
      dense_matrix_type q_all (Teuchos::View, Qhat, m, n+s, 0, 0);
      Teuchos::BLAS<LO, SC> blas;
      blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                n+s+1, n+s+1, m,
                one,  q_all.values(), q_all.stride(),
                      q_all.values(), q_all.stride(),
                zero, g_all.values(), g_all.stride());
      /*std::cout << "- Qall = [" <<std::endl;
      for (int i=0; i<2*(s+1); i++) {
        for (int j=0; j<n+s+1; j++) std::cout << Qhat(i,j) << " ";
          std::cout << std::endl;
      }
      std::cout << "];" << std::endl;*/
      mag_type ortho_error(0.0);
      std::cout << "T=[" << std::endl;
      for (int i=0; i<n+s+1; i++) {
        for (int j=0; j<n+s+1; j++) {
          mag_type tij = std::abs(i==j ? g_all(i,j)-one : g_all(i,j));
          ortho_error = (ortho_error > tij ? ortho_error : tij);
          std::cout << (i==j ? g_all(i,j)-one : g_all(i,j)) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "];" << std::endl;
      std::cout << " > Ortho error (projected vectors) = " << ortho_error << std::endl;
    }
#endif

    return rank;
  }

private:
  bool useRandomQR_;
  bool useRandomCGS2_;
  bool useCholQR2_;
  Teuchos::RCP<CholQR<SC, MV, OP> > cholqr_;
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
