#ifndef BELOS_TPETRA_GMRES_SSTEP_HPP
#define BELOS_TPETRA_GMRES_SSTEP_HPP

#include "Belos_Tpetra_Gmres.hpp"
#include "Belos_Tpetra_UpdateNewton.hpp"

// #include "wrapTpetraQR.hpp"
// #include "wrapTpetraCholQR.hpp"

namespace BelosTpetra {
namespace Impl {

template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<>,
         class OP = Tpetra::Operator<> >
class CholQR {
private:
  using LO = typename MV::local_ordinal_type;
  typedef Teuchos::BLAS<LO, SC> blas_type;
  typedef Teuchos::LAPACK<LO, SC> lapack_type;

public:
  /// \typedef FactorOutput
  /// \brief Return value of \c factor().
  ///
  /// Here, FactorOutput is just a minimal object whose value is
  /// irrelevant, so that this class' interface looks like that of
  /// \c CholQR.
  typedef int FactorOutput;
  typedef Teuchos::ScalarTraits<SC> STS;
  typedef typename STS::magnitudeType mag_type;
  typedef Teuchos::SerialDenseMatrix<LO, SC> dense_matrix_type;
  typedef Belos::MultiVecTraits<SC, MV> MVT;

  /// \brief Constructor
  ///
  /// \param theCacheSizeHint [in] Cache size hint in bytes.  If 0,
  ///   the implementation will pick a reasonable size, which may be
  ///   queried by calling cache_size_hint().
  CholQR () = default;

  /// \brief Compute the QR factorization of the matrix A.
  ///
  /// Compute the QR factorization of the nrows by ncols matrix A,
  /// with nrows >= ncols, stored either in column-major order (the
  /// default) or as contiguous column-major cache blocks, with
  /// leading dimension lda >= nrows.
  FactorOutput
  factor (MV& A, dense_matrix_type& R)
  {
    blas_type blas;
    lapack_type lapack;

    const SC zero = STS::zero ();
    const SC one  = STS::one ();

    LO ncols = A.getNumVectors ();
    LO nrows = A.getLocalLength ();

    // Compute R := A^T * A, using a single BLAS call.
    MVT::MvTransMv(one, A, A, R);

    // Compute the Cholesky factorization of R in place, so that
    // A^T * A = R^T * R, where R is ncols by ncols upper
    // triangular.
    int info = 0;
    lapack.POTRF ('U', ncols, R.values (), R.stride(), &info);
    if (info < 0) {
      ncols = -info;
      // FIXME (mfh 17 Sep 2018) Don't throw; report an error code.
      throw std::runtime_error("Cholesky factorization failed");
    }
    // TODO: replace with a routine to zero out lower-triangular
    //     : not needed, but done for testing
    for (int i=0; i<ncols; i++) {
      for (int j=0; j<i; j++) {
        R(i, j) = zero;
      }
    }

    // Compute A := A * R^{-1}.  We do this in place in A, using
    // BLAS' TRSM with the R factor (form POTRF) stored in the upper
    // triangle of R.

    // Compute A_cur / R (Matlab notation for A_cur * R^{-1}) in place.
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    A.template sync<Kokkos::HostSpace> ();
    A.template modify<Kokkos::HostSpace> ();
    auto A_lcl = A.template getLocalView<Kokkos::HostSpace> ();
#else
    A.sync_host ();
    A.modify_host ();
    auto A_lcl = A.getLocalViewHost ();
#endif
    SC* const A_lcl_raw = reinterpret_cast<SC*> (A_lcl.data ());
    const LO LDA = LO (A.getStride ());

    blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
               nrows, ncols, one, R.values(), R.stride(),
               A_lcl_raw, LDA);
    A.template sync<typename MV::device_type::memory_space> ();

    return (info > 0 ? info : ncols);
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
    stepSize_ (1),
    useCholQR2_ (false),
    tsqr_ (Teuchos::null)
  {
    this->input_.computeRitzValues = true;
  }

  GmresSstep (const Teuchos::RCP<const OP>& A) :
    base_type::Gmres (A),
    stepSize_ (1),
    useCholQR2_ (false),
    tsqr_ (Teuchos::null)
  {
    this->input_.computeRitzValues = true;
  }

  virtual ~GmresSstep ()
  {}

  virtual void
  getParameters (Teuchos::ParameterList& params,
                 const bool defaultValues) const
  {
    base_type::getParameters (params, defaultValues);

    const int stepSize = defaultValues ? 100 : stepSize_;

    params.set ("Step Size", stepSize );
  }

  virtual void
  setParameters (Teuchos::ParameterList& params) {
    base_type::setParameters (params);
    int stepSize = stepSize_;
    if (params.isParameter ("Step Size")) {
      stepSize = params.get<int> ("Step Size");
    }
    this->input_.stepSize = stepSize;

    bool useCholQR  = true;
    if (params.isParameter ("CholeskyQR")) {
      useCholQR = params.get<bool> ("CholeskyQR");
    }
    bool useCholQR2 = false;
    if (params.isParameter ("CholeskyQR2")) {
      useCholQR2 = params.get<bool> ("CholeskyQR2");
    }

    if ((useCholQR || useCholQR2) && tsqr_.is_null ()) {
      tsqr_ = Teuchos::rcp (new CholQR<SC, MV, OP> ());
    }
    stepSize_ = stepSize;
    useCholQR2_ = useCholQR2;
  }

protected:
  virtual void
  setOrthogonalizer (const std::string& ortho)
  {
    if (ortho != "CGS2x" && ortho != "CGS2" && ortho != "CGS1" &&
        ortho != "CGS" && ortho != "MGS") {
      Gmres<SC, MV, OP>::setOrthogonalizer (ortho);
    } else {
      this->input_.orthoType = ortho;
    }
  }

  SolverOutput<SC>
  solveOneVec (Teuchos::FancyOStream* outPtr,
               vec_type& X, // in X/out X
               vec_type& B, // in B/out R (not left-preconditioned)
               const OP& A,
               const OP& M,
               const SolverInput<SC>& input)
  {
    using std::endl;
    const int stepSize = stepSize_;
    int restart = input.resCycle;
    int step = stepSize;
    const SC zero = STS::zero ();
    const SC one  = STS::one ();
    const bool computeRitzValues = input.computeRitzValues;

    // initialize output parameters
    SolverOutput<SC> output {};
    output.converged = false;
    output.numRests = 0;
    output.numIters = 0;

    if (outPtr != nullptr) {
      *outPtr << "GmresSstep" << endl;
    }
    Indent indent1 (outPtr);
    if (outPtr != nullptr) {
      *outPtr << "Solver input:" << endl;
      Indent indentInner (outPtr);
      *outPtr << input;
    }

    Teuchos::BLAS<LO, SC> blas;
    Teuchos::LAPACK<LO, SC> lapack;
    dense_matrix_type  H (restart+1, restart,   true);  // Hessenburg matrix
    dense_matrix_type  T (restart+1, restart,   true);  // H reduced to upper-triangular matrix
    dense_matrix_type  G (restart+1, step+1,    true);  // Upper-triangular matrix from ortho process
    dense_matrix_type  C (restart+1, restart+1, true);
    dense_vector_type  y (restart+1, true);

    dense_matrix_type  H2 (restart+1, restart,   true);  // Hessenburg matrix, used for early convergence check for CGS2
    dense_matrix_type  T2 (restart+1, restart,   true);  // H reduced to upper-triangular matrix
    dense_vector_type  y2 (restart+1, true);
    std::vector<mag_type> cs (restart);
    std::vector<SC> sn (restart);

    mag_type b_norm;  // initial residual norm
    mag_type b0_norm; // initial residual norm, not left preconditioned
    mag_type r_norm;
    mag_type metric;
    vec_type R (B.getMap ());
    vec_type MP (B.getMap ());
    MV  Q (B.getMap (), restart+1);
    MV  W (B.getMap (), restart+1); // for lagged re-normalization
    vec_type P = * (Q.getVectorNonConst (0));

    // Compute initial residual (making sure R = B - Ax)
    A.apply (X, R);
    R.update (one, B, -one);
    b0_norm = R.norm2 (); // initial residual norm, not preconditioned
    if (input.precoSide == "left") {
      M.apply (R, P);
      r_norm = P.norm2 (); // initial residual norm, left-preconditioned
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
      Tpetra::deep_copy (B, P);
      return output;
    } else if (computeRitzValues) {
      // Invoke standard Gmres for the first restart cycle, to compute
      // Ritz values for use as Newton shifts
      if (outPtr != nullptr) {
        *outPtr << "Run standard GMRES for first restart cycle" << endl;
      }
      std::string oldOrthoType (this->input_.orthoType);
      SolverInput<SC> input_gmres = input;
      input_gmres.maxNumIters = input.resCycle;
      input_gmres.computeRitzValues = true;
      input_gmres.orthoType = "ICGS";

      // default to ICGS
      setOrthogonalizer (input_gmres.orthoType);

      Tpetra::deep_copy (R, B);
      output = Gmres<SC, MV, OP>::solveOneVec (outPtr, X, R, A, M,
                                               input_gmres);

      if (output.converged) {
        return output; // standard GMRES converged
      }

      if (input.precoSide == "left") {
        M.apply (R, P);
        r_norm = P.norm2 (); // residual norm
      }
      else {
        r_norm = output.absResid;
      }
      output.numRests++;

      // reset orthogonalizer
      setOrthogonalizer (oldOrthoType);
    }

    // Initialize starting vector
    if (input.precoSide != "left") {
      Tpetra::deep_copy (P, R);
    }
    P.scale (one / r_norm);
    y[0] = r_norm;
    y2[0] = r_norm;

    // Main loop
    Teuchos::RCP< Teuchos::Time > mainTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep iteration");
    Teuchos::RCP< Teuchos::Time > spmvTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep matrix-vector");
    Teuchos::RCP< Teuchos::Time > precTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep preconditioner");
    Teuchos::RCP< Teuchos::Time > orthTimer = Teuchos::TimeMonitor::getNewCounter ("GmresSstep orthogonalization");
    while (output.numIters < input.maxNumIters && ! output.converged) {
      Teuchos::TimeMonitor LocalTimer (*mainTimer);
      if (outPtr != nullptr) {
        *outPtr << "Restart cycle " << output.numRests << "(numIters = " << output.numIters << "):" << endl;
      }
      Indent indent2 (outPtr);
      /*if (outPtr != nullptr) {
        *outPtr << output;
      }*/

      if (input.maxNumIters < output.numIters+restart) {
        restart = input.maxNumIters-output.numIters;
      }

      // Restart cycle
      int rank = 0;
      int iter = 0;
      for (iter = 0; iter < restart && metric > input.tol; iter+=step) {
        Indent indent3 (outPtr);

        // Compute matrix powers
        //printf( "\n -- matrix-power(%d:%d) --\n",iter+1,iter+stepSize );
        for (step=0; step < stepSize && iter+step < restart; step++) {
          // AP = A*P
          vec_type P  = * (Q.getVectorNonConst (iter+step));
          vec_type AP = * (Q.getVectorNonConst (iter+step+1));
          if (input.precoSide == "none") {
            Teuchos::TimeMonitor LocalTimer (*spmvTimer);
            A.apply (P, AP);
          }
          else if (input.precoSide == "right") {
            // right preconditioner
            {
              Teuchos::TimeMonitor LocalTimer (*precTimer);
              M.apply (P, MP);
            }
            // matrix-vector
            {
              Teuchos::TimeMonitor LocalTimer (*spmvTimer);
              A.apply (MP, AP);
            }
          }
          else {
            // matrix-vector
            {
              Teuchos::TimeMonitor LocalTimer (*spmvTimer);
              A.apply (P, MP);
            }
            // left preconditioner
            {
              Teuchos::TimeMonitor LocalTimer (*precTimer);
              M.apply (MP, AP);
            }
          }
          // Shift for Newton basis
          if ( int (output.ritzValues.size()) > step) {
            //AP.update (-output.ritzValues(step), P, one);
            const complex_type theta = output.ritzValues[step];
            UpdateNewton<SC, MV>::updateNewtonV(iter+step, Q, theta);
          }

          // Save A*Q for lagged re-orthogonalization
          vec_type Wj = * (W.getVectorNonConst (iter+step));
          Tpetra::deep_copy (Wj, AP);

          output.numIters++;
        }

        // Orthogonalization
        if (this->input_.orthoType == "CGS2x" || // block CGS2 + explicit reortho + CholQR2
            this->input_.orthoType == "CGS2"  || // block CGS2 + CholQR2
            this->input_.orthoType == "CGS1"  || // block CGS1 + CholQR2
            this->input_.orthoType == "CGS"   || // block CGS1 + CholQR
            this->input_.orthoType == "MGS") {   // block MGS  + CholQR2
          Teuchos::TimeMonitor LocalTimer (*orthTimer);

          if (iter > 0) {
            int iterPrev = iter-stepSize;
            if (this->input_.orthoType == "CGS") {
              // dot-product for single-reduce orthogonalization
              Teuchos::Range1D index_next(iter, iter+step);
              MV Qnext = * (Q.subView(index_next));

              // vectors to be orthogonalized against
              Teuchos::Range1D index(0, iter+step);
              Teuchos::RCP< const MV > Qi = MVT::CloneView( Q, index );

              // compute coefficient, C(:,iter-stepSize:iter+step) = Q(:,0:iter+step)'*Q(iter-stepSize:iter+step)
              Teuchos::RCP< dense_matrix_type > c
                = Teuchos::rcp( new dense_matrix_type( Teuchos::View, C, iter+step+1, step+1, 0, iter ) );
              MVT::MvTransMv(one, *Qi, Qnext, *c);
            } else {
              // dot-product for re-normalization, and single-reduce orthogonalization
              // vector to be orthogonalized, and the vectors to be lagged-normalized
              Teuchos::Range1D index_next(iterPrev, iter+step);
              MV Qnext = * (Q.subView(index_next));

              // vectors to be orthogonalized against
              Teuchos::Range1D index(0, iter+step);
              Teuchos::RCP< const MV > Qi = MVT::CloneView( Q, index );

              // compute coefficient, C(:,iter-stepSize:iter+step) = Q(:,0:iter+step)'*Q(iter-stepSize:iter+step)
              Teuchos::RCP< dense_matrix_type > c
                = Teuchos::rcp( new dense_matrix_type( Teuchos::View, C, iter+step+1, stepSize+step+1, 0, iterPrev ) );
              MVT::MvTransMv(one, *Qi, Qnext, *c);

              // re-normalize the previous s-step set of vectors (lagged)
              if (useCholQR2_) {
                reNormalizeCholQR2 (iterPrev, stepSize, step, Q, C, G);
              }

              // update Hessenburg matrix from previous iter (lagged)
              //printf( "\n updateHessenbug(iterPrev=%d, stepSize=%d)\n",iterPrev,stepSize );
              updateHessenburg (iterPrev, stepSize, output.ritzValues, H, G);

              // Check negative norm from previous iter (lagged)
              TEUCHOS_TEST_FOR_EXCEPTION
                (STS::real (H(iter, iter-1)) < STM::zero (),
                 std::runtime_error, "At iteration " << output.numIters << ", H("
                 << iter << ", " << iter-1 << ") = "
                 << H(iter, iter-1) << " < 0.");

              // Convergence check from previous iter (lagged)
              if (rank == stepSize+1 && H(iter, iter-1) != zero) {
                // Copy H to T and apply Givens rotations to new columns of T and y
                for (int iiter = 0; iiter < stepSize; iiter++) {
                  for (int i = 0; i <= iterPrev+iiter+1; i++) {
                    T(i, iterPrev+iiter) = H(i, iterPrev+iiter);
                  }
                  this->reduceHessenburgToTriangular(iterPrev+iiter, T, cs, sn, y);
                }
                metric = this->getConvergenceMetric (STS::magnitude (y(iterPrev+stepSize)), b_norm, input);
              }
              else {
                metric = STM::zero ();
              }
              //printf( " > Convergence check(iter=%d, step=%d) metric=%.2e from previous iter ..\n",iterPrev,stepSize,metric );
              if (outPtr != nullptr) {
                // Update solution
                vec_type Y (B.getMap ());
                vec_type Z (B.getMap ());

                vec_type W1 (B.getMap ());
                vec_type W2 (B.getMap ());
                Tpetra::deep_copy (Y, X);
                if (iter > 0) {
                  dense_vector_type  z (iter, true);
                  blas.COPY (iter, y.values(), 1, z.values(), 1);
                  blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                             Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                             iter, 1, one,
                             T.values(), T.stride(), z.values(), z.stride());
                  Teuchos::Range1D cols(0, iter-1);
                  Teuchos::RCP<const MV> Qj = Q.subView(cols);
                  dense_vector_type z_iter (Teuchos::View, z.values (), iter);
                  if (input.precoSide == "right") {
                      MVT::MvTimesMatAddMv (one, *Qj, z_iter, zero, W1);
                      M.apply (W1, W2);
                      Y.update (one, W2, one);
                  }
                  else {
                      MVT::MvTimesMatAddMv (one, *Qj, z_iter, one, Y);
                  }
                }
                A.apply (Y, Z);
                Z.update (one, B, -one);
                SC z_norm = Z.norm2 (); // residual norm from previous step
                *outPtr << "> Current iteration: iter=" << output.numIters-stepSize-1 << ", " << iter
                        << ", restart=" << restart
                        << ", metric=" << metric
                        << ", real resnorm=" << z_norm << " " << z_norm/b_norm
                        << endl;
              }
            }

            // orthogonalize the new vectors against the previous columns
            rank = projectAndNormalizeCholQR2 (output.numIters-stepSize-1,
                                               iter, stepSize, step, Q, C, T, G);
          } else {
            // orthogonalize
            rank = normalizeCholQR (iter, step, Q, G);
          }
        } else { // Belos block OrthoManager + CholQR
          Teuchos::TimeMonitor LocalTimer (*orthTimer);

          this->projectBelosOrthoManager (iter, step, Q, G);
          rank = normalizeCholQR (iter, step, Q, G);
        } // End of orthogonalization

        { // convergence check
          //printf( " iter+step = %d+%d = %d, restart=%d\n",iter,step,iter+step,restart );
          if ((this->input_.orthoType != "CGS2x" &&
               this->input_.orthoType != "CGS2"  &&
               this->input_.orthoType != "CGS1"  &&
               //this->input_.orthoType != "CGS" && // single-reduce, without renormalization
               this->input_.orthoType != "MGS") ||
              iter+step >= restart || metric <= input.tol) {

            updateHessenburg (iter, step, output.ritzValues, H, G);

            // Check negative norm
            TEUCHOS_TEST_FOR_EXCEPTION
              (STS::real (H(iter+step, iter+step-1)) < STM::zero (),
               std::runtime_error, "At iteration " << output.numIters << ", H("
               << iter+step << ", " << iter+step-1 << ") = "
               << H(iter+step, iter+step-1) << " < 0.");

            // Convergence check
            if (rank == step+1 && H(iter+step, iter+step-1) != zero) {
              // Copy H to T and apply Givens rotations to new columns of T and y
              for (int iiter = 0; iiter < step; iiter++) {
                for (int i = 0; i <= iter+iiter+1; i++) {
                  T(i, iter+iiter) = H(i, iter+iiter);
                }
                this->reduceHessenburgToTriangular(iter+iiter, T, cs, sn, y);
              }
              metric = this->getConvergenceMetric (STS::magnitude (y(iter+step)), b_norm, input);
            }
            else {
              metric = STM::zero ();
            }

            //printf( " Convergence check(iter=%d, step=%d) metric=%.2e..\n",iter,step,metric );
            if (outPtr != nullptr) {
              // Update solution
              vec_type Y (B.getMap ());
              vec_type Z (B.getMap ());

              vec_type W1 (B.getMap ());
              vec_type W2 (B.getMap ());
              Tpetra::deep_copy (Y, X);
              dense_vector_type  z (iter+step, true);
              blas.COPY (iter+step, y.values(), 1, z.values(), 1);
              blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                         Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                         iter+step, 1, one,
                         T.values(), T.stride(), z.values(), z.stride());
              Teuchos::Range1D cols(0, iter+step-1);
              Teuchos::RCP<const MV> Qj = Q.subView(cols);
              dense_vector_type z_iter (Teuchos::View, z.values (), iter+step);
              if (input.precoSide == "right") {
                MVT::MvTimesMatAddMv (one, *Qj, z_iter, zero, W1);
                M.apply (W1, W2);
                Y.update (one, W2, one);
              }
              else {
                 MVT::MvTimesMatAddMv (one, *Qj, z_iter, one, Y);
              }
              A.apply (Y, Z);
              Z.update (one, B, -one);
              SC z_norm = Z.norm2 (); // residual norm
              *outPtr << "+ Current iteration: iter=" << output.numIters-1 << ", " << iter
                      << ", restart=" << restart
                      << ", metric=" << metric
                      << ", real resnorm=" << z_norm << " " << z_norm/b_norm
                      << endl;
            }
          } else {
            // convergence check (before re-normalization) in temporary spaces H2, T2, and y2
            // to avoid extra steps 
            updateHessenburg (iter, step, output.ritzValues, H2, G);

            // Check negative norm
            TEUCHOS_TEST_FOR_EXCEPTION
              (STS::real (H2(iter+step, iter+step-1)) < STM::zero (),
               std::runtime_error, "At iteration " << output.numIters << ", H("
               << iter+step << ", " << iter+step-1 << ") = "
               << H2(iter+step, iter+step-1) << " < 0.");

            // Convergence check
            if (rank == step+1 && H2(iter+step, iter+step-1) != zero) {
              // Copy H to T and apply Givens rotations to new columns of T and y
              for (int iiter = 0; iiter < step; iiter++) {
                for (int i = 0; i <= iter+iiter+1; i++) {
                  T2(i, iter+iiter) = H2(i, iter+iiter);
                }
                this->reduceHessenburgToTriangular(iter+iiter, T2, cs, sn, y2);
              }
              metric = this->getConvergenceMetric (STS::magnitude (y2(iter+step)), b_norm, input);
            }
            else {
              metric = STM::zero ();
            }

            // converged, copy the temporary matrices to the real ones
            if (metric <= input.tol) {
              for (int iiter = 0; iiter < step; iiter++) {
                  for (int i = 0; i <= iter+iiter+1; i++) {
                    T(i, iter+iiter) = T2(i, iter+iiter);
                  }
                  y[iter+iiter] = y2[iter+iiter];
              }
            }
          }
        } // End of convergence check
        //printf ("metric=%.2e, tol=%.2e, iter+step=%d\n\n", metric, input.tol, iter+step );
      } // End of restart cycle

      // Update solution
      blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                 Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                 iter, 1, one,
                 T.values(), T.stride(), y.values(), y.stride());
      Teuchos::Range1D cols(0, iter-1);

      dense_vector_type y_iter (Teuchos::View, y.values (), iter);
      Teuchos::RCP<const MV> Qj = Q.subView(cols);
      if (input.precoSide == "right") {
        MVT::MvTimesMatAddMv (one, *Qj, y_iter, zero, R);
        M.apply (R, MP);
        X.update (one, MP, one);
      }
      else {
        MVT::MvTimesMatAddMv (one, *Qj, y_iter, one, X);
      }
      // Compute real residual (not-preconditioned)
      P = * (Q.getVectorNonConst (0));
      A.apply (X, P);
      P.update (one, B, -one);
      r_norm = P.norm2 (); // residual norm
      output.absResid = r_norm;
      output.relResid = r_norm / b0_norm;
      //if (outPtr != nullptr) {
      //    *outPtr << "Residual norm at restart(iter=" << iter << ")=" << output.absResid << " " << output.relResid << endl;
      //}
      //printf( " restart(iter=%d, metrit=%.2e, r_norm=%.2e)..\n",iter,metric,r_norm );

      metric = this->getConvergenceMetric (r_norm, b0_norm, input);
      if (metric <= input.tol) {
        output.converged = true;
      }
      else if (output.numIters < input.maxNumIters) {
        // Initialize starting vector for restart
        if (input.precoSide == "left") { // left-precond'd residual norm
          Tpetra::deep_copy (R, P);
          M.apply (R, P);
          r_norm = P.norm2 ();
        }
        P.scale (one / r_norm);
        y[0] = SC {r_norm};
        y2[0] = SC {r_norm};
        for (int i=1; i < restart+1; ++i) {
          y[i] = STS::zero ();
          y2[i] = STS::zero ();
        }
        output.numRests++;
      }
    }

    // Return residual norm as B
    Tpetra::deep_copy (B, P);

    if (outPtr != nullptr) {
      *outPtr << "At end of solve:" << endl;
      Indent indentInner (outPtr);
      *outPtr << output;
    }
    return output;
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

    // copy: H(j:n-1, j:n-1) = R(j:n-1, j:n-1), i.e., H = R*B
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
    dense_matrix_type r_diag (Teuchos::View, R, s+1, s+1, n, 0);
    dense_matrix_type h_diag (Teuchos::View, H, s+1, s,   n, n);
    Teuchos::BLAS<LO, SC> blas;

    // H = H*R^{-1}
    if (n == 0) { // >> first matrix-power iteration <<
      // diagonal block
      blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS,
                 Teuchos::NON_UNIT_DIAG, s+1, s, one,
                 r_diag.values(), r_diag.stride(),
                 h_diag.values(), h_diag.stride());
    } else  { // >> rest of iterations <<
      for (int j = 1; j < s; j++ ) {
        H(n, n+j) -= H(n, n-1) * R(n-1, j);
      }
      // diagonal block
      blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS,
                 Teuchos::NON_UNIT_DIAG, s, s, one,
                 r_diag.values(), r_diag.stride(),
                 h_diag.values(), h_diag.stride());
      H(n+s, n+s-1) /= R(n+s-1, s-1);

      // upper off-diagonal block: H(0:j-1, j:j+n-2)
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
  normalizeCholQR (const int n,
                   const int s,
                   MV& Q,
                   dense_matrix_type& R)
  {
    // vector to be orthogonalized
    Teuchos::Range1D index_prev(n, n+s);
    MV Qnew = * (Q.subView(index_prev));

    dense_matrix_type r_new (Teuchos::View, R, s+1, s+1, n, 0);

    int rank = 0;
    if (tsqr_ != Teuchos::null) {
      rank = tsqr_->factor (Qnew, r_new);
    }
    else {
      rank = this->normalizeBelosOrthoManager (Qnew, r_new);
    }
    return rank;
  }


  //! Apply the orthogonalization using Belos' OrthoManager
  int
  reNormalizeCholQR2 (const int iterPrev, // starting index of columns for re-orthogonalization 
                      const int stepSize, // number of columns for re-orthogonalization
                      const int step,     // number of new columns, need to be re-scaled
                      MV& Q,
                      dense_matrix_type& C, // store aggregated coefficient, results of block dot-products
                      dense_matrix_type& G) // Hessenburg matrix
  {
    const SC one  = STS::one  ();

    Teuchos::BLAS<LO, SC> blas;
    Teuchos::LAPACK<LO, SC> lapack;

    int rank = 0;
    int iter = iterPrev+stepSize;

    //printf( " reNormalizeCholQR2(%d:%d, iterPrev=%d, iter=%d, stepSize=%d, step=%d..\n",iterPrev,iter, iterPrev,iter,stepSize,step );
    // re-normalize the previous s-step set of vectors (lagged)
    // making a copy of C(iterPrev,iterPrev):
    // G still contains coeff from previous step for convergence check
    dense_matrix_type Rfix (stepSize, stepSize, true);
    for (int i=0; i < stepSize; i++) {
      for (int j=i; j < stepSize; j++) {
        Rfix(i, j) = C(iterPrev+i, iterPrev+j);
      }
    }

    // Compute the Cholesky factorization of R in place
    int info = 0;
    lapack.POTRF ('U', stepSize, Rfix.values (), Rfix.stride(), &info);
    if (info < 0) {
      // FIXME (mfh 17 Sep 2018) Don't throw; report an error code.
      rank = info;
      throw std::runtime_error("Cholesky factorization failed");
    } else {
      rank = stepSize;
    }

    // Compute A_cur / R (Matlab notation for A_cur * R^{-1}) in place.
    Teuchos::Range1D index_old(iterPrev, iter-1);
    MV Qold = * (Q.subView(index_old));

    Qold.template sync<Kokkos::HostSpace> ();
    Qold.template modify<Kokkos::HostSpace> ();
    auto Q_lcl = Qold.template getLocalView<Kokkos::HostSpace> ();
    SC* const Q_lcl_raw = reinterpret_cast<SC*> (Q_lcl.data ());

    // rescale the previous vector, MVT::MvScale (Qn, one / tnn);
    const LO LDQ = LO (Qold.getStride ());
    const LO ncols = Qold.getNumVectors ();
    const LO nrows = Qold.getLocalLength ();
    blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
               nrows, ncols,
               one, Rfix.values(), Rfix.stride(),
                    Q_lcl_raw, LDQ);

    // merge two R
    // H(n, n-1) *= tnn;
    dense_matrix_type Rold (Teuchos::View, G, stepSize, stepSize, iterPrev, 0);
    blas.TRMM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
               stepSize, stepSize,
               one, Rfix.values(), Rfix.stride(),
                    Rold.values(), Rold.stride());

    // update coefficients
    // for (int i = 0; i < n; i++) T(i, n) /= tnn;
    dense_matrix_type c1(Teuchos::View, C, iter, stepSize, 0, iterPrev);
    blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
               iter+1, stepSize,
               one, Rfix.values(), Rfix.stride(),
                    c1.values(), c1.stride());
    // H(n, n) /= tnn;
    dense_matrix_type c2 (Teuchos::View, C, stepSize, step, iterPrev, iterPrev);
    blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
               Teuchos::TRANS, Teuchos::NON_UNIT_DIAG,
               stepSize, stepSize+step+1,
               one, Rfix.values(), Rfix.stride(),
                    c2.values(), c2.stride());

    Qold.template sync<typename MV::device_type::memory_space> ();
    return rank;
  }

  //! Apply the orthogonalization using Belos' OrthoManager
  int
  projectAndNormalizeCholQR2 (const int numIters,
                              const int iter,
                              const int stepSize,
                              const int step,
                              MV& Q,
                              dense_matrix_type& C, // store aggregated coefficient, results of block dot-products
                              dense_matrix_type& T, // store Q'*Q
                              dense_matrix_type& G) // Hessenburg matrix
  {
    const SC zero = STS::zero ();
    const SC one  = STS::one  ();

    Teuchos::BLAS<LO, SC> blas;
    Teuchos::LAPACK<LO, SC> lapack;

    int rank = 0;
    int iterPrev = iter-stepSize;

    //printf( " projectAndNormalizeCholQR2\n" );
    if (iter > 0) {
      // extract new coefficients (note: C(:,iter) is used for T(:,iter) and G(:,0))
      for (int i = 0; i < iter+step+1; i++) {
        for (int j = 0; j < step+1; j++) {
          G(i, j) = C(i, iter+j);
        }
      }

      dense_matrix_type Gnew (Teuchos::View, G, iter, step+1, 0, 0);
      dense_matrix_type Cnew (iter, step+1, true);
      if (this->input_.orthoType != "CGS") {
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
            C(iterPrev+i, iterPrev+j) = zero;
          }
        }
        #else
        for (int i=0; i < stepSize; i++) {
          //C(iterPrev+i, iterPrev+i) = zero;
          C(iterPrev+i, iterPrev+i) -= one;
        }
        #endif

        // expand T
        for (int j=iterPrev; j <= iter; j++) {
          for (int i=0; i < iterPrev; i++) C(j, i) = C(i, j);
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
        if (this->input_.orthoType == "MGS") {
            // H := (I+L)^(-1)H
            blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI,
                       Teuchos::NO_TRANS, Teuchos::UNIT_DIAG,
                       iter, step+1,
                       one, C.values(), C.stride(),
                            Gnew.values(), Gnew.stride());
        } else if (this->input_.orthoType == "CGS2x" ||
                   this->input_.orthoType == "CGS2" ) {
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

      MVT::MvTimesMatAddMv(-one, *Qprev, Gnew, one, Qnew);

      // the scaling factor
      dense_matrix_type Rnew (Teuchos::View, G, step+1, step+1, iter, 0);
      if (this->input_.orthoType != "CGS") {
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
      } else {
        blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                  step+1, step+1, iter,
                 -one, Gnew.values(), Gnew.stride(),
                       Gnew.values(), Gnew.stride(),
                  one, Rnew.values(), Rnew.stride());
      }

      // -------------------------
      // normalize the new vectors
      // -------------------------
      {
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
        int info = 0;
        lapack.POTRF ('U', step+1, Rnew.values (), Rnew.stride(), &info);
        if (info < 0) {
          // FIXME (mfh 17 Sep 2018) Don't throw; report an error code.
          throw std::runtime_error("Cholesky factorization failed");
        }
        for (int i=0; i<step+1; i++) {
          for (int j=0; j<i; j++) {
            Rnew(i, j) = zero;
          }
        }

        // Compute A_cur / R (Matlab notation for A_cur * R^{-1}) in place.
        Qnew.template sync<Kokkos::HostSpace> ();
        Qnew.template modify<Kokkos::HostSpace> ();
        auto Q_lcl = Qnew.template getLocalView<Kokkos::HostSpace> ();
        SC* const Q_lcl_raw = reinterpret_cast<SC*> (Q_lcl.data ());
        const LO LDQ = LO (Qnew.getStride ());

        LO ncols = Qnew.getNumVectors ();
        LO nrows = Qnew.getLocalLength ();
        blas.TRSM (Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI,
                   Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                   nrows, ncols, one, Rnew.values(), Rnew.stride(),
                   Q_lcl_raw, LDQ);
        Qnew.template sync<typename MV::device_type::memory_space> ();
        rank = ncols;
      }

      if (this->input_.orthoType == "CGS2x") {
        // reorthogonalize against previous vectors
        dense_matrix_type Greo (iter, step+1, true);
        MVT::MvTransMv(one, *Qprev, Qnew, Greo);

        MVT::MvTimesMatAddMv(-one, *Qprev, Greo, one, Qnew);
        for (int i=0; i<iter; i++) {
          for (int j=0; j<step+1; j++) Gnew(i, j) += Greo(i, j);
        }
      }
    } else {
      rank = normalizeCholQR (iter, step, Q, G);
    }

    return rank;
  }

private:
  int stepSize_;
  bool useCholQR2_;
  Teuchos::RCP<CholQR<SC, MV, OP> > tsqr_;
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
