#ifndef BELOS_TPETRA_GMRES_SINGLE_REDUCE_HPP
#define BELOS_TPETRA_GMRES_SINGLE_REDUCE_HPP

#include "Belos_Tpetra_Gmres.hpp"
#include "Belos_Tpetra_UpdateNewton.hpp"

namespace BelosTpetra {
namespace Impl {

template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<SC>,
         class OP = Tpetra::Operator<SC>>
class GmresSingleReduce : public Gmres<SC, MV, OP> {
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

public:
  GmresSingleReduce () :
    base_type::Gmres (),
    stepSize_ (1)
  {
    this->input_.computeRitzValues = true;
  }

  GmresSingleReduce (const Teuchos::RCP<const OP>& A) :
    base_type::Gmres (A),
    stepSize_ (1)
  {
    this->input_.computeRitzValues = true;
  }

  virtual ~GmresSingleReduce ()
  {}

  virtual void
  setParameters (Teuchos::ParameterList& params) {
    Gmres<SC, MV, OP>::setParameters (params);

    int stepSize = stepSize_;
    if (params.isParameter ("Step Size")) {
      stepSize = params.get<int> ("Step Size");
    }
    stepSize_ = stepSize;
  }

  void
  setStepSize(int stepSize) {
    stepSize_ = stepSize;
  }

  int
  getStepSize() {
    return stepSize_;
  }
  
protected:
  virtual void
  setOrthogonalizer (const std::string& ortho)
  {
    if (ortho != "CGS2" && ortho != "MGS") {
      Gmres<SC, MV, OP>::setOrthogonalizer (ortho);
    }
  }

  //! Apply the orthogonalization using a single all-reduce
  mag_type
  projectCgsSingleReduce (int n,
                          const SolverInput<SC>& input, 
                          MV& Q,
                          dense_matrix_type& H,
                          dense_matrix_type& WORK) const
  {
    Teuchos::BLAS<LO, SC> blas;
    const SC one = STS::one ();

    Teuchos::Range1D index_all(0, n+1);
    Teuchos::Range1D index_prev(0, n);
    const MV Qall  = * (Q.subView(index_all));
    const MV Qprev = * (Q.subView(index_prev));

    // Q(:,0:j+1)'*Q(:,j+1)
    vec_type AP = * (Q.getVectorNonConst (n+1));
    dense_matrix_type h_all (Teuchos::View, H, n+2, 1, 0, n);
    MVT::MvTransMv(one, Qall, AP, h_all);

    // orthogonalize (project)
    dense_matrix_type h_prev (Teuchos::View, H, n+1, 1, 0, n);
    MVT::MvTimesMatAddMv (-one, Qprev, h_prev, one, AP);

    // fix the norm
    mag_type oldNorm = STS::real (H(n+1, n));
    for (int i = 0; i <= n; ++i) {
      H(n+1, n) -= (H(i, n)*H(i, n));
    }

    // reorthogonalize if requested
    if (input.needToReortho) {
      // Q(:,0:j+1)'*Q(:,j+1)
      MVT::MvTransMv(one, Qall, AP, WORK);
      // orthogonalize (project)
      MVT::MvTimesMatAddMv (-one, Qprev, WORK, one, AP);
      // recompute the norm
      for (int i = 0; i <= n; ++i) {
        WORK(n+1, 0) -= (WORK(i, 0)*WORK(i, 0));
      }

      // accumulate results
      blas.AXPY (n+1, one, WORK.values (), 1, h_prev.values (), 1);
      H(n+1, n) = WORK(n+1, 0); 
    }

    return oldNorm;
  }

  //! Apply the orthogonalization using a single all-reduce
  mag_type
  projectIGsSingleReduce (int n,
                          const SolverInput<SC>& input, 
                          MV& Q,
                          dense_matrix_type& H,
                          dense_matrix_type& T) const
  {
    Teuchos::BLAS<LO, SC> blas;
    const SC one = STS::one ();
    const SC zero = STS::zero ();

    Teuchos::Range1D index_all(0, n+1);
    Teuchos::Range1D index_prev(0, n);
    const MV Qall  = * (Q.subView(index_all));
    const MV Qprev = * (Q.subView(index_prev));

    // c = Q(:, 0:n+1)'*Q(:, n:n+1)
    // (note: Q(:,n+1)'*Q(:,n) is not needed)
    dense_matrix_type c (n+2, 2);
    Teuchos::Range1D index(n, n+1);
    const MV q  = * (Q.subView(index));
    MVT::MvTransMv(one, Qall, q, c);

    // T(1:n, n) = c(:,0)
    for (int i=0; i<n+1; i++) T(i, n) = c(i,0);
    // H(1:n+1, n) = c(:,1)
    for (int i=0; i<n+2; i++) H(i, n) = c(i,1);

    vec_type AP = * (Q.getVectorNonConst (n+1));
    vec_type Qn = * (Q.getVectorNonConst (n));

    // rescaling the previous vector
    const mag_type tnn = STM::squareroot (STS::real (T(n, n)));
    MVT::MvScale (Qn, one / tnn);
    // update T
    for (int i = 0; i < n; i++) T(i, n) /= tnn;
    T(n, n) /= T(n, n);
    // update H
    H(n, n) /= tnn;
    if (n > 0) {
      H(n, n-1) *= tnn;
    }

    // copy T(n, 0:n-1) = T(0:n-1, n);
    for (int i = 0; i < n; i++) T(n, i) = T(i, n);
    T(n, n) -= one; // T = Q'*Q - I
    //mag_type maxT = std::abs(T(0, 0));
    //for (int i = 0; i < n; i++) 
    //  for (int j = 0; j <= i; j++) 
    //    maxT = std::max(maxT, std::abs(T(i, j)));
    //std::cout << "max:" << maxT << " " << tnn << std::endl;

    // save original coefficient, Q(:,0:n+1)'*Q(:,n+1)
    dense_vector_type h0 (n+1);
    for (int i = 0; i < n+1; i++) {
      h0(i) = H(i, n); // original
    }

    dense_vector_type t (n+1);
    dense_matrix_type h_prev (Teuchos::View, H, n+1, 1, 0, n);
    if (this->input_.orthoType == "MGS") {
      // compute H(0:n, n) := (I+L)^{-1} H(0:n, n)
      blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI,
                 Teuchos::NO_TRANS, Teuchos::UNIT_DIAG,
                 n+1, 1,
                 one, T.values(), T.stride(), 
                      h_prev.values(), h_prev.stride());
    } else if (this->input_.orthoType == "CGS2") {
      // compute H(0:n, n) := (I-T) * H(0:n, n), T~=I stores T-I, (for implicitly re-orthogonalize)
      blas.GEMV(Teuchos::NO_TRANS, n+1, n+1,
                -one, T.values(), T.stride(), 
                      h_prev.values(), 1,
                zero, t.values(), 1);

      //#define SPLIT_REORTH_SINGLEREDUCE
      #if defined(SPLIT_REORTH_SINGLEREDUCE)
      // orthogonalize (project)
      MVT::MvTimesMatAddMv (-one, Qprev, h_prev, one, AP);

      // re-orthogonalize (project)
      MVT::MvTimesMatAddMv (-one, Qprev, t, one, AP);
      #endif

      //update H(0:n, n)
      for (int i = 0; i < n+1; i++) {
        H(i, n) += t(i);
      }
    }

    // make a copy of the updated coefficient
    for (int i = 0; i < n+1; i++) {
      t(i) = H(i, n);  // updated
    }
    #if !defined(SPLIT_REORTH_SINGLEREDUCE)
    // orthogonalize+reortho
    MVT::MvTimesMatAddMv (-one, Qprev, h_prev, one, AP);
    #endif

    // fix the norm, and update H(0:n, n)
    mag_type oldNorm = STS::real (H(n+1, n));
    #if 0
    blas.GEMV(Teuchos::NO_TRANS, n+1, n+1,
              one,  T.values(), T.stride(), 
                    h_prev.values(), 1,
              zero, t.values(), 1);
    for (int i = 0; i <= n; ++i) {
      H(n+1, n) -= (H(i, n) * (H(i, n) - t(i)));
    }
    #else
    blas.GEMV(Teuchos::NO_TRANS, n+1, n+1,
              one, T.values(), T.stride(), 
                   h_prev.values(), 1,
              one, t.values(), 1);
    for (int i = 0; i <= n; ++i) {
      H(n+1, n) -= (H(i, n) * ((h0(i)+h0(i)) - t(i)));
    }
    #endif
    return oldNorm;
  }

  //! Apply the orthogonalization using a single all-reduce
  int
  projectAndNormalizeSingleReduce (int n,
                                   const SolverInput<SC>& input, 
                                   MV& Q,
                                   dense_matrix_type& H,
                                   dense_matrix_type& T,
                                   dense_matrix_type& WORK) const
  {
    mag_type oldNorm;
    if (this->input_.orthoType == "MGS" ||
        this->input_.orthoType == "CGS2") {
        oldNorm = projectIGsSingleReduce (n, input, Q, H, T);
    } else{
        oldNorm = projectCgsSingleReduce (n, input, Q, H, WORK);
    }

    // check for negative norm
    TEUCHOS_TEST_FOR_EXCEPTION
      (STS::real (H(n+1, n)) < STM::zero (), std::runtime_error, "At iteration "
       << n << ", H(" << n+1 << ", " << n << ") = " << H(n+1, n) << " < 0.");
    // Check for zero norm.  OK to take real part of H(n+1, n), since
    // this is a magnitude (cosine) anyway and therefore should always
    // be real.
    int rank = 0;
    const mag_type eps  = STS::eps ();
    const mag_type tolOrtho = static_cast<mag_type> (10.0) * eps;

    const mag_type H_np1_n = STS::real (H(n+1, n));
    if (H_np1_n > oldNorm * tolOrtho) {
      const mag_type H_np1_n_sqrt = STM::squareroot (H_np1_n);
      vec_type AP = * (Q.getVectorNonConst (n+1));
      H(n+1, n) = SC {H_np1_n_sqrt};
      MVT::MvScale (AP, STS::one () / H(n+1, n)); // normalize
      rank = 1;
    }
    else {
      H(n+1, n) = STS::zero ();
      rank = 0;
    }
    return rank;
  }

  SolverOutput<SC>
  solveOneVec (Teuchos::FancyOStream* outPtr,
               vec_type& X, // in X/out X
               vec_type& B, // in B
               const OP& A,
               const OP& M,
               const SolverInput<SC>& input)
  {
    using std::endl;
    int restart = input.resCycle;
    const SC zero = STS::zero ();
    const SC one  = STS::one ();
    const bool computeRitzValues = input.computeRitzValues;

    SolverOutput<SC> output {};
    // initialize output parameters
    output.numRests = 0;
    output.numIters = 0;
    output.converged = false;

    mag_type b_norm;  // initial residual norm
    mag_type b0_norm; // initial residual norm, not left-preconditioned
    mag_type r_norm;
    MV Q (B.getMap (), restart+1);
    vec_type P = * (Q.getVectorNonConst (0));
    vec_type R (B.getMap ());
    vec_type MP (B.getMap ());

    Teuchos::BLAS<LO ,SC> blas;
    dense_matrix_type T (restart+1, restart, true);
    dense_matrix_type H (restart+1, restart, true);
    dense_matrix_type h (restart+1, 1, true);
    dense_vector_type y (restart+1, true);
    std::vector<mag_type> cs (restart);
    std::vector<SC> sn (restart);

    // initial residual (making sure R = B - Ax)
    A.apply (X, R);
    R.update (one, B, -one);
    b0_norm = R.norm2 (); // initial residual norm, no-preconditioned
    if (input.precoSide == "left") {
      M.apply (R, P);
      r_norm = P.norm2 (); // initial residual norm, left-preconditioned
    } else {
      r_norm = b0_norm;
    }
    b_norm = r_norm;

    mag_type metric = this->getConvergenceMetric (b0_norm, b0_norm, input);
    if (metric <= input.tol) {
      if (outPtr != NULL) {
        *outPtr << "Initial guess' residual norm " << b0_norm
                << " meets tolerance " << input.tol << endl;
      }
      output.absResid = b0_norm;
      output.relResid = STM::one ();
      output.converged = true;
      return output;
    }

    // compute Ritz values as Newton shifts
    if (computeRitzValues) {
      // Invoke ordinary GMRES for the first restart
      SolverInput<SC> input_gmres = input;
      input_gmres.maxNumIters = input.resCycle;
      input_gmres.computeRitzValues = computeRitzValues;

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
    }

    // initialize starting vector
    if (input.precoSide != "left") {
      Tpetra::deep_copy (P, R);
    }
    P.scale (one / r_norm);
    y[0] = SC {r_norm};
    const int s = getStepSize ();

    // main loop
    while (output.numIters < input.maxNumIters && ! output.converged) {
      int iter = 0;
      if (input.maxNumIters < output.numIters+restart) {
        restart = input.maxNumIters-output.numIters;
      }
      // restart cycle
      for (iter = 0; iter < restart && metric > input.tol; ++iter) {
        int iiter = iter;
        if (this->input_.orthoType == "MGS" ||
            this->input_.orthoType == "CGS2") {
          iiter --;
        }
        if (outPtr != nullptr) {
          // Update X
          vec_type Y (B.getMap ());
          vec_type Z (B.getMap ());
          Tpetra::deep_copy (Y, X);
          if (iiter > 0) {
            dense_vector_type  z (iiter, true);
            blas.COPY (iiter, y.values(), 1, z.values(), 1);
            blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                       Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                       iiter, 1, one,
                       H.values(), H.stride(), z.values(), z.stride());
            Teuchos::Range1D cols(0, iiter-1);
            Teuchos::RCP<const MV> Qj = Q.subView(cols);
            dense_vector_type z_iter (Teuchos::View, z.values (), iiter);
            MVT::MvTimesMatAddMv (one, *Qj, z_iter, one, Y);
          }
          A.apply (Y, Z);
          Z.update (one, B, -one);
          SC z_norm = Z.norm2 (); // residual norm

          *outPtr << "Current iteration: iter=" << iter << "->" << iiter
                  << ", restart=" << restart
                  << ", metric=" << metric 
                  << ", real resnorm=" << z_norm << " " << z_norm/b_norm
                  << endl;
        }
        // AP = A*P
        vec_type P  = * (Q.getVectorNonConst (iter));
        vec_type AP = * (Q.getVectorNonConst (iter+1));
        if (input.precoSide == "none") { // no preconditioner
          A.apply (P, AP);
        }
        else if (input.precoSide == "right") {
          M.apply (P, MP);
          A.apply (MP, AP);
        }
        else {
          A.apply (P, MP);
          M.apply (MP, AP);
        }
        // Shift for Newton basis
        if (computeRitzValues) {
          //AP.update (-output.ritzValues[iter],  P, one);
          const complex_type theta = output.ritzValues[iter % s];
          UpdateNewton<SC, MV>::updateNewtonV (iter, Q, theta);
        }
        output.numIters++; 

        // Orthogonalization
        projectAndNormalizeSingleReduce (iter, input, Q, H, T, h);
        // Shift back for Newton basis
        if (computeRitzValues) {
          // H(iter, iter) += output.ritzValues[iter];
          const complex_type theta = output.ritzValues[iter % s];
          UpdateNewton<SC, MV>::updateNewtonH (iter, H, theta);
        }

        // Convergence check
        if (iiter >= 0) {
          if (H(iiter+1, iiter) != zero) {
            // Apply Givens rotations to new column of H and y
            this->reduceHessenburgToTriangular (iiter, H, cs, sn, y.values ());
            // Convergence check
            metric = this->getConvergenceMetric (STS::magnitude (y[iiter+1]),
                                                 b_norm, input);
          }
          else {
            H(iiter+1, iiter) = zero;
            metric = STM::zero ();
          }
        }
      } // end of restart cycle 
      if (this->input_.orthoType == "MGS" ||
          this->input_.orthoType == "CGS2") {
        int iiter = iter - 1;
        if (iiter >= 0) {
          if (H(iiter+1, iiter) != zero) {
            // Apply Givens rotations to new column of H and y
            this->reduceHessenburgToTriangular (iiter, H, cs, sn, y.values ());
            // Convergence check
            metric = this->getConvergenceMetric (STS::magnitude (y[iiter+1]),
                                                 b_norm, input);
          }
          else {
            H(iiter+1, iiter) = zero;
            metric = STM::zero ();
          }
        }
      }
      if (iter > 0) {
        // Update solution
        blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS,
                   Teuchos::NON_UNIT_DIAG, iter, 1, one,
                   H.values(), H.stride(), y.values (), y.stride ());
        Teuchos::Range1D cols(0, iter-1);
        Teuchos::RCP<const MV> Qj = Q.subView(cols);
        y.resize (iter);
        if (input.precoSide == "right") {
          MVT::MvTimesMatAddMv (one, *Qj, y, zero, R);
          M.apply (R, MP);
          X.update (one, MP, one);
        }
        else {
          MVT::MvTimesMatAddMv (one, *Qj, y, one, X);
        }
        y.resize (restart+1);
      }

      // Compute explicit residual vector in preparation for restart.
      P = * (Q.getVectorNonConst (0));
      A.apply (X, P);
      P.update (one, B, -one);
      r_norm = P.norm2 (); // residual norm
      output.absResid = r_norm;
      output.relResid = r_norm / b0_norm;

      metric = this->getConvergenceMetric (r_norm, b0_norm, input);
      if (metric <= input.tol) {
        output.converged = true;
        return output;
      }
      else if (output.numIters < input.maxNumIters) {
        // Initialize starting vector for restart
        if (input.precoSide == "left") {
          Tpetra::deep_copy (R, P);
          M.apply (R, P);
          // FIXME (mfh 14 Aug 2018) Didn't we already compute this above?
          r_norm = P.norm2 ();
        }
        P.scale (one / r_norm);
        y[0] = SC {r_norm};
        for (int i=1; i < restart+1; i++) {
          y[i] = zero;
        }
        output.numRests++; // restart
      }
    }

    return output;
  }
  
  int stepSize_; // "step size" for Newton basis
};

template<class SC, class MV, class OP,
         template<class, class, class> class KrylovSubclassType>
class SolverManager;

// This is the Belos::SolverManager subclass that gets registered with
// Belos::SolverFactory.
template<class SC, class MV, class OP>
using GmresSingleReduceSolverManager = SolverManager<SC, MV, OP, GmresSingleReduce>;

/// \brief Register GmresSingleReduceSolverManager for all enabled Tpetra
///   template parameter combinations.
void register_GmresSingleReduce (const bool verbose);

} // namespace Impl
} // namespace BelosTpetra

#endif // BELOS_TPETRA_GMRES_SINGLE_REDUCE_HPP
