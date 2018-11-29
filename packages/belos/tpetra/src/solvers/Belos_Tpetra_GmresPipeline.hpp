#ifndef BELOS_TPETRA_GMRES_PIPELINE_HPP
#define BELOS_TPETRA_GMRES_PIPELINE_HPP

#include "Belos_Tpetra_Gmres.hpp"
#include "Belos_Tpetra_UpdateNewton.hpp"
#include "Tpetra_idot.hpp"

namespace BelosTpetra {
namespace Impl {

template<class SC = Tpetra::Operator<>::scalar_type,
         class MV = Tpetra::MultiVector<SC>,
         class OP = Tpetra::Operator<SC>>
class GmresPipeline : public Gmres<SC, MV, OP> {
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
  GmresPipeline () :
    base_type::Gmres ()
  {
    this->input_.computeRitzValues = true;
  }

  GmresPipeline (const Teuchos::RCP<const OP>& A) :
    base_type::Gmres (A)
  {
    this->input_.computeRitzValues = true;
  }

  virtual ~GmresPipeline ()
  {}

protected:
  virtual void
  setOrthogonalizer (const std::string& ortho)
  {
    if (ortho != "CGS2" && ortho != "MGS") {
      Gmres<SC, MV, OP>::setOrthogonalizer (ortho);
    }
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
    int ell = 1;
    const SC zero = STS::zero ();
    const SC one  = STS::one ();
    const mag_type eps = STS::eps ();
    const mag_type tolOrtho = mag_type (10.0) * STM::squareroot (eps);
    const bool computeRitzValues = input.computeRitzValues;

    // initialize output parameters
    SolverOutput<SC> output {};
    output.converged = false;
    output.numRests = 0;
    output.numIters = 0;

    Teuchos::BLAS<LO ,SC> blas;

    mag_type b_norm; // initial residual norm
    mag_type b0_norm; // initial residual norm, not left-preconditioned
    mag_type r_norm;
    dense_matrix_type  C (restart+1, 2, true);
    dense_matrix_type  G (restart+1, restart+1, true);
    dense_matrix_type  H (restart+1, restart,   true);
    dense_matrix_type  T (restart+1, restart+1, true);
    dense_vector_type  y (restart+1, true);
    std::vector<mag_type> cs (restart);
    std::vector<SC> sn (restart);
    MV  Q (B.getMap (), restart+1);
    MV  V (B.getMap (), restart+1);
    vec_type Z = * (V.getVectorNonConst (0));
    vec_type R (B.getMap ());
    vec_type MZ (B.getMap ());

    // initial residual (making sure R = B - Ax)
    A.apply (X, R);
    R.update (one, B, -one);
    // TODO: this should be idot?
    b0_norm = STM::squareroot (STS::real (R.dot (R))); // initial residual norm, no preconditioned
    if (input.precoSide == "left") {
      M.apply (R, Z);
      // TODO: this should be idot?
      b_norm = STS::real (Z.dot( Z )); //Z.norm2 (); // initial residual norm, preconditioned
    }
    else {
      b_norm = b0_norm;
    }
    r_norm = b_norm;

    if (computeRitzValues) {
      // Invoke standard Gmres for the first restart cycle, to compute
      // Ritz values as Newton shifts
      SolverInput<SC> input_gmres = input;
      input_gmres.maxNumIters = input.resCycle;
      input_gmres.computeRitzValues = true;

      Tpetra::deep_copy (R, B);
      output = Gmres<SC, MV, OP>::solveOneVec (outPtr, X, R, A, M,
                                               input_gmres);
      if (output.converged) {
        return output; // standard GMRES converged
      }
      if (input.precoSide == "left") {
        M.apply (R, Z);
        r_norm = Z.norm2 (); // residual norm
      }
      else {
        r_norm = output.absResid;
      }
      output.numRests++;
    }
    if (input.precoSide != "left") {
      Tpetra::deep_copy (Z, R);
    }

    // for idot
    std::shared_ptr<Tpetra::Details::CommRequest> req;
    Kokkos::View<dot_type*, device_type> vals ("results[numVecs]",
                                               restart+1);
    auto vals_h = Kokkos::create_mirror_view (vals);

    #define USE_IDOT_FOR_CGS2
    #ifdef USE_IDOT_FOR_CGS2
    std::shared_ptr<Tpetra::Details::CommRequest> req2;
    Kokkos::View<dot_type*, device_type> vals2 ("results2[numVecs]",
                                               restart+1);
    auto vals2_h = Kokkos::create_mirror_view (vals2);
    #endif

    // Initialize starting vector
    //Z.scale (one / b_norm);
    G(0, 0) = r_norm*r_norm;
    y[0] = r_norm;

    // Main loop
    mag_type metric = 2*input.tol; // to make sure to hit the first synch
    while (output.numIters < input.maxNumIters && ! output.converged) {
      int iter = 0;
      if (input.maxNumIters < output.numIters+restart) {
        restart = input.maxNumIters-output.numIters;
      }

      // Normalize initial vector
      MVT::MvScale (Z, one/std::sqrt(G(0, 0)));

      // Copy initial vector
      vec_type AP = * (Q.getVectorNonConst (0));
      Tpetra::deep_copy (AP, Z);

      // Restart cycle
      for (iter = 0; iter < restart+ell && metric > input.tol; ++iter) {
        if (iter < restart) {
          // W = A*Z
          vec_type Z = * (V.getVectorNonConst (iter));
          vec_type W = * (V.getVectorNonConst (iter+1));
          if (input.precoSide == "none") {
            A.apply (Z, W);
          }
          else if (input.precoSide == "right") {
            M.apply (Z, MZ);
            A.apply (MZ, W);
          }
          else {
            A.apply (Z, MZ);
            M.apply (MZ, W);
          }
          // Shift for Newton basis, explicitly for the first iter
          // (rest is done through change-of-basis)
          if (computeRitzValues && iter == 0) {
            const complex_type theta = output.ritzValues[iter%ell];
            UpdateNewton<SC, MV>::updateNewtonV (iter, V, theta);
          }
          output.numIters ++;
        }
        int k = iter - ell; // we synch idot from k-th iteration

        // Compute G and H
        if (k >= 0) {
          if (this->input_.orthoType == "MGS" ||
              this->input_.orthoType == "CGS2") {
            // low-synchronous CGS2
            #ifdef USE_IDOT_FOR_CGS2
            req->wait (); // wait for idot
            req2->wait (); // wait for idot

            Kokkos::deep_copy (vals_h, vals);
            for (int i = 0; i <= k; i++) {
              T(i, k) = vals_h[i];
            }
            Kokkos::deep_copy (vals2_h, vals2);
            for (int i = 0; i <= k+1; i++) {
              G(i, k+1) = vals2_h[i];
            }
            #else
            for (int i = 0; i <= k; i++) {
              T(i, k) = C(i, 0);
            }
            for (int i = 0; i <= k+1; i++) {
              G(i, k+1) = C(i, 1);
            }
            #endif

            // rescaling the previous vector
            vec_type Vn = * (V.getVectorNonConst (k));
            vec_type Qn = * (Q.getVectorNonConst (k));
            const mag_type tkk = STM::squareroot (STS::real (T(k, k)));
            MVT::MvScale (Vn, one / tkk);
            MVT::MvScale (Qn, one / tkk);
            // update T
            for (int i = 0; i < k; i++) T(i, k) /= tkk;
            T(k, k) /= T(k, k);
            // update G
            G(k, k+1) /= tkk;
            G(k, k) *= tkk;
            if (k > 0) {
              H(k, k-1) *= tkk;
            }

            // copy T(k, 0:k-1) = T(0:k-1, k);
            for (int i = 0; i < k; i++) T(k, i) = T(i, k);
            T(k, k) -= one; // T = Q'*Q - I
mag_type maxT = std::abs(T(0, 0));
for (int i = 0; i < k; i++) {
  for (int j = 0; j <= i; j++) {
    maxT = std::max(maxT, std::abs(T(i, j)));
    //printf( " %.2e ",T(i,j) );
  }
  //printf( "\n" );
}
std::cout << "max:" << maxT << " " << tkk << std::endl;

            // save original coefficients, Q(:,0:n+1)'*Q(:,n+1)
            dense_vector_type h0 (k+1);
            for (int i = 0; i < k+1; i++) {
              h0(i) = G(i, k+1); // original
            }

            dense_vector_type t (k+1);
            dense_matrix_type g_prev (Teuchos::View, G, k+1, 1, 0, k+1);
            if (this->input_.orthoType == "MGS") {
              // compute G(0:k, k+1) := (I+L)^{-1} G(0:k, k+1)
              blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI,
                         Teuchos::NO_TRANS, Teuchos::UNIT_DIAG,
                         k+1, 1,
                         one, T.values(), T.stride(),
                              g_prev.values(), g_prev.stride());
            } else if (this->input_.orthoType == "CGS2") {
              // update G(0:k, k+1) := (I-T) * G(0:k, k+1), T~=I stores T-I, (for implicitly re-orthogonalize)
              blas.GEMV(Teuchos::NO_TRANS, k+1, k+1,
                        -one, T.values(), T.stride(),
                              g_prev.values(), 1,
                        zero, t.values(), 1);

              #define REORTHO
              #ifdef REORTHO
              //update G(0:k, k)
              for (int i = 0; i < k+1; i++) {
                G(i, k+1) += t(i);
              }
              #endif
            }

            // make a copy of the updated coefficient
            for (int i = 0; i < k+1; i++) {
              t(i) = G(i, k+1);
            }

            // copy G into H
            for (int i = 0; i <= k+1; i++) {
              H(i, k) = G(i, k+1);
//printf( " H(%d, %d) = %.2e\n",i,k,H(i,k));
            }


            // merge re-orthogonalization into the coefficient H
#ifdef REORTHO
            mag_type oldNorm = STS::real (H(k+1, k));
            dense_matrix_type h_prev (Teuchos::View, H, k+1, 1, 0, k);
            #if 1
            blas.GEMV(Teuchos::NO_TRANS, k+1, k+1,
                      one, T.values(), T.stride(),
                           h_prev.values(), 1,
                      one, t.values(), 1);
            for (int i = 0; i <= k; ++i) {
              H(k+1, k) -= (H(i, k) * (h0(i)+(h0(i)- t(i))));
            }
            #else
            blas.GEMV(Teuchos::NO_TRANS, k+1, k+1,
                      one,  T.values(), T.stride(),
                            h_prev.values(), 1,
                      zero, t.values(), 1);
            for (int i = 0; i <= k; ++i) {
              H(k+1, k) -= (H(i, k) * (H(i, k) - t(i)));
            }
            #endif
//printf( " H(%d, %d) = %.2e\n",k+1,k,H(k+1,k));
#else
            // fix the norm, and update H(0:n, n)
            for (int i = 0; i <= k; ++i) {
              H(k+1, k) -= G(i, k+1) * G(i, k+1);
            }
#endif
          } else {
            // CGS (original single-reduce GMRES)
            req->wait (); // wait for idot
            Kokkos::deep_copy (vals_h, vals);
            for (int i = 0; i <= k+1; i++) {
              G(i, k+1) = vals_h[i];
              H(i, k)   = vals_h[i];
//printf( " H(%d, %d) = %.2e\n",i,k,H(i,k) );
            }

            // Fix H
            for (int i = 0; i <= k; ++i) {
              H(k+1, k) -= (G(i, k+1)*G(i, k+1));
            }
          }

          // Integrate shift for Newton basis (applied through
          // change-of-basis)
          //if (computeRitzValues && k < ell) {
          if (computeRitzValues) {
            const complex_type theta = output.ritzValues[k%ell];
            UpdateNewton<SC, MV>::updateNewtonH(k, H, theta);
          }

          TEUCHOS_TEST_FOR_EXCEPTION
            (STS::real (H(k+1, k)) < STM::zero (), std::runtime_error,
             "At iteration " << iter << ", H(" << k+1 << ", "
             << k << ") = " << H(k+1, k) << " < 0.");
          H(k+1, k) = std::sqrt( H(k+1, k) );

          // Orthogonalize V(:, k+1), k+1 = iter-ell (using G, potentially removing shifts)
          vec_type AP = * (Q.getVectorNonConst (k+1));
          Teuchos::Range1D index_prev(0, k);
          const MV Qprev = * (Q.subView(index_prev));
          dense_matrix_type g_prev (Teuchos::View, G, k+1, 1, 0, k+1);

          MVT::MvTimesMatAddMv (-one, Qprev, g_prev, one, AP);
          MVT::MvScale (AP, one/H(k+1, k));
        }

        if (k >= 0 && iter < restart) {
          // Apply change-of-basis to W (using H, potentially with shifts)
          vec_type W = * (V.getVectorNonConst (iter+1));
          Teuchos::Range1D index_prev(ell, iter);
          const MV Zprev = * (V.subView(index_prev));

          dense_matrix_type h_prev (Teuchos::View, H, k+1, 1, 0, k);
          MVT::MvTimesMatAddMv (-one, Zprev, h_prev, one, W);

          MVT::MvScale (W, one/H(k+1, k));
        }

        int kk = k;
        if (this->input_.orthoType == "MGS" ||
            this->input_.orthoType == "CGS2") {
          kk --;
        }
        if (kk >= 0) {
          TEUCHOS_TEST_FOR_EXCEPTION
            (STS::real (H(kk+1, kk)) < STM::zero (), std::runtime_error,
             "At iteration " << kk << ", H(" << kk+1 << ", " << kk << ") = "
             << H(kk+1, kk) << " < 0.");
          // NOTE (mfh 16 Sep 2018) It's not entirely clear to me
          // whether the code as given to me was correct for complex
          // arithmetic.  I'll do my best to make it compile.
          if (STS::real (H(kk+1, kk)) > STS::real (tolOrtho*G(kk+1, kk+1))) {
            // Apply Givens rotations to new column of H and y
            this->reduceHessenburgToTriangular (kk, H, cs, sn, y.values());
            // Convergence check
            metric = this->getConvergenceMetric (STS::magnitude (y(kk+1)),
                                                 b_norm, input);
          }
          else { // breakdown
            H(kk+1, kk) = zero;
            metric = STM::zero ();
          }
          if (outPtr != nullptr) {
            // Update solution
            vec_type Y (B.getMap ());
            vec_type Z (B.getMap ());
            Tpetra::deep_copy (Y, X);
            if (kk > 0) {
              dense_vector_type  z (kk, true);
              blas.COPY (kk, y.values(), 1, z.values(), 1);
              blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                         Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                         kk, 1, one,
                         H.values(), H.stride(), z.values(), z.stride());
              Teuchos::Range1D cols(0, kk-1);
              Teuchos::RCP<const MV> Qj = Q.subView(cols);
              dense_vector_type z_iter (Teuchos::View, z.values (), kk);
              MVT::MvTimesMatAddMv (one, *Qj, z_iter, one, Y);
            }
            A.apply (Y, Z);
            Z.update (one, B, -one);
            SC z_norm = Z.norm2 (); // residual norm
            *outPtr << "Current iteration: iter=" << iter << "->" << kk
                    << ", restart=" << restart
                    << ", metric=" << metric
                    << ", real resnorm=" << z_norm << " " << z_norm/b_norm
                    << endl;
          }
        }

        if (iter < restart && metric > input.tol) {
          // Copy the new vector
          vec_type AP = * (Q.getVectorNonConst (iter+1));
          Tpetra::deep_copy (AP, * (V.getVectorNonConst (iter+1)));

          Teuchos::Range1D index_prev(0, iter+1);
          const MV Qprev  = * (Q.subView(index_prev));
          if (this->input_.orthoType == "MGS" ||
              this->input_.orthoType == "CGS2") {
            // Start all-reduce to compute G(:, iter+1)
            // [Q(:,1:k), V(:,k+1:iter+1)]'* [Q(:,k), V(:,iter+1)]
            #ifdef USE_IDOT_FOR_CGS2
            vec_type w1 = * (Q.getVectorNonConst (iter));
            req = Tpetra::idot (vals, Qprev, w1);

            vec_type w2 = * (Q.getVectorNonConst (iter+1));
            req2 = Tpetra::idot (vals2, Qprev, w2);
            #else
            dense_matrix_type c (Teuchos::View, C, iter+2, 2, 0, 0);
            Teuchos::Range1D index(iter, iter+1);
            const MV q  = * (Q.subView(index));
            MVT::MvTransMv(one, Qprev, q, c);
            #endif
          } else {
            // Start all-reduce to compute G(:, iter+1)
            // [Q(:,1:k), V(:,k+1:iter+1)]'*W
            vec_type W = * (V.getVectorNonConst (iter+1));
            req = Tpetra::idot (vals, Qprev, W);
          }
        }
      } // End of restart cycle
      if (this->input_.orthoType == "MGS" ||
          this->input_.orthoType == "CGS2") {
        int kk = iter - ell - 1;
        if (kk >= 0) {
          TEUCHOS_TEST_FOR_EXCEPTION
            (STS::real (H(kk+1, kk)) < STM::zero (), std::runtime_error,
             "At iteration " << kk << ", H(" << kk+1 << ", " << kk << ") = "
             << H(kk+1, kk) << " < 0.");
          // NOTE (mfh 16 Sep 2018) It's not entirely clear to me
          // whether the code as given to me was correct for complex
          // arithmetic.  I'll do my best to make it compile.
          if (STS::real (H(kk+1, kk)) > STS::real (tolOrtho*G(kk+1, kk+1))) {
            // Apply Givens rotations to new column of H and y
            this->reduceHessenburgToTriangular (kk, H, cs, sn, y.values());
            // Convergence check
            metric = this->getConvergenceMetric (STS::magnitude (y(kk+1)),
                                                 b_norm, input);
          }
          else { // breakdown
            H(kk+1, kk) = zero;
            metric = STM::zero ();
          }
        }
      }
      if (iter > 0) {
        // Update solution
        blas.TRSM (Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI,
                   Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG,
                   iter-ell, 1, one,
                   H.values(), H.stride(), y.values(), y.stride());
        Teuchos::Range1D cols(0, (iter-ell)-1);
        Teuchos::RCP<const MV> Qj = Q.subView(cols);
        y.resize (iter);
        if (input.precoSide == "right") {
          dense_vector_type y_iter (Teuchos::View, y.values (), iter-ell);

          //MVT::MvTimesMatAddMv (one, *Qj, y, zero, R);
          MVT::MvTimesMatAddMv (one, *Qj, y_iter, zero, R);
          M.apply (R, MZ);
          X.update (one, MZ, one);
        }
        else {
          dense_vector_type y_iter (Teuchos::View, y.values (), iter-ell);
          MVT::MvTimesMatAddMv (one, *Qj, y_iter, one, X);
        }
        y.resize (restart+1);
      }
      // Compute real residual
      Z = * (V.getVectorNonConst (0));
      A.apply (X, Z);
      Z.update (one, B, -one);
      r_norm = Z.norm2 (); // residual norm
      output.absResid = r_norm;
      output.relResid = r_norm / b_norm;
      // Convergence check (with explicitly computed residual norm)
      metric = this->getConvergenceMetric (r_norm, b_norm, input);
      if (metric <= input.tol) {
        output.converged = true;
      }
      else if (output.numIters < input.maxNumIters) {
        // Initialize starting vector for restart
        if (input.precoSide == "left") {
          Tpetra::deep_copy (R, Z);
          M.apply (R, Z);
        }
        // TODO: recomputing all-reduce, should be idot?
        r_norm = STS::real (Z.dot (Z)); //norm2 (); // residual norm
        G(0, 0) = r_norm;
        r_norm = STM::squareroot (r_norm);
        //Z.scale (one / r_norm);
        y[0] = r_norm;
        for (int i=1; i < restart+1; ++i) {
          y[i] = zero;
        }
        // Restart
        output.numRests ++;
      }
    }

    return output;
  }
};

template<class SC, class MV, class OP,
         template<class, class, class> class KrylovSubclassType>
class SolverManager;

// This is the Belos::SolverManager subclass that gets registered with
// Belos::SolverFactory.
template<class SC, class MV, class OP>
using GmresPipelineSolverManager = SolverManager<SC, MV, OP, GmresPipeline>;

/// \brief Register GmresPipelineSolverManager for all enabled Tpetra
///   template parameter combinations.
void register_GmresPipeline (const bool verbose);

} // namespace Impl
} // namespace BelosTpetra

#endif // BELOS_TPETRA_GMRES_PIPELINE_HPP
