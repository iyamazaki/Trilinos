#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <rocblas/rocblas.h>
#include <stdio.h>               // for printf
#include <stdlib.h>              // for malloc

int main(int argc, char **argv) {
  int verbose = 0;
  int n  = 10; // matrix dimension
  int nb = 10; // number of batches
  int nstreams = 1; // number of streams
  int option = 0;
  bool fence = false;
  for (int i=0; i<argc; i++) {
    if (0 == strcmp(argv[i], "--verbose")) verbose = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--option"))  option = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--fence"))   fence = true;
    if (0 == strcmp(argv[i], "--dim"))     n = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--nb"))      nb = atoi(argv[++i]);
  }
  printf( "\n n=%d, nd=%d, nstreams=%d, option=%d\n",n,nb,nstreams,option );

  const rocblas_int N = n;
  const rocblas_int nrhs = 1;
  const rocblas_int lda = N;
  const rocblas_int lde = N;
  const rocblas_int ldd = N;
  const rocblas_int ldb = N;

  // Solving "nb" 2-by-2 SPD linear systems [A E; E' D]
  // > on host
  double **hA = (double**)malloc(nb * sizeof(double*));
  double **hE = (double**)malloc(nb * sizeof(double*));
  double **hD = (double**)malloc(nb * sizeof(double*));
  // > on device
  double **dA = (double**)malloc(nb * sizeof(double*));
  double **dE = (double**)malloc(nb * sizeof(double*));
  double **dD = (double**)malloc(nb * sizeof(double*));
  // vectors
  double **hB = (double**)malloc(nb * sizeof(double*));
  double **hX = (double**)malloc(nb * sizeof(double*));
  double **dB = (double**)malloc(2*nb * sizeof(double*));
  for (int k=0; k<nb; k++) {
    // Generate SPD matrices
    hA[k] = (double*)malloc((N*N) * sizeof(double));
    hE[k] = (double*)malloc((N*N) * sizeof(double));
    hD[k] = (double*)malloc((N*N) * sizeof(double));
    for (rocblas_int i=0; i<N; i++) {
      // A & D, diagonal blocks
      for (rocblas_int j=i; j<N; j++) {
        hA[k][i+j*N] = double(std::rand()) / double(RAND_MAX);
        hD[k][i+j*N] = double(std::rand()) / double(RAND_MAX);
      }
      hA[k][i+i*N] += 2*N;
      hD[k][i+i*N] += 2*N;
      // E, off-diagonal block
      for (rocblas_int j=0; j<N; j++) {
        hE[k][i+j*N] = double(std::rand()) / double(RAND_MAX);
      }
    }
    // explicitly symmetrize the diagonal block
    // (to simply "check" the residual norms)
    for (rocblas_int i=0; i<N; i++) {
      for (rocblas_int j=0; j<i; j++) {
        hA[k][i+j*N] = hA[k][j+i*N];
        hD[k][i+j*N] = hD[k][j+i*N];
      }
    }

    // Generate RHS, B = A*ones(n,1)
    hB[k] = (double*)malloc(2*N * sizeof(double));
    hX[k] = (double*)malloc(2*N * sizeof(double));
    for (rocblas_int i=0; i<N; i++) {
      hB[k][i] = 0.0;
      for (rocblas_int j=0; j<N; j++) {
        hB[k][i] += hA[k][i+j*N];
      }
      for (rocblas_int j=0; j<N; j++) {
        hB[k][i] += hE[k][i+j*N];
      }
    }
    for (rocblas_int i=0; i<N; i++) {
      hB[k][N+i] = 0.0;
      for (rocblas_int j=0; j<N; j++) {
        hB[k][N+i] += hD[k][i+j*N];
      }
      for (rocblas_int j=0; j<N; j++) {
        hB[k][N+i] += hE[k][j+i*N];
      }
    }
    // Copy matrix & vector from host to device
    // > matrix
    hipMalloc((void**)&(dA[k]), sizeof(double) * N*N);
    hipMalloc((void**)&(dE[k]), sizeof(double) * N*N);
    hipMalloc((void**)&(dD[k]), sizeof(double) * N*N);
    hipMemcpy(dA[k], hA[k], sizeof(double) * N*N, hipMemcpyHostToDevice);
    hipMemcpy(dE[k], hE[k], sizeof(double) * N*N, hipMemcpyHostToDevice);
    hipMemcpy(dD[k], hD[k], sizeof(double) * N*N, hipMemcpyHostToDevice);
    // > vector
    hipMalloc((void**)&(dB[k]), sizeof(double) * 2*N);
    hipMemcpy(dB[k], hB[k], sizeof(double) * 2*N, hipMemcpyHostToDevice);
  } // End of generating matrix & vectors

  // Create rocSOLVER/BLAS handle
  rocblas_handle *handles = (rocblas_handle*)malloc(nstreams * sizeof(rocblas_handle));
  hipStream_t *streams = (hipStream_t*)malloc(nstreams * sizeof(hipStream_t));
  for (int k=0; k<nstreams; k++) {
    rocblas_create_handle(&handles[k]);
    if (option == 1) {
      hipStreamCreateWithFlags(&streams[k], hipStreamDefault);
      rocblas_set_stream(handles[k], streams[k]);
    } else if (option == 2) {
      hipStreamCreateWithFlags(&streams[k], hipStreamNonBlocking);
      rocblas_set_stream(handles[k], streams[k]);
    }
  }

  // Allocate integer device integer-array to return error code
  rocblas_int *dInfo;
  hipMalloc((void**)&dInfo, nb*sizeof(rocblas_int));

  // !!! factor the 2-by-2 blocks !!!
  double  one( 1.0);
  double mone(-1.0);
  for (int k=0; k<nb; k++) {
    // factor the first diagonal blocks, R := chol(A)
    int qid = k%nstreams;
    rocsolver_dpotrf(handles[qid], rocblas_fill_upper, N, dA[k], lda, &dInfo[k]);
  }
  if (fence) hipDeviceSynchronize();
  for (int k=0; k<nb; k++) {
    // compute the off-diagonal factor, E := R^{-1}*E
    int qid = k%nstreams;
    rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                  rocblas_operation_transpose, rocblas_diagonal_non_unit,
                  N, N, &one, dA[k], lda, dE[k], lde);
    // update the second diagonal, D -= E'*E
    rocblas_dsyrk(handles[qid], rocblas_fill_upper, rocblas_operation_transpose,
                  N, N, &mone, dE[k], lde, &one, dD[k], ldd);
  }
  if (fence) hipDeviceSynchronize();
  for (int k=0; k<nb; k++) {
    // factor the second diagonal block, chol(D)
    int qid = k%nstreams;
    rocsolver_dpotrf(handles[qid], rocblas_fill_upper, N, dD[k], ldd, &dInfo[k]);
  }

  // !!! check (on stream-0) !!!
  hipDeviceSynchronize();

  // Solve [A,E; E' D] [x1;x2] = [b1;b2]
  for (int k=0; k<nb; k++) {
    int qid = 0; //k%nstreams;
    // --- step 1 --
    // b(1) := L(A)\b(1)
    rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                  rocblas_operation_transpose, rocblas_diagonal_non_unit,
                  N, nrhs, &one, dA[k], lda, &dB[k][0], ldb);

    // --- step 2 --
    // b(2) := b(2) - E'*b(1)
    rocblas_dgemm(handles[qid], rocblas_operation_transpose, rocblas_operation_none,
		  N, nrhs, N, &mone, dE[k], lde, &dB[k][0], ldb, &one, &dB[k][N], ldb);

    // --- step 3 --
    // b(2) = L(D)\b(2)
    rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                  rocblas_operation_transpose, rocblas_diagonal_non_unit,
                  N, nrhs, &one, dD[k], ldd, &dB[k][N], ldb);
    // x(2) = U(D)\x(2)
    rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                  rocblas_operation_none, rocblas_diagonal_non_unit,
                  N, nrhs, &one, dD[k], ldd, &dB[k][N], ldb);

    // --- step 4 --
    // b(1) = b(1) - E*x(2)
    rocblas_dgemm(handles[qid], rocblas_operation_none, rocblas_operation_none,
		  N, nrhs, N, &mone, dE[k], lde, &dB[k][N], ldb, &one, &dB[k][0], ldb);

    // --- step 5 --
    // x(1) = U(A)\b(1)
    rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                  rocblas_operation_none, rocblas_diagonal_non_unit,
                  N, nrhs, &one, dA[k], lda, &dB[k][0], ldb);
  }
  // Compute the residual norm
  for (int k=0; k<nb; k++) {
    // Copy solution and error code to host
    rocblas_int hInfo;
    hipMemcpy(hX[k], dB[k], sizeof(double) * 2*N, hipMemcpyDeviceToHost);
    hipMemcpy(&hInfo, &dInfo[k], sizeof(rocblas_int), hipMemcpyDeviceToHost);
    if (verbose == 1) for (int i = 0; i < 2*N; ++i) printf("%e\n",hX[k][i]);

    if (hInfo == 0) {
      double rnorm = 0.0;
      double bnorm = 0.0;
      printf("%d: Checking residual norms:\n",k);
      // residual norm for the first block row:
      for (int i = 0; i < N; ++i) {
        bnorm += hB[k][i]*hB[k][i];
        for (int j = 0; j < N; ++j) {
          hB[k][i] -= hA[k][i+j*N] * hX[k][j];
        }
        for (int j = 0; j < N; ++j) {
          hB[k][i] -= hE[k][i+j*N] * hX[k][N+j];
        }
        rnorm += hB[k][i]*hB[k][i];
      }
      double rnorm1 = std::sqrt(rnorm);
      double bnorm1 = std::sqrt(bnorm);
      printf( " 1: rnorm = %e / %e = %e\n",rnorm1,bnorm1,rnorm1/bnorm1 );
      // residual norm for the second block row:
      for (int i = 0; i < N; ++i) {
        bnorm += hB[k][N+i]*hB[k][N+i];
        for (int j = 0; j < N; ++j) {
          hB[k][N+i] -= hE[k][j+i*N] * hX[k][j];
        }
        for (int j = 0; j < N; ++j) {
          hB[k][N+i] -= hD[k][i+j*N] * hX[k][N+j];
        }
        rnorm += hB[k][N+i]*hB[k][N+i];
      }

      rnorm = std::sqrt(rnorm);
      bnorm = std::sqrt(bnorm);
      printf( " 2: rnorm = %e / %e = %e\n",rnorm,bnorm,rnorm/bnorm );
      double tol = 0.000000001;
      if (rnorm > tol) printf("%d: FAIL\n",k);
      else printf("%d: PASS\n",k);
    } else {
      printf("Cholesky factorization failed. Info code: %d\n", hInfo);
    }
    printf("\n");
  }
  // Free memory
  for (int k=0; k<nb; k++) {
    free(hA[k]);
    free(hB[k]);
    free(hX[k]);

    hipFree(dA[k]);
    hipFree(dB[k]);
  }
  free(hA);
  free(hB);
  free(hX);
  free(dA);
  free(dB);
  hipFree(dInfo);
  for (int k=0; k<nstreams; k++) {
    rocblas_destroy_handle(handles[k]);
    hipStreamDestroy(streams[k]);
  }
  free(handles);
  free(streams);

  return 0;
}
