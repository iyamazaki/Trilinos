#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <rocblas/rocblas.h>
#include <stdio.h>               // for printf
#include <stdlib.h>              // for malloc

int main(int argc, char **argv) {
  int verbose = 0;
  int niters = 200; // number of outer iterations
  int nb = 10; // number of batches
  int N1 = 100, N2 = 1000;  // matrix dimension
  int nstreams = 16; // number of streams
  int option = 0;
  char* filename = nullptr;
  bool fence = false;
  for (int i=0; i<argc; i++) {
    if (0 == strcmp(argv[i], "--fence"))    fence = true;
    if (0 == strcmp(argv[i], "--verbose"))  verbose = 1;
    if (0 == strcmp(argv[i], "--file"))     filename = argv[++i];
    if (0 == strcmp(argv[i], "--option"))   option = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--n1"))       N1 = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--n2"))       N2 = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--nb"))       nb = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--niters"))   niters = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--nstreams")) nstreams = atoi(argv[++i]);
  }
  printf( "\n n=%d+%d, nd=%d, nstreams=%d, option=%d %s\n",N1,N2,nb,nstreams,option,filename );

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
  const rocblas_int nrhs = 1;
  double **hB = (double**)malloc(nb * sizeof(double*));
  double **hX = (double**)malloc(nb * sizeof(double*));
  double **dB = (double**)malloc(nb * sizeof(double*));
  // info
  rocblas_int *dInfo;
  hipMalloc((void**)&dInfo, nb*sizeof(rocblas_int));
  // block sizes
  int *n1 = (int*)malloc(nb * sizeof(int));
  int *n2 = (int*)malloc(nb * sizeof(int));

  // Setup matrices/vectors
  FILE *fp = nullptr;
  if (filename) {
    printf( " READING block sizes from %s\n",filename );
    fp = fopen(filename,"r");
  }
  for (int k=0; k<nb; k++) {
    // Matrix dimension
    if (fp) {
      fscanf(fp,"%d %d\n",&n1[k],&n2[k]);
    } else {
      n1[k] = N1;
      n2[k] = N2;
    }
    // Generate SPD matrices [A E; E' D]
    hA[k] = (double*)malloc((n1[k]*n1[k]) * sizeof(double));
    hE[k] = (double*)malloc((n1[k]*n2[k]) * sizeof(double));
    hD[k] = (double*)malloc((n2[k]*n2[k]) * sizeof(double));

    // A, first diagonal blocks, and E, off-diagonal block
    for (rocblas_int i=0; i<n1[k]; i++) {
      // A, first diagonal block,  n1-by-n1
      for (rocblas_int j=i; j<n1[k]; j++) {
        hA[k][i+j*n1[k]] = double(std::rand()) / double(RAND_MAX);
      }
      hA[k][i+i*n1[k]] += (n1[k]+n2[k]);
      // E, off-diagonal block, n1-by-n2
      for (rocblas_int j=0; j<n2[k]; j++) {
        hE[k][i+j*n1[k]] = double(std::rand()) / double(RAND_MAX);
      }
    }
    // D, second diagonal block, and off-diagonal block
    for (rocblas_int i=0; i<n2[k]; i++) {
      // D, second diagonal blocks, n2-by-n2
      for (rocblas_int j=i; j<n2[k]; j++) {
        hD[k][i+j*n2[k]] = double(std::rand()) / double(RAND_MAX);
      }
      hD[k][i+i*n2[k]] += (n1[k]+n2[k]);
    }
    // explicitly symmetrize the diagonal block
    // (to simply "check" the residual norms)
    for (rocblas_int i=0; i<n1[k]; i++) {
      for (rocblas_int j=0; j<i; j++) {
        hA[k][i+j*n1[k]] = hA[k][j+i*n1[k]];
      }
    }
    for (rocblas_int i=0; i<n2[k]; i++) {
      for (rocblas_int j=0; j<i; j++) {
        hD[k][i+j*n2[k]] = hD[k][j+i*n2[k]];
      }
    }

    // Generate RHS, B = A*ones(n,1)
    hB[k] = (double*)malloc((n1[k]+n2[k]) * sizeof(double));
    hX[k] = (double*)malloc((n1[k]+n2[k]) * sizeof(double));
    // B1 = [A, E]*ones(n,1)
    for (rocblas_int i=0; i<n1[k]; i++) {
      hB[k][i] = 0.0;
      // B1 += A*ones(n1)
      for (rocblas_int j=0; j<n1[k]; j++) {
        hB[k][i] += hA[k][i+j*n1[k]];
      }
      // B1 += E*ones(n2)
      for (rocblas_int j=0; j<n2[k]; j++) {
        hB[k][i] += hE[k][i+j*n1[k]];
      }
    }
    // B2 = [E', D]*ones(n,1)
    for (rocblas_int i=0; i<n2[k]; i++) {
      hB[k][n1[k]+i] = 0.0;
      // B2 += E'*ones(n2)
      for (rocblas_int j=0; j<n1[k]; j++) {
        hB[k][n1[k]+i] += hE[k][j+i*n1[k]];
      }
      // B2 += D*ones(n2)
      for (rocblas_int j=0; j<n2[k]; j++) {
        hB[k][n1[k]+i] += hD[k][i+j*n2[k]];
      }
    }
    // Allocate matrix & vector on device
    // > matrix
    hipMalloc((void**)&(dA[k]), sizeof(double) * n1[k]*n1[k]);
    hipMalloc((void**)&(dE[k]), sizeof(double) * n1[k]*n2[k]);
    hipMalloc((void**)&(dD[k]), sizeof(double) * n2[k]*n2[k]);
    // > vector
    int n = n1[k] + n2[k];
    hipMalloc((void**)&(dB[k]), sizeof(double) * n);
  } // End of generating matrix & vectors
  if (fp) fclose(fp);


  //
  // !!!!! Outer Iterations !!!!!
  //
  for (int iter = 0; iter < niters; iter++) {
    printf( "\n ===== iteration %d =====\n",iter );

    // Copy matrix from host to device
    for (int k=0; k<nb; k++) {
      hipMemcpy(dA[k], hA[k], sizeof(double) * n1[k]*n1[k], hipMemcpyHostToDevice);
      hipMemcpy(dE[k], hE[k], sizeof(double) * n1[k]*n2[k], hipMemcpyHostToDevice);
      hipMemcpy(dD[k], hD[k], sizeof(double) * n2[k]*n2[k], hipMemcpyHostToDevice);
    }

    // !!! factor the 2-by-2 blocks !!!
    double  one( 1.0);
    double mone(-1.0);
    for (int k=0; k<nb; k++) {
      // factor the first diagonal blocks, R := chol(A)
      int qid = k%nstreams;
      rocsolver_dpotrf(handles[qid], rocblas_fill_upper, n1[k], dA[k], n1[k], &dInfo[k]);
    }
    if (fence) hipDeviceSynchronize();
    for (int k=0; k<nb; k++) {
      // compute the off-diagonal factor, E := R^{-1}*E
      int qid = k%nstreams;
      rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                    rocblas_operation_transpose, rocblas_diagonal_non_unit,
                    n1[k], n2[k], &one, dA[k], n1[k], dE[k], n1[k]);
      // update the second diagonal, D -= E'*E
      rocblas_dsyrk(handles[qid], rocblas_fill_upper, rocblas_operation_transpose,
                    n2[k], n1[k], &mone, dE[k], n1[k], &one, dD[k], n2[k]);
    }
    if (fence) hipDeviceSynchronize();
    for (int k=0; k<nb; k++) {
      // factor the second diagonal block, chol(D)
      int qid = k%nstreams;
      rocsolver_dpotrf(handles[qid], rocblas_fill_upper, n2[k], dD[k], n2[k], &dInfo[k]);
    }

    // !!! check (on stream-0) !!!
    hipDeviceSynchronize();
    // Copy vector from host to device
    for (int k=0; k<nb; k++) {
      int n = n1[k] + n2[k];
      hipMemcpy(dB[k], hB[k], sizeof(double) * n, hipMemcpyHostToDevice);
    }

    // Solve [A,E; E' D] [x1;x2] = [b1;b2]
    for (int k=0; k<nb; k++) {
      int qid = 0; //k%nstreams;
      int n = n1[k]+n2[k];
      // --- step 1 --
      // b(1) := L(A)\b(1)
      rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                    rocblas_operation_transpose, rocblas_diagonal_non_unit,
                    n1[k], nrhs, &one, dA[k], n1[k], &dB[k][0], n);

      // --- step 2 --
      // b(2) := b(2) - E'*b(1)
      rocblas_dgemm(handles[qid], rocblas_operation_transpose, rocblas_operation_none,
                    n2[k], nrhs, n1[k], &mone, dE[k], n1[k], &dB[k][0], n, &one, &dB[k][n1[k]], n);

      // --- step 3 --
      // b(2) = L(D)\b(2)
      rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                    rocblas_operation_transpose, rocblas_diagonal_non_unit,
                    n2[k], nrhs, &one, dD[k], n2[k], &dB[k][n1[k]], n);
      // x(2) = U(D)\x(2)
      rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                    rocblas_operation_none, rocblas_diagonal_non_unit,
                    n2[k], nrhs, &one, dD[k], n2[k], &dB[k][n1[k]], n);

      // --- step 4 --
      // b(1) = b(1) - E*x(2)
      rocblas_dgemm(handles[qid], rocblas_operation_none, rocblas_operation_none,
                    n1[k], nrhs, n2[k], &mone, dE[k], n1[k], &dB[k][n1[k]], n, &one, &dB[k][0], n);

      // --- step 5 --
      // x(1) = U(A)\b(1)
      rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                    rocblas_operation_none, rocblas_diagonal_non_unit,
                    n1[k], nrhs, &one, dA[k], n1[k], &dB[k][0], n);
    }

    // Compute the residual norm
    hipDeviceSynchronize();
    for (int k=0; k<nb; k++) {
      // Copy solution and error code to host
      rocblas_int hInfo;
      hipMemcpy(hX[k], dB[k], sizeof(double) * (n1[k]+n2[k]), hipMemcpyDeviceToHost);
      hipMemcpy(&hInfo, &dInfo[k], sizeof(rocblas_int), hipMemcpyDeviceToHost);
      if (verbose == 1) {
        printf("A=[\n");
        for (int i = 0; i < n1[k]; ++i) {
          for (int j = 0; j < n1[k]; ++j) printf("%.16e ",hA[k][i+j*n1[k]]);
          printf("\n");
        }
        printf("];\n");
        printf("E=[\n");
        for (int i = 0; i < n1[k]; ++i) {
          for (int j = 0; j < n2[k]; ++j) printf("%.16e ",hE[k][i+j*n1[k]]);
          printf("\n");
        }
        printf("];\n");
        printf("D=[\n");
        for (int i = 0; i < n2[k]; ++i) {
          for (int j = 0; j < n2[k]; ++j) printf("%.16e ",hD[k][i+j*n2[k]]);
          printf("\n");
        }
        printf("];\n");
        printf("ixb=[\n");
        for (int i = 0; i < n1[k]+n2[k]; ++i) printf("%d %.16e %.16e\n",i,hX[k][i],hB[k][i]);
        printf("];\n");
      }

      if (hInfo == 0) {
        double rnorm = 0.0;
        double bnorm = 0.0;
        printf("%d: Checking residual norms (n=%d+%d):\n",k,n1[k],n2[k]);
        // residual norm for the first block row:
        for (int i = 0; i < n1[k]; ++i) {
          double Bi = hB[k][i];
          bnorm += Bi*Bi;
          for (int j = 0; j < n1[k]; ++j) {
            Bi -= hA[k][i+j*n1[k]] * hX[k][j];
          }
          for (int j = 0; j < n2[k]; ++j) {
            Bi -= hE[k][i+j*n1[k]] * hX[k][n1[k]+j];
          }
          rnorm += Bi*Bi;
        }
        double rnorm1 = std::sqrt(rnorm);
        double bnorm1 = std::sqrt(bnorm);
        printf( " 1: rnorm = %e / %e = %e\n",rnorm1,bnorm1,rnorm1/bnorm1 );
        // residual norm for the second block row:
        for (int i = 0; i < n2[k]; ++i) {
          double Bi = hB[k][n1[k]+i];
          bnorm += Bi*Bi;
          for (int j = 0; j < n1[k]; ++j) {
            Bi -= hE[k][j+i*n1[k]] * hX[k][j];
          }
          for (int j = 0; j < n2[k]; ++j) {
            Bi -= hD[k][i+j*n2[k]] * hX[k][n1[k]+j];
          }
          rnorm += Bi*Bi;
        }
        rnorm = std::sqrt(rnorm);
        bnorm = std::sqrt(bnorm);
        printf( " 2: rnorm = %e / %e = %e\n",rnorm,bnorm,rnorm/bnorm );
        double tol = 0.000000001;
        if (rnorm > tol*bnorm) printf("%d: FAIL (%dx%d %e / %e = %e)\n",k,n1[k],n2[k], rnorm,bnorm,rnorm/bnorm);
        else printf("%d: PASS\n",k);
      } else {
        printf("Cholesky factorization failed. Info code: %d\n", hInfo);
      }
      printf("\n");
    }
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
