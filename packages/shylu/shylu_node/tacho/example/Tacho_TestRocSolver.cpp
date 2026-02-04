#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <rocblas/rocblas.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
        exit(0);                               \
    }                                          \
}

#define ROC_CHECK(expression)                  \
{                                              \
    const rocblas_status status = expression;  \
    if(status != rocblas_status_success){      \
        std::cerr << "rocBLAS error "          \
                  << status << ": "            \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
        exit(0);                               \
    }                                          \
}


int main(int argc, char **argv) {
  int verbose = 0;
  int niters = 200; // number of outer iterations
  int nb = 10; // number of batches
  int N1 = 100, N2 = 1000;  // matrix dimension
  int nstreams = 16; // number of streams
  int option = 0;
  char* filename = nullptr;
  bool no_blas = false;     // drop off-diagonal block
  bool fence = false;       // fence between potrf and blas calls
  bool fence_all = false;   // fence after each kernel call
  bool fence_po = false;    // fence after potrf
  bool fence_blas = false;  // fence after blas call
  for (int i=0; i<argc; i++) {
    if (0 == strcmp(argv[i], "--fence"))      fence = true;
    if (0 == strcmp(argv[i], "--fence_blas")) fence_blas = true;
    if (0 == strcmp(argv[i], "--fence_po"))   fence_po = true;
    if (0 == strcmp(argv[i], "--fence_all"))  fence_all = true;
    if (0 == strcmp(argv[i], "--verbose"))    verbose = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--file"))       filename = argv[++i];
    if (0 == strcmp(argv[i], "--option"))     option = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--n1"))         N1 = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--n2"))         N2 = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--no_blas"))    no_blas = true;
    if (0 == strcmp(argv[i], "--nb"))         nb = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--niters"))     niters = atoi(argv[++i]);
    if (0 == strcmp(argv[i], "--nstreams"))   nstreams = atoi(argv[++i]);
  }

  // Create rocSOLVER/BLAS handle
  rocblas_handle *handles = (rocblas_handle*)malloc(nstreams * sizeof(rocblas_handle));
  hipStream_t *streams = (hipStream_t*)malloc(nstreams * sizeof(hipStream_t));
  for (int k=0; k<nstreams; k++) {
    ROC_CHECK(rocblas_create_handle(&handles[k]));
    if (option == 1) {
      HIP_CHECK(hipStreamCreateWithFlags(&streams[k], hipStreamDefault));
      ROC_CHECK(rocblas_set_stream(handles[k], streams[k]));
    } else if (option == 2) {
      HIP_CHECK(hipStreamCreateWithFlags(&streams[k], hipStreamNonBlocking));
      ROC_CHECK(rocblas_set_stream(handles[k], streams[k]));
    }
  }

  // Solving "nb" 2-by-2 SPD linear systems [A E; E' D]
  FILE *fp = nullptr;
  if (filename) {
    printf( "\n READING block sizes from %s\n",filename );
    fp = fopen(filename,"r");
    if (!fp) {
      printf( "\n ** FAILED to open %s **\n\n",filename );
      exit(0);
    }
    nb = 0;
    char c;
    while (EOF != (c=getc(fp))) {
      if ('\n' == c) ++nb;
    }
    printf( " > There are nb=%d lines <\n",nb );
    rewind(fp);
  }
  // > Setup matrices/vectors
  printf( "\n n=%d+%d, nd=%d, nstreams=%d, option=%d%s%s%s%s\n",N1,N2,nb,nstreams,option,
              (fence ? ", with fence" : ""),(fence_po ? ", with fence potrf" : ""),
              (fence_blas ? ", with fence blas" : ""),(fence_all ? ", with fence all" :""));
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
  rocblas_int *dInfo1, *dInfo2;
  HIP_CHECK(hipMalloc((void**)&dInfo1, nb*sizeof(rocblas_int)));
  HIP_CHECK(hipMalloc((void**)&dInfo2, nb*sizeof(rocblas_int)));
  // block sizes
  int *n1 = (int*)malloc(nb * sizeof(int));
  int *n2 = (int*)malloc(nb * sizeof(int));

  // Set random seed for determistic matrix blocks
  std::srand(1);
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
    if (n2[k] > 0) {
      hE[k] = (double*)malloc((n1[k]*n2[k]) * sizeof(double));
      hD[k] = (double*)malloc((n2[k]*n2[k]) * sizeof(double));
    }

    // A, first diagonal blocks, and E, off-diagonal block
    for (rocblas_int i=0; i<n1[k]; i++) {
      // A, first diagonal block,  n1-by-n1
      for (rocblas_int j=i; j<n1[k]; j++) {
        hA[k][i+j*n1[k]] = double(std::rand()) / double(RAND_MAX);
      }
      hA[k][i+i*n1[k]] += (n1[k]+n2[k]);
      // E, off-diagonal block, n1-by-n2
      for (rocblas_int j=0; j<n2[k]; j++) {
        hE[k][i+j*n1[k]] = (no_blas ? 0.0 : double(std::rand()) / double(RAND_MAX));
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
    if (verbose == 1) {
      char filename[250];
      FILE *fp;

      sprintf(filename,"A_%d.dat", k);
      fp = fopen(filename,"w");
      for (int i = 0; i < n1[k]; ++i) {
        for (int j = 0; j < n1[k]; ++j) fprintf(fp,"%.16e ",hA[k][i+j*n1[k]]);
        fprintf(fp,"\n");
      }
      fclose(fp);

      sprintf(filename,"E_%d.dat", k);
      fp = fopen(filename,"w");
      for (int i = 0; i < n1[k]; ++i) {
        for (int j = 0; j < n2[k]; ++j) fprintf(fp,"%.16e ",hE[k][i+j*n1[k]]);
        fprintf(fp,"\n");
      }
      fclose(fp);

      sprintf(filename,"D_%d.dat", k);
      fp = fopen(filename,"w");
      for (int i = 0; i < n2[k]; ++i) {
        for (int j = 0; j < n2[k]; ++j) fprintf(fp,"%.16e ",hD[k][i+j*n2[k]]);
        fprintf(fp,"\n");
      }
      fclose(fp);

      sprintf(filename,"b_%d.dat", k);
      fp = fopen(filename,"w");
      for (int i = 0; i < n1[k]+n2[k]; ++i) fprintf(fp,"%d %.16e\n",i,hB[k][i]);
      fclose(fp);
    }
    // Allocate matrix & vector on device
    // > matrix
    HIP_CHECK(hipMalloc((void**)&(dA[k]), sizeof(double) * n1[k]*n1[k]));
    if (n2[k] > 0) {
      HIP_CHECK(hipMalloc((void**)&(dE[k]), sizeof(double) * n1[k]*n2[k]));
      HIP_CHECK(hipMalloc((void**)&(dD[k]), sizeof(double) * n2[k]*n2[k]));
    }
    // > vector
    int n = n1[k] + n2[k];
    HIP_CHECK(hipMalloc((void**)&(dB[k]), sizeof(double) * n));
  } // End of generating matrix & vectors
  if (fp) fclose(fp);


  //
  // !!!!! Outer Iterations !!!!!
  //
  for (int iter = 0; iter < niters; iter++) {
    printf( "\n ===== iteration %d =====\n",iter );

    // Copy matrix from host to device
    for (int k=0; k<nb; k++) {
      HIP_CHECK(hipMemcpy(dA[k], hA[k], sizeof(double) * n1[k]*n1[k], hipMemcpyHostToDevice));
      if (n2[k] > 0) {
        HIP_CHECK(hipMemcpy(dE[k], hE[k], sizeof(double) * n1[k]*n2[k], hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dD[k], hD[k], sizeof(double) * n2[k]*n2[k], hipMemcpyHostToDevice));
      }
    }
    HIP_CHECK(hipDeviceSynchronize());

    // !!! factor the 2-by-2 blocks !!!
    double  one( 1.0);
    double mone(-1.0);
    for (int k=0; k<nb; k++) {
      // factor the first diagonal blocks, R := chol(A)
      int qid = k%nstreams;
      ROC_CHECK(rocsolver_dpotrf(handles[qid], rocblas_fill_upper, n1[k], dA[k], n1[k], &dInfo1[k]));
      if (fence_all || fence_po) HIP_CHECK(hipDeviceSynchronize());
    }
    if (!no_blas) {
      if (fence || fence_blas) HIP_CHECK(hipDeviceSynchronize());
      for (int k=0; k<nb; k++) {
        // compute the off-diagonal factor, E := R^{-1}*E
        int qid = k%nstreams;
        ROC_CHECK(rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                                rocblas_operation_transpose, rocblas_diagonal_non_unit,
                                n1[k], n2[k], &one, dA[k], n1[k], dE[k], n1[k]));
        if (fence_all || fence_blas) HIP_CHECK(hipDeviceSynchronize());

        // update the second diagonal, D -= E'*E
        ROC_CHECK(rocblas_dsyrk(handles[qid], rocblas_fill_upper, rocblas_operation_transpose,
                                n2[k], n1[k], &mone, dE[k], n1[k], &one, dD[k], n2[k]));
        if (fence_all || fence_blas) HIP_CHECK(hipDeviceSynchronize());
      }
    }
    if (fence || fence_po) HIP_CHECK(hipDeviceSynchronize());
    for (int k=0; k<nb; k++) {
      // factor the second diagonal block, chol(D)
      int qid = k%nstreams;
      ROC_CHECK(rocsolver_dpotrf(handles[qid], rocblas_fill_upper, n2[k], dD[k], n2[k], &dInfo2[k]));
      if (fence_all || fence_po) HIP_CHECK(hipDeviceSynchronize());
    }

    // !!! check (on stream-0) !!!
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipPeekAtLastError());
    // Copy vector from host to device
    for (int k=0; k<nb; k++) {
      int n = n1[k] + n2[k];
      HIP_CHECK(hipMemcpy(dB[k], hB[k], sizeof(double) * n, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Solve [A,E; E' D] [x1;x2] = [b1;b2]
    for (int k=0; k<nb; k++) {
      int qid = 0; //k%nstreams;
      int n = n1[k]+n2[k];
      // --- step 1 --
      // b(1) := L(A)\b(1)
      ROC_CHECK(rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                              rocblas_operation_transpose, rocblas_diagonal_non_unit,
                              n1[k], nrhs, &one, dA[k], n1[k], &dB[k][0], n));
      if (!no_blas) {
        // --- step 2 --
        // b(2) := b(2) - E'*b(1)
        ROC_CHECK(rocblas_dgemm(handles[qid], rocblas_operation_transpose, rocblas_operation_none,
                                n2[k], nrhs, n1[k], &mone, dE[k], n1[k], &dB[k][0], n, &one, &dB[k][n1[k]], n));
      }

      // --- step 3 --
      // b(2) = L(D)\b(2)
      ROC_CHECK(rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                              rocblas_operation_transpose, rocblas_diagonal_non_unit,
                              n2[k], nrhs, &one, dD[k], n2[k], &dB[k][n1[k]], n));
      // x(2) = U(D)\x(2)
      ROC_CHECK(rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                              rocblas_operation_none, rocblas_diagonal_non_unit,
                              n2[k], nrhs, &one, dD[k], n2[k], &dB[k][n1[k]], n));

      if (!no_blas) {
        // --- step 4 --
        // b(1) = b(1) - E*x(2)
        ROC_CHECK(rocblas_dgemm(handles[qid], rocblas_operation_none, rocblas_operation_none,
                                n1[k], nrhs, n2[k], &mone, dE[k], n1[k], &dB[k][n1[k]], n, &one, &dB[k][0], n));
      }
      // --- step 5 --
      // x(1) = U(A)\b(1)
      ROC_CHECK(rocblas_dtrsm(handles[qid], rocblas_side_left, rocblas_fill_upper,
                              rocblas_operation_none, rocblas_diagonal_non_unit,
                              n1[k], nrhs, &one, dA[k], n1[k], &dB[k][0], n));
    }

    // Compute the residual norm
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipPeekAtLastError());
    // Copy vector from host to device
    for (int k=0; k<nb; k++) {
      // Copy solution and error code to host
      rocblas_int hInfo1, hInfo2;
      HIP_CHECK(hipMemcpy(hX[k], dB[k], sizeof(double) * (n1[k]+n2[k]), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&hInfo1, &dInfo1[k], sizeof(rocblas_int), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&hInfo2, &dInfo2[k], sizeof(rocblas_int), hipMemcpyDeviceToHost));
      if (verbose == 2) {
        printf("x%d=[\n",k);
        for (int i = 0; i < n1[k]+n2[k]; ++i) printf("%d %.16e %.16e\n",i,hX[k][i],hB[k][i]);
        printf("];\n");
      }

      if (hInfo1 == 0 && hInfo2 == 0) {
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
        printf("rocsolver_dpotrf FAILed with info = %d,%d\n", hInfo1,hInfo2);
      }
      printf("\n");
    }
  }
  // Free memory
  for (int k=0; k<nb; k++) {
    // matrix
    free(hA[k]);
    HIP_CHECK(hipFree(dA[k]));
    if (n2[k] > 0) {
      free(hE[k]);
      free(hD[k]);
      HIP_CHECK(hipFree(dE[k]));
      HIP_CHECK(hipFree(dD[k]));
    }
    // vector
    free(hX[k]);
    free(hB[k]);
    HIP_CHECK(hipFree(dB[k]));
  }
  free(n1);
  free(n2);
  // matrices
  free(hA);
  free(hE);
  free(hD);
  free(dA);
  free(dE);
  free(dD);
  // vectors
  free(hX);
  free(hB);
  free(dB);
  HIP_CHECK(hipFree(dInfo1));
  HIP_CHECK(hipFree(dInfo2));
  for (int k=0; k<nstreams; k++) {
    ROC_CHECK(rocblas_destroy_handle(handles[k]));
    if (option != 0) {
      HIP_CHECK(hipStreamDestroy(streams[k]));
    }
  }
  free(handles);
  free(streams);

  return 0;
}
