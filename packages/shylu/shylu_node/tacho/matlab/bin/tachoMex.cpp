// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "tachoMex.h"

#define IS_FALSE 0
#define IS_TRUE 1
#define MUEMEX_ERROR -1

extern void _main();

/* MUEMEX Teuchos Parameters*/
#define MUEMEX_INTERFACE "Linear Algebra"


namespace Tacho {

template <>
TachoSystem<double>::TachoSystem() {}
template <>
TachoSystem<double>::~TachoSystem() {}



/// SetParameters function
template <typename Scalar>
int TachoSystem<Scalar>::option(const mxArray** mx) {
  std::string option_name = loadDataFromMatlab<std::string>(mx[0]);
  if (option_name == "verbose") {
    bool verbose = true;
    solver.setVerbose(verbose);
  } else if (option_name == "method") {
    int method = 1;
    std::string method_name = loadDataFromMatlab<std::string>(mx[1]);
    if (method_name == "chol")
      method = 1;
    else if (method_name == "ldl")
      method = 2;
    else if (method_name == "lu")
      method = 3;
    else {
      std::cout << "Error: not supported solution method\n";
  }
  }
  return IS_TRUE;
}


/// Setup function
template <typename Scalar>
int TachoSystem<Scalar>::setup(const mxArray* mx) {
  // decide whether do do default or custom setup
  CrsMatrixBaseTypeHost A;
  loadMatrixFromMatlab<double, host_device_type>(mx, A);
  try {
    printf( " solveer.analyze\n" );
    solver.analyze(A.NumRows(), A.RowPtr(), A.Cols());
    printf( " solveer.initialize\n" );
    solver.initialize();
    printf( " setup done\n" );
  } catch (std::exception& e) {
    std::cout << "An error occurred during Tpetra custom problem setup:" << std::endl;
    std::cout << e.what() << std::endl;
    return IS_FALSE;
  }
  return IS_TRUE;
}


/// Solve function
template <typename Scalar>
mxArray* TachoSystem<Scalar>::solve(const mxArray* mx) {
  DenseMultiVectorType b;
  loadVectorsFromMatlab<double, host_device_type>(mx, b);
  DenseMultiVectorType x ("x", b.extent(0), b.extent(1));
  DenseMultiVectorType t ("t", b.extent(0), b.extent(1));
  mxArray* output;
  try {
    printf( " solveer.solve\n" );
    this->solver.solve(x, b, t);
    printf( " solveer.solve done\n" );
    output = saveVectorsToMatlab<double, host_device_type>(x);
    printf( " solution saved\n" );
  } catch (std::exception& e) {
    mexPrintf("Error occurred while running Belos solver:\n");
    std::cout << e.what() << std::endl;
    output = mxCreateDoubleScalar(0);
  }
  return output;
}


/// Apply function
template <typename Scalar>
mxArray* TachoSystem<Scalar>::apply(const mxArray* mx) {
  //DenseMultiVectorType b = loadDataFromMatlab<DenseMultiVectorType>(mx);
  //DenseMultiVectorType x ("x", b.extent(0), b.extent(1));
  //DenseMultiVectorType t ("t", b.extent(0), b.extent(1));
  mxArray* output;
  try {
    //this->solver.solve(x, b, t);
  } catch (std::exception& e) {
    mexPrintf("Error occurred while applying MueLu-Tpetra preconditioner:\n");
    std::cout << e.what() << std::endl;
  }
  //output = saveDataToMatlab(x);
  return output;
}


// Helper function
MODE_TYPE sanity_check(int nrhs, const mxArray* prhs[]) {
  MODE_TYPE rv = MODE_ERROR;
  /* Check for mode */
  if (nrhs == 0)
    mexErrMsgTxt("Error: TachoMex() expects at least one argument\n");
  /* Pull mode data from 1st Input */
  MODE_TYPE mode = (MODE_TYPE)loadDataFromMatlab<int>(prhs[0]);
  switch (mode) {
    case MODE_SETUP:
      if(nrhs == 2) {
        rv = MODE_SETUP;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for setup phase\n");
      }
      break;
    case MODE_SOLVE:
      // problem ID and matrix or rhs must be numeric
      if(nrhs == 2) {
        rv = MODE_SOLVE;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for solve phase\n");
      }
      break;
    case MODE_APPLY:
      // problem ID and RHS must be numeric
      if(nrhs == 2) {
        rv = MODE_APPLY;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for apply phase\n");
      }
      break;
    case MODE_CLEANUP:
      if(nrhs == 1) {
        rv = MODE_CLEANUP;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for cleanup phase\n");
      }
      break;
    case MODE_OPTION:
      if(nrhs == 2 || nrhs == 3) {
        rv = MODE_OPTION;
      } else {
        printf("nrhs = %d\n", (int)nrhs);
        mexErrMsgTxt("TachoMex Error: Invalid input for option phase\n");
      }
      break;
     default:
      printf("Mode number = %d\n", (int)mode);
      mexErrMsgTxt("Error: Invalid input mode\n");
  };
  return rv;
}

} // close Tacho namespace


// Entry fuction
using namespace Tacho; 
TachoSystem<double> dp;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (!Kokkos::is_initialized()) {
    int argc    = 0;
    char** argv = NULL;
    Kokkos::initialize(argc, argv);
  }
  int rv;
  /* Sanity Check Input */
  MODE_TYPE  mode = sanity_check(nrhs, prhs);

  switch (mode) {
    case MODE_SETUP: {
      try {
        printf( " -- Setup --\n" );
        rv =  dp.setup(prhs[1]);
	if (nlhs > 0) {
          plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
          *((int*)mxGetData(plhs[0])) = rv;
	}
        printf( " -- Setup done --\n" );
        mexLock();
        printf( " -- Locked --\n\n" );
      } catch (std::exception& e) {
        mexPrintf("An error occurred during setup routine:\n");
        std::cout << e.what() << std::endl;
      }
      break;
    }
    case MODE_SOLVE: {
      try {
        // get pointer to MATLAB array that will be "B" or "rhs" multivector
        printf( " -- Solve --\n" );
        mxArray* output = dp.solve(prhs[1]);
	if (nlhs > 0) {
          printf( " > return x\n" );
          plhs[0] = output;
	}
        printf( " -- Solve done! --\n\n" );
      } catch (std::exception& e) {
        mexPrintf("An error occurred during the solve routine:\n");
        std::cout << e.what() << std::endl;
      }
      break;
    }
    case MODE_APPLY: {
      try {
        // MODE_APPLY, probID, rhsVec
        // prhs[0] holds the MODE_APPLY enum value
        printf( " -- Apply --\n" );
        plhs[0] = dp.apply(prhs[1]);
        printf( " -- Apply done --\nn\n" );
      } catch (std::exception& e) {
        mexPrintf("An error occurred during the apply routine:\n");
        std::cout << e.what() << std::endl;
      }
      break;
    }
    case MODE_CLEANUP: {
      try {
        mexUnlock();
      } catch (std::exception& e) {
        mexPrintf("An error occurred during the cleanup routine:\n");
        std::cout << e.what() << std::endl;
      }
      break;
    }
    case MODE_OPTION: {
      try {
        rv =  dp.option(&prhs[1]);
      } catch (std::exception& e) {
        mexPrintf("An error occurred during the cleanup routine:\n");
        std::cout << e.what() << std::endl;
      }
      break;
    }
    case MODE_ERROR:
      mexPrintf("\n **tachoMex error.**\n\n");
      break;
    default:
      mexPrintf("Mode not supported yet.");
  }
}

