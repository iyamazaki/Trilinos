// clang-format off
// @HEADER
// *****************************************************************************
//                            Tacho package
//
// Copyright 2022 NTESS and the Tacho contributors.
// SPDX-License-Identifier: BSD-2-Clause
// *****************************************************************************
// @HEADER

#include "tachoMex.h"

extern void _main();

namespace Tacho {

template <>
TachoSystem<double>::TachoSystem() :
 _verbose(false),
 _dofs_per_node(1),
 _max_match(-1),
 _scale_mat(false)
{}
template <>
TachoSystem<double>::~TachoSystem() {}



/// SetParameters function
template <typename Scalar>
int TachoSystem<Scalar>::option(const mxArray** mx) {
  std::string option_name = loadDataFromMatlab<std::string>(mx[0]);
  if (option_name == "verbose") {
    _verbose = true;
    solver.setVerbose(_verbose);
    if (_verbose) printf( " > option(verbose)\n" );
  } else if (option_name == "method") {
    int method = 1;
    std::string method_name = loadDataFromMatlab<std::string>(mx[1]);
    if (_verbose) printf( " > option(method=%s)\n",method_name.c_str() );
    if (method_name == "chol")
      method = 1;
    else if (method_name == "ldl")
      method = 2;
    else if (method_name == "lu")
      method = 3;
    else if (method_name == "sk")
      method = 5;
    else {
      std::cout << "Error: not supported solution method\n";
    }
    solver.setSolutionMethod(method);
  } else if (option_name == "dofs-per-node") {
    _dofs_per_node = loadDataFromMatlab<int>(mx[1]);
    if (_verbose) printf( " > option(dofs-per-node=%d)\n",_dofs_per_node );
  } else if (option_name == "small-problem-thres") {
    int small_problem_thres = loadDataFromMatlab<int>(mx[1]);
    if (_verbose) printf( " > option(small-problem-thres=%d)\n",small_problem_thres );
    solver.setSmallProblemThresholdsize(small_problem_thres);
  } else if (option_name == "max-match") {
    if (_verbose) printf( " > option(max-match = true)\n" );
    _max_match = 0;
  } else if (option_name == "max-weight") {
    _max_match = loadDataFromMatlab<int>(mx[1]);
    if (_verbose) printf( " > option(max-weight = %d)\n",_max_match );
  } else if (option_name == "scale-mat") {
    if (_verbose) printf( " > option(scale-mat = true)\n" );
    _scale_mat = true;
  }
  return 0;
}


/// Setup function
template <typename Scalar>
int TachoSystem<Scalar>::setup(const mxArray* mx) {
  loadMatrixFromMatlab<double, host_device_type>(mx, A);
  try {
    if (_verbose) printf( " solver.analyze(%d%s%s)\n",_dofs_per_node,(_max_match >= 0 ? ", max-match" : ""),(_max_match > 0 ? ", max-weight" : "") );
    if (_dofs_per_node > 1) {
      if (_max_match > 0) {
        solver.analyze(A.NumRows(), _dofs_per_node, A.RowPtr(), A.Cols(), A.Values(), _max_match, _scale_mat);
      } else {
        solver.analyze(A.NumRows(), _dofs_per_node, A.RowPtr(), A.Cols(), _max_match);
      }
    } else {
      solver.analyze(A.NumRows(), A.RowPtr(), A.Cols());
    }
    if (_verbose) printf( " solver.initialize\n" );
    solver.initialize();
    if (_verbose) printf( " setup done\n" );
  } catch (std::exception& e) {
    std::cout << "An error occurred during TachoMex setup:" << std::endl;
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}


/// Factor function
template <typename Scalar>
int TachoSystem<Scalar>::factor(const mxArray* mx) {
  loadMatrixFromMatlab<double, host_device_type>(mx, A);
  try {
    if (_verbose) printf( " solver.factorize\n" );
    solver.factorize(A.Values());
    if (_verbose) printf( " factorize done\n" );
  } catch (std::exception& e) {
    std::cout << "An error occurred during TachoMex setup:" << std::endl;
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}


/// Solve function
template <typename Scalar>
mxArray* TachoSystem<Scalar>::solve(const mxArray* mx) {
  DenseMultiVectorType b;
  loadMultiVectorsFromMatlab<double, host_device_type>(mx, b);
  DenseMultiVectorType x ("x", b.extent(0), b.extent(1));
  DenseMultiVectorType t ("t", b.extent(0), b.extent(1));
  mxArray* output;
  try {
    if (_verbose) printf( " solver.solve\n" );
    this->solver.solve(x, b, t);
    if (_verbose) printf( " solver.solve done\n" );
    output = saveMultiVectorsToMatlab<double, host_device_type>(x);
    if (_verbose) printf( " solution saved\n" );
  } catch (std::exception& e) {
    mexPrintf("Error occurred during TachoMex solve:\n");
    std::cout << e.what() << std::endl;
    output = mxCreateDoubleScalar(0);
  }
  return output;
}


/// Diag function
template <typename Scalar>
mxArray* TachoSystem<Scalar>::diag() {
  DenseVectorType d ("d", A.NumRows());
  mxArray* output;
  try {
    if (_verbose) printf( " solver.diag\n" );
    this->solver.diag(d);
    output = saveVectorToMatlab<double, host_device_type>(d);
  } catch (std::exception& e) {
    mexPrintf("Error occurred during TachoMex diag:\n");
    std::cout << e.what() << std::endl;
    output = mxCreateDoubleScalar(0);
  }
  return output;
}


/// Pfaffian function
template <typename Scalar>
int TachoSystem<Scalar>::pfaffian() {
  int pf = 0;
  try {
    if (_verbose) printf( " solver.diag\n" );
    pf = this->solver.pfaffian();
  } catch (std::exception& e) {
    mexPrintf("Error occurred during TachoMex pfaffian:\n");
    std::cout << e.what() << std::endl;
  }
  return pf;
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
    case MODE_FACTOR:
      if(nrhs == 2) {
        rv = MODE_FACTOR;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for factor phase\n");
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
    case MODE_CLEANUP:
      if(nrhs == 1) {
        rv = MODE_CLEANUP;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for cleanup phase\n");
      }
      break;
    case MODE_DIAG:
      // problem ID and matrix or rhs must be numeric
      if(nrhs == 1) {
        rv = MODE_DIAG;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for diag phase\n");
      }
      break;
    case MODE_PFAFFIAN:
      if(nrhs == 1) {
        rv = MODE_PFAFFIAN;
      } else {
        mexErrMsgTxt("TachoMex Error: Invalid input for pfaffian phase\n");
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
      printf("Mode number = %d (nrhs = %d)\n", (int)mode, (int)nrhs);
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
  int rv = -1;
  /* Sanity Check Input */
  MODE_TYPE  mode = sanity_check(nrhs, prhs);

  switch (mode) {
    case MODE_SETUP: {
      if (dp.verbose()) printf( " -- Setup --\n" );
      try {
        rv = dp.setup(prhs[1]);
      } catch (std::exception& e) {
        mexPrintf("An error occurred during setup routine:\n");
        std::cout << e.what() << std::endl;
      }
      if (nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        *((int*)mxGetData(plhs[0])) = rv;
      }
      if (dp.verbose()) printf( " -- Setup done (rval = %d, nlhs = %d) --\n",rv,nlhs );
      mexLock();
      break;
    }
    case MODE_FACTOR: {
      if (dp.verbose()) printf( " -- Factor --\n" );
      try {
        rv =  dp.factor(prhs[1]);
      } catch (std::exception& e) {
        mexPrintf("An error occurred during factor routine:\n");
        std::cout << e.what() << std::endl;
      }
      if (nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        *((int*)mxGetData(plhs[0])) = rv;
      }
      if (dp.verbose()) printf( " -- Factor done --\n" );
      break;
    }
    case MODE_SOLVE: {
      if (dp.verbose()) printf( " -- Solve --\n" );
      try {
        // get pointer to MATLAB array that will be "B" or "rhs" multivector
        mxArray* output = dp.solve(prhs[1]);
        if (nlhs > 0) {
          if (dp.verbose()) printf( " > return x\n" );
          plhs[0] = output;
        }
      } catch (std::exception& e) {
        mexPrintf("An error occurred during the solve routine:\n");
        std::cout << e.what() << std::endl;
      }
      if (dp.verbose()) printf( " -- Solve done! --\n\n" );
      break;
    }
    case MODE_DIAG: {
      if (dp.verbose()) printf( " -- Diag --\n" );
      try {
        // get pointer to MATLAB array that will be "B" or "rhs" multivector
        mxArray* output = dp.diag();
        if (nlhs > 0) {
          if (dp.verbose()) printf( " > return d\n" );
          plhs[0] = output;
        }
      } catch (std::exception& e) {
        mexPrintf("An error occurred during the solve routine:\n");
        std::cout << e.what() << std::endl;
      }
      if (dp.verbose()) printf( " -- Dian done! --\n\n" );
      break;
    }
    case MODE_PFAFFIAN: {
      int pf = 0;
      if (dp.verbose()) printf( " -- Pfaffian --\n" );
      try {
        pf = dp.pfaffian();
      } catch (std::exception& e) {
        pf = 0;
        mexPrintf("An error occurred during the solve routine:\n");
        std::cout << e.what() << std::endl;
      }
      if (nlhs > 0) {
        if (dp.verbose()) printf( " > return pf = %d\n",pf );
        plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        *((int*)mxGetData(plhs[0])) = pf;
      }
      if (dp.verbose()) printf( " -- Pfaffian done! --\n\n" );
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
        mexPrintf("An error occurred during the option routine:\n");
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

