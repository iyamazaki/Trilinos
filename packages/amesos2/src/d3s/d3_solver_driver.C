#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stack>
#include <sys/time.h>
#include <sys/resource.h>
#include <cmath>

#include "generate_problem.h"
#include "d3_solver.h"

void checkSolution(const std::vector<int> & rowBegin,
                   const std::vector<int> & columns,
                   const std::vector<double> & values,
                   const std::vector<double> & rhs,
                   const std::vector<double> & solAll,
                   MPI_Comm comm)
{
  double rhs2(0), diff2(0);
  const int numRows = rowBegin.size() - 1;
  for (int i=0; i<numRows; i++) {
    rhs2 += rhs[i]*rhs[i];
    double diff = rhs[i];
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      diff -= values[j]*solAll[col];
    }
    diff2 += diff*diff;
  }
  double array[2], array_sum[2];
  array[0] = rhs2;
  array[1] = diff2;
  int myPID;
  MPI_Comm_rank(comm, &myPID);
  const int root = 0;
  MPI_Reduce(array, array_sum, 2, MPI_DOUBLE, MPI_SUM, root, comm);
  if (myPID == root) {
    const double rhs2_sum = array_sum[0];
    const double diff2_sum = array_sum[1];
    if (rhs2_sum == 0) {
      std::cout << "rhs has norm of zero" << std::endl;
    }
    else {
      const double relative_residual = std::sqrt(diff2_sum/rhs2_sum);
      std::cout << "relative residual = " << relative_residual << std::endl;
    }
  }
}

inline double clockIt()
{
  struct timeval start;
  gettimeofday(&start, NULL);
  double duration = 
    (double)(start.tv_sec + start.tv_usec/1000000.0);
  return duration;
}

void printTiming(std::string message,
                 const int myPID,
                 const double elapsedTime)
{
  if (myPID != 0) return;
  message = message + " time = ";
  std::cout << message << elapsedTime << std::endl;
}

int main(int argc, char *argv[]){
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int myPID, numProc;
  MPI_Comm_rank(comm, &myPID);
  MPI_Comm_size(comm, &numProc);
  if (numProc < 2) {
    std::cout << "linear solver currently requires at least 2 MPI processes" << std::endl;
    return 1;
  }
  // The same file is used either when reading a matrix from files
  // or when generating it on-the-fly. Some entries in the file will be ignored
  // depending on matrix_option, but all need to be in the file.
  const std::string inputFile = "d3_driver.inp";
  GenerateProblem problem(comm, inputFile);
  const int numRowsAll = problem.getNumRowsAll();
  const int numProcSolver = problem.getNumProcSolver();
  const int num_threads = problem.getNumThreads();
  const int add_nonsymmetry = problem.getAddNonsymmetry();
  const int remove_some_entries = problem.getRemoveSomeEntries();
  const int msg_level = problem.getMessageLevel();
  const int reorder_option = problem.getReorderOption();
  const int debugLevel = problem.getDebugLevel();
  const int num_factorizations = problem.getNumFactorizations();
  std::vector<int> rowBegin, columns;
  std::vector<double> values, rhs;
  int startGID;
  problem.getLinearSystem(rowBegin, columns, values, rhs, startGID);
  if (myPID == 0) {
    std::cout << "number of rows       = " << numRowsAll << std::endl;
    std::cout << "number of subdomains = " << numProcSolver << std::endl;
    std::cout << "number of threads    = " << num_threads << std::endl;
    if (add_nonsymmetry) {
      std::cout << "artificially adding nonsymmetry to matrix (for testing only)" << std::endl;
    }
    if (remove_some_entries) {
      std::cout << "artificially removing some matrix entries to remove structural symmetry"
                << " (for testing only)" << std::endl;
    }
  }
  {
    D3Solver solver(comm, msg_level, num_threads, reorder_option, debugLevel);
    // initialization
    MPI_Barrier(comm);
    double startTime = clockIt();
    solver.initialize(rowBegin, columns, startGID, numProcSolver);
    MPI_Barrier(comm);
    double elapsedTime = clockIt() - startTime;
    printTiming("initialization", myPID, elapsedTime);
    const int numRows = rowBegin.size() - 1;
    std::vector<double> sol(numRows), solAll;
    for (int i=0; i<num_factorizations; i++) {
      // factorization
      // scale diagonal for subsequent factorizations and solves
      if (i > 0) problem.scaleDiagonal();
      MPI_Barrier(comm);
      startTime = clockIt();
      solver.factorize(values);
      MPI_Barrier(comm);
      elapsedTime = clockIt() - startTime;
      std::string text = std::to_string(i) + ": d3_solver numeric factorization";
      printTiming(text, myPID, elapsedTime);
      // solve
      MPI_Barrier(comm);
      startTime = clockIt();
      solver.solve(rhs, sol);
      MPI_Barrier(comm);
      elapsedTime = clockIt() - startTime;
      text = std::to_string(i)             + ": d3_solver solve                ";
      printTiming(text, myPID, elapsedTime);
      solver.gatherScatterSol(sol, solAll);
      checkSolution(rowBegin, columns, values, rhs, solAll, comm);
    }
    // solver timers
    solver.output_timers();
    
  }
  
  MPI_Finalize();
  return 0;  
}
