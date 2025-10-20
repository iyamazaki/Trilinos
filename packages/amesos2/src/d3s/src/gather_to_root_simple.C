#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string>
#include <algorithm>
#include <map>
#include <cstdlib>
#include "gather_to_root_simple.h"

GatherToRootSimple::GatherToRootSimple(const std::vector<int> & rowBeginIn,
                                       const std::vector<int> & columnsIn,
                                       MPI_Comm commIn):
  rowBegin(rowBeginIn),
  columns(columnsIn),
  comm(commIn)
{
  MPI_Comm_rank(comm, &myPID);
  MPI_Comm_size(comm, &numProc);
}

const std::vector<int> & GatherToRootSimple::getRowBeginRoot()
{
  return rowBeginRoot;
}

const std::vector<int> & GatherToRootSimple::getColumnsRoot()
{
  return columnsRoot;
}

int GatherToRootSimple::getMyPID()
{
  return myPID;
}

void GatherToRootSimple::gatherMatrix(const std::vector<double> & values,
                                           std::vector<double> & valuesRoot)
{
  std::vector<int> displs;
  getDispls(nnzProc, displs);
  const int numRows = rowBegin.size() - 1;
  const int nnz = rowBegin[numRows];
  const int numRowsRoot = rowBeginRoot.size() - 1;
  const int nnzRoot = rowBeginRoot[numRowsRoot];
  valuesRoot.resize(nnzRoot);
  MPI_Gatherv(values.data(), nnz, MPI_DOUBLE, valuesRoot.data(), nnzProc.data(),
              displs.data(), MPI_DOUBLE, root, comm);
}

void GatherToRootSimple::gatherRhs(const std::vector<double> & rhs,
                                        std::vector<double> & rhsRoot)
{
  std::vector<int> displs;
  const int numRows = rowBegin.size() - 1;
  getDispls(numRowsProc, displs);
  const int numRowsRoot = rowBeginRoot.size() - 1;
  rhsRoot.resize(numRowsRoot);
  MPI_Gatherv(rhs.data(), numRows, MPI_DOUBLE, rhsRoot.data(), numRowsProc.data(),
              displs.data(), MPI_DOUBLE, root, comm);
}

void GatherToRootSimple::scatterSol(const std::vector<double> & solRoot,
                                    std::vector<double> & sol)
{
  std::vector<int> displs;
  getDispls(numRowsProc, displs);
  const int numRows = rowBegin.size() - 1;
  sol.resize(numRows);
  MPI_Scatterv(solRoot.data(), numRowsProc.data(), displs.data(), MPI_DOUBLE,
               sol.data(), numRows, MPI_DOUBLE, root, comm);
}

void GatherToRootSimple::broadcastSol(std::vector<double> & solRoot)
{
  int numRowsRoot = rowBeginRoot.size() - 1;
  MPI_Bcast(&numRowsRoot, 1, MPI_DOUBLE, root, comm);
  solRoot.resize(numRowsRoot);
  MPI_Bcast(solRoot.data(), numRowsRoot, MPI_DOUBLE, root, comm);
}

void GatherToRootSimple::getDispls(const std::vector<int> & numEntriesProc,
                                   std::vector<int> & displs) const
{
  if (myPID == root) {
    displs.resize(numProc, 0);
    for (int i=1; i<numProc; i++) {
      displs[i] = displs[i-1] + numEntriesProc[i-1];
    }
  }
}

void GatherToRootSimple::initialize()
{
  const int numRows = rowBegin.size() - 1;
  const int nnz = rowBegin[numRows];
  if (myPID == root) {
    numRowsProc.resize(numProc);
    nnzProc.resize(numProc);
  }
  MPI_Gather(&numRows, 1, MPI_INT, numRowsProc.data(), 1, MPI_INT, root, comm);
  MPI_Gather(&nnz, 1, MPI_INT, nnzProc.data(), 1, MPI_INT, root, comm);
  int numRowsRoot(0), nnzRoot(0);
  for (size_t i=0; i<numRowsProc.size(); i++) {
    numRowsRoot += numRowsProc[i];
    nnzRoot += nnzProc[i];
  }
  std::vector<int> count(numRows);
  for (int i=0; i<numRows; i++) {
    count[i] = rowBegin[i+1] - rowBegin[i];
  }
  // gather number of nonzeros in each row
  std::vector<int> countRoot(numRowsRoot), displs;
  getDispls(numRowsProc, displs);
  MPI_Gatherv(count.data(), numRows, MPI_INT, countRoot.data(), numRowsProc.data(), displs.data(),
              MPI_INT, root, comm);
  rowBeginRoot.resize(numRowsRoot+1, 0);
  for (int i=0; i<numRowsRoot; i++) {
    rowBeginRoot[i+1] = rowBeginRoot[i] + countRoot[i];
  }
  // gather nonzero columns
  columnsRoot.resize(nnzRoot);
  getDispls(nnzProc, displs);
  MPI_Gatherv(columns.data(), nnz, MPI_INT, columnsRoot.data(), nnzProc.data(), displs.data(),
              MPI_INT, root, comm);
}

double GatherToRootSimple::checkMatrix(const std::vector<double> & values,
                                       const std::vector<double> & valuesRoot)
{
  const int numRows = rowBegin.size() - 1;
  std::vector<double> x(numRows), Ax(numRows);
  srand(myPID + 7);
  for (int i=0; i<numRows; i++) {
    x[i] = 0.7*rand()/RAND_MAX;
  }
  int numRowsRoot = rowBeginRoot.size() - 1;
  std::vector<double> xRoot(numRowsRoot);
  std::vector<int> displs;
  getDispls(numRowsProc, displs);
  MPI_Gatherv(x.data(), numRows, MPI_DOUBLE, xRoot.data(), numRowsProc.data(), displs.data(),
              MPI_DOUBLE, root, comm);
  MPI_Bcast(&numRowsRoot, 1, MPI_INT, root, comm);
  xRoot.resize(numRowsRoot);
  MPI_Bcast(xRoot.data(), numRowsRoot, MPI_DOUBLE, root, comm);
  for (int i=0; i<numRows; i++) {
    double sum = 0;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      sum += values[j] * xRoot[columns[j]];
    }
    Ax[i] = sum;
  }
  numRowsRoot = rowBeginRoot.size() - 1;
  std::vector<double> AxRootTrue(numRowsRoot);
  getDispls(numRowsProc, displs);
  MPI_Gatherv(Ax.data(), numRows, MPI_DOUBLE, AxRootTrue.data(), numRowsProc.data(), displs.data(),
              MPI_DOUBLE, root, comm);
  double maxRelError = 0;
  for (int i=0; i<numRowsRoot; i++) {
    double sum(0), sumAbsCoeff(0);
    for (int j=rowBeginRoot[i]; j<rowBeginRoot[i+1]; j++) {
      sum += valuesRoot[j] * xRoot[columnsRoot[j]];
      sumAbsCoeff += std::abs(valuesRoot[j]);
    }
    const double relError = std::abs(sum - AxRootTrue[i]) / sumAbsCoeff;
    if (relError > maxRelError) maxRelError = relError;
  }
  MPI_Bcast(&maxRelError, 1, MPI_DOUBLE, root, comm);
  return maxRelError;
}
