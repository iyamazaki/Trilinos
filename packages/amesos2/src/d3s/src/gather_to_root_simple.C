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

template <typename SC>
GatherToRootSimple<SC>::GatherToRootSimple(const std::vector<int> & rowBeginIn,
                                           const std::vector<int> & columnsIn,
                                           Teuchos::RCP<comm_type> commIn):
  rowBegin(rowBeginIn),
  columns(columnsIn),
  comm(commIn)
{
  myPID = comm->getRank();
  numProc = comm->getSize();
}

template <typename SC>
const std::vector<int> & GatherToRootSimple<SC>::getRowBeginRoot()
{
  return rowBeginRoot;
}

template <typename SC>
const std::vector<int> & GatherToRootSimple<SC>::getColumnsRoot()
{
  return columnsRoot;
}

template <typename SC>
int GatherToRootSimple<SC>::getMyPID()
{
  return myPID;
}

template <typename SC>
void GatherToRootSimple<SC>::gatherMatrix(const std::vector<SC> & values,
                                                std::vector<SC> & valuesRoot)
{
  std::vector<int> displs;
  getDispls(nnzProc, displs);
  const int numRows = rowBegin.size() - 1;
  const int nnz = rowBegin[numRows];
  const int numRowsRoot = rowBeginRoot.size() - 1;
  const int nnzRoot = rowBeginRoot[numRowsRoot];
  valuesRoot.resize(nnzRoot);
  Teuchos::gatherv<int, SC>(values.data(), nnz, valuesRoot.data(), nnzProc.data(),
                            displs.data(), root, *comm);
}

template <typename SC>
void GatherToRootSimple<SC>::gatherRhs(const std::vector<SC> & rhs,
                                             std::vector<SC> & rhsRoot)
{
  std::vector<int> displs;
  const int numRows = rowBegin.size() - 1;
  getDispls(numRowsProc, displs);
  const int numRowsRoot = rowBeginRoot.size() - 1;
  rhsRoot.resize(numRowsRoot);
  Teuchos::gatherv<int, SC>(rhs.data(), numRows, rhsRoot.data(), numRowsProc.data(),
                            displs.data(), root, *comm);
}

template <typename SC>
void GatherToRootSimple<SC>::scatterSol(const std::vector<SC> & solRoot,
                                              std::vector<SC> & sol)
{
  std::vector<int> displs;
  getDispls(numRowsProc, displs);
  const int numRows = rowBegin.size() - 1;
  sol.resize(numRows);
  Teuchos::scatterv<int, SC>(solRoot.data(), numRowsProc.data(), displs.data(),
                             sol.data(), numRows, root, *comm);
}

template <typename SC>
void GatherToRootSimple<SC>::broadcastSol(std::vector<SC> & solRoot)
{
  int numRowsRoot = rowBeginRoot.size() - 1;
  Teuchos::broadcast<int, int>(*comm, root, 1, &numRowsRoot);
  solRoot.resize(numRowsRoot);
  Teuchos::broadcast<int, SC>(*comm, root, numRowsRoot, solRoot.data());
}

template <typename SC>
void GatherToRootSimple<SC>::getDispls(const std::vector<int> & numEntriesProc,
                                             std::vector<int> & displs) const
{
  if (myPID == root) {
    displs.resize(numProc, 0);
    for (int i=1; i<numProc; i++) {
      displs[i] = displs[i-1] + numEntriesProc[i-1];
    }
  }
}

template <typename SC>
void GatherToRootSimple<SC>::initialize()
{
  const int numRows = rowBegin.size() - 1;
  const int nnz = rowBegin[numRows];
  if (myPID == root) {
    numRowsProc.resize(numProc);
    nnzProc.resize(numProc);
  }
  Teuchos::gather<int,int>(&numRows, 1, numRowsProc.data(), 1, root, *comm);
  Teuchos::gather<int,int>(&nnz, 1, nnzProc.data(), 1, root, *comm);
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
  Teuchos::gatherv<int, int>(count.data(), numRows, countRoot.data(), numRowsProc.data(),
                             displs.data(), root, *comm);
  rowBeginRoot.resize(numRowsRoot+1, 0);
  for (int i=0; i<numRowsRoot; i++) {
    rowBeginRoot[i+1] = rowBeginRoot[i] + countRoot[i];
  }
  // gather nonzero columns
  columnsRoot.resize(nnzRoot);
  getDispls(nnzProc, displs);
  Teuchos::gatherv<int, int>(columns.data(), nnz, columnsRoot.data(), nnzProc.data(),
                             displs.data(), root, *comm);
}

template <typename SC>
SC GatherToRootSimple<SC>::checkMatrix(const std::vector<SC> & values,
                                       const std::vector<SC> & valuesRoot)
{
  using STS = Teuchos::ScalarTraits<SC>;
  using MAG = STS::magnitudeType;
  const int numRows = rowBegin.size() - 1;
  std::vector<SC> x(numRows), Ax(numRows);
  srand(myPID + 7);
  for (int i=0; i<numRows; i++) {
    x[i] = 0.7*rand()/RAND_MAX;
  }
  int numRowsRoot = rowBeginRoot.size() - 1;
  std::vector<SC> xRoot(numRowsRoot);
  std::vector<int> displs;
  getDispls(numRowsProc, displs);
  Teuchos::gatherv<int, SC>(x.data(), numRows, xRoot.data(), numRowsProc.data(),
                            displs.data(), root, *comm);
  Teuchos::broadcast<int, int>(*comm, root, 1, &numRowsRoot);
  xRoot.resize(numRowsRoot);
  Teuchos::broadcast<int, SC>(*comm, root, numRowsRoot, xRoot.data());
  for (int i=0; i<numRows; i++) {
    SC sum = 0;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      sum += values[j] * xRoot[columns[j]];
    }
    Ax[i] = sum;
  }
  numRowsRoot = rowBeginRoot.size() - 1;
  std::vector<SC> AxRootTrue(numRowsRoot);
  getDispls(numRowsProc, displs);
  Teuchos::gatherv<int, SC>(Ax.data(), numRows, AxRootTrue.data(), numRowsProc.data(),
                            displs.data(), root, *comm);
  MAG maxRelError = 0;
  for (int i=0; i<numRowsRoot; i++) {
    SC sum(0);
    MAG sumAbsCoeff(0);
    for (int j=rowBeginRoot[i]; j<rowBeginRoot[i+1]; j++) {
      sum += valuesRoot[j] * xRoot[columnsRoot[j]];
      sumAbsCoeff += STS::magnitude(valuesRoot[j]);
    }
    const MAG relError = STS::magnitude(sum - AxRootTrue[i]) / sumAbsCoeff;
    if (relError > maxRelError) maxRelError = relError;
  }
  Teuchos::broadcast<int, MAG>(*comm, root, 1, &maxRelError);
  return maxRelError;
}

// ...ETI...
template class GatherToRootSimple<double>;
#ifdef HAVE_TEUCHOS_INST_FLOAT
template class GatherToRootSimple<float>;
#endif
#ifdef HAVE_TEUCHOS_INST_COMPLEX_DOUBLE
//template class GatherToRootSimple<std::complex<double>>;
//template class GatherToRootSimple<Kokkos::complex<double>>;
#endif
#ifdef HAVE_TEUCHOS_INST_COMPLEX_FLOAT
//template class GatherToRootSimple<std::complex<float>>;
//template class GatherToRootSimple<Kokkos::complex<float>>;
#endif
