#pragma once

#include <vector>
#include <mpi.h>
#include "throwAssert.h"

class GatherToRootSimple
{
public:
  GatherToRootSimple(const std::vector<int> & rowBeginIn,
                     const std::vector<int> & columnsIn,
                     MPI_Comm commIn);

  int getMyPID();
  
  void initialize();
  
  void gatherMatrix(const std::vector<double> & values,
                    std::vector<double> & valuesTarget);

  void gatherRhs(const std::vector<double> & rhs,
                 std::vector<double> & rhsRoot);
  
  void scatterSol(const std::vector<double> & solRoot,
                  std::vector<double> & sol);
  
  void broadcastSol(std::vector<double> & solRoot);
  
  double checkMatrix(const std::vector<double> & values,
                     const std::vector<double> & valuesTarget);

  const std::vector<int> & getRowBeginRoot();

  const std::vector<int> & getColumnsRoot();

 private:
  
  void getDispls(const std::vector<int> & numEntriesProc,
                 std::vector<int> & displs) const;

  const std::vector<int> & rowBegin;
  const std::vector<int> & columns;
  MPI_Comm comm;

  std::vector<int> rowBeginRoot, columnsRoot, numRowsProc, nnzProc;
  int myPID, numProc, root=0;
  
};

