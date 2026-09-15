#pragma once

#include <vector>
#include <mpi.h>
#include "throwAssert.h"

#include "Teuchos_CommHelpers.hpp"

template <typename SC>
class GatherToRootSimple
{
public:
  using comm_type = Teuchos::MpiComm<int>;

  GatherToRootSimple(const std::vector<int> & rowBeginIn,
                     const std::vector<int> & columnsIn,
                     Teuchos::RCP<comm_type> commIn);

  int getMyPID();
  
  void initialize();
  
  void gatherMatrix(const std::vector<SC> & values,
                          std::vector<SC> & valuesTarget);

  void gatherRhs(const std::vector<SC> & rhs,
                       std::vector<SC> & rhsRoot);
  
  void scatterSol(const std::vector<SC> & solRoot,
                        std::vector<SC> & sol);
  
  void broadcastSol(std::vector<SC> & solRoot);
  
  SC checkMatrix(const std::vector<SC> & values,
                 const std::vector<SC> & valuesTarget);

  const std::vector<int> & getRowBeginRoot();

  const std::vector<int> & getColumnsRoot();

 private:
  
  void getDispls(const std::vector<int> & numEntriesProc,
                 std::vector<int> & displs) const;

  const std::vector<int> & rowBegin;
  const std::vector<int> & columns;
  Teuchos::RCP<comm_type> comm;

  std::vector<int> rowBeginRoot, columnsRoot, numRowsProc, nnzProc;
  int myPID, numProc, root=0;
  
};

