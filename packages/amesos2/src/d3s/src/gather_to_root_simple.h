#pragma once

#include <vector>
#include <mpi.h>
#include "throwAssert.h"

#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_Core.hpp"

template <typename SC, typename GO = int, typename GS = int>
class GatherToRootSimple
{
public:
  using comm_type = Teuchos::MpiComm<int>;

  GatherToRootSimple(const std::vector<GS> & rowBeginIn,
                     const std::vector<GO> & columnsIn,
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

  const std::vector<GS> & getRowBeginRoot();

  const std::vector<GO> & getColumnsRoot();

 private:
  
  void getDispls(const std::vector<int> & numEntriesProc,
                 std::vector<int> & displs) const;

  const std::vector<GS> & rowBegin;
  const std::vector<GO> & columns;
  Teuchos::RCP<comm_type> comm;

  std::vector<GS> rowBeginRoot;
  std::vector<GO> columnsRoot;
  std::vector<int> numRowsProc, nnzProc;
  int myPID, numProc, root=0;
  
};

