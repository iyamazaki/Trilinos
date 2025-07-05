// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_CRSREINDEXFILTER_LINEARPROBLEM_DEF_HPP
#define TPETRA_CRSREINDEXFILTER_LINEARPROBLEM_DEF_HPP

/// \file Tpetra_CrsReindexFilter_LinearProblem_def.hpp
/// \brief Definition of the Tpetra::CrsReindexFilter_LinearProblem class

//#include "Epetra_ConfigDefs.h"
//#include "Epetra_Map.h"
//#include "Epetra_Util.h"
//#include "Epetra_Export.h"
//#include "Epetra_Import.h"
//#include "Epetra_MultiVector.h"
//#include "Epetra_Vector.h"
//#include "Epetra_GIDTypeVector.h"
//#include "Epetra_Comm.h"
//#include "Epetra_LinearProblem.h"
//#include "Epetra_MapColoring.h"
//#include "EpetraExt_CrsReindexFilter_LinearProblem.h"

#include "Tpetra_CrsReindexFilter_LinearProblem_decl.hpp"


namespace Tpetra {


//==============================================================================

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsReindexFilter_LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  CrsReindexFilter_LinearProblem( bool verbose ) :
    verbose_(verbose)
  {
  }

//==============================================================================

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsReindexFilter_LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  ~CrsReindexFilter_LinearProblem()
  {
  }

//==============================================================================

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  typename CrsReindexFilter_LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::NewType
  CrsReindexFilter_LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  operator()(const CrsReindexFilter_LinearProblem::OriginalType & OriginalProblem)
  {
    // Initialize Original Problem
    OriginalProblem_ = OriginalProblem;

    // Original RowMatrix
    OriginalRowMatrix_ = Teuchos::rcp_dynamic_cast<row_matrix_type>(OriginalProblem->getMatrix());
    TEUCHOS_TEST_FOR_EXCEPTION(OriginalRowMatrix_ == Teuchos::null, std::runtime_error,
      "Tpetra::CrsReindexFilter_LinearProblem::operator() needs a RowMatrix.");
    // Original CrsMatrix
    OriginalCrsMatrix_ = Teuchos::rcp_dynamic_cast<crs_matrix_type>(OriginalRowMatrix_);
    TEUCHOS_TEST_FOR_EXCEPTION(OriginalCrsMatrix_ == Teuchos::null, std::runtime_error,
      "Tpetra::CrsReindexFilter_LinearProblem::operator() failed to type-cast to CrsMatrix.");

    // Original Vectors
    OriginalLHS_ = OriginalProblem->getLHS();
    OriginalRHS_ = OriginalProblem->getRHS();
    TEUCHOS_TEST_FOR_EXCEPTION(OriginalProblem->getRHS() == Teuchos::null,
      std::runtime_error, "Tpetra::CrsReindexFilter_LinearProblem::operator() needs a RHS.");
    TEUCHOS_TEST_FOR_EXCEPTION(OriginalProblem->getLHS() == Teuchos::null,
      std::runtime_error, "Tpetra::CrsReindexFilter_LinearProblem::operator() need a LHS.");

    // Create Reindexed Problem
    ReindexedProblem_ = Teuchos::rcp( new LinearProblem(OriginalRowMatrix_, OriginalLHS_, OriginalRHS_) );

    return ReindexedProblem();
  }

//==============================================================================

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsReindexFilter_LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  fwd()
  {
      using gid_mv_t = Tpetra::MultiVector<GlobalOrdinal,
                                           LocalOrdinal,
                                           GlobalOrdinal,
                                           Node>;
      using import_t =  Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
      using HostExecSpaceType = Kokkos::DefaultHostExecutionSpace;

      auto rowMap = OriginalMatrixRowMap();
      auto colMap = OriginalMatrixColMap();
      auto rowComm = rowMap->getComm();
      auto colComm = colMap->getComm();

      GlobalOrdinal indexBase = rowMap->getIndexBase();
      GlobalOrdinal numDoFs = OriginalMatrix()->getGlobalNumRows();
      LocalOrdinal nRows = OriginalMatrix()->getLocalNumRows();
      LocalOrdinal nCols = OriginalMatrix()->getLocalNumCols(); // CHECK
      {
        auto tmpMap = Teuchos::rcp (new map_type (numDoFs, nRows, indexBase, rowComm));
        GlobalOrdinal frow = tmpMap->getMinGlobalIndex();

        // Create new GID list for RowMap
        Kokkos::View<GlobalOrdinal*, HostExecSpaceType> rowIndexList ("rowIndexList", nRows);
        for (LocalOrdinal k = 0; k < nRows; k++) {
          rowIndexList(k) = frow+k; // based on index-base of rowMap
        }
        // Create new GID list for ColMap
        Kokkos::View<GlobalOrdinal*, HostExecSpaceType> colIndexList ("colIndexList", nCols);
        // initialize to catch col GIDs that are not in row GIDs
        // they will be all assigned to (n+1)th columns
        for (LocalOrdinal k = 0; k < nCols; k++) {
          colIndexList(k) = numDoFs+indexBase;
        }

        Teuchos::ArrayView<const GlobalOrdinal> rowIndexArray(rowIndexList.data(), nRows);
        Teuchos::ArrayView<const GlobalOrdinal> colIndexArray(colIndexList.data(), nCols);
        gid_mv_t row_mv (rowMap, rowIndexArray, nRows, 1);
        gid_mv_t col_mv (colMap, colIndexArray, nCols, 1);
	Teuchos::RCP<import_t> importer_r2c = Teuchos::rcp (new import_t (rowMap, colMap));
        col_mv.doImport (row_mv, *importer_r2c, Tpetra::INSERT);
        {
          // col_mv is imported from rowIndexList, which is based on index-base of rowMap
          auto col_view = col_mv.getLocalViewHost(Tpetra::Access::ReadOnly);
          for(int i=0; i<nCols; i++) colIndexList(i) = col_view(i,0);
        }
        // Create new Row & Col Maps (both based on indexBase of rowMap)
        auto contigRowMap = Teuchos::rcp (new map_type (numDoFs, rowIndexList.data(), nRows, indexBase, rowComm));
        auto contigColMap = Teuchos::rcp (new map_type (numDoFs, colIndexList.data(), nCols, indexBase, colComm));

        // Create contiguous Matrix
        auto lclMatrix = OriginalCrsMatrix()->getLocalMatrixDevice();
        ReindexedCrsMatrix_ = Teuchos::rcp( new crs_matrix_type(contigRowMap, contigColMap, lclMatrix));
        ReindexedProblem_->setMatrix( ReindexedCrsMatrix_ );

	// Swap rowmap Vectors
	ReindexedLHS_ = OriginalLHS_; // just a shallow pointer-copy
	ReindexedRHS_ = OriginalRHS_; // just a shallow pointer-copy
	ReindexedLHS_->replaceMap(contigRowMap); // also replace row-map of OriginalLHS
	ReindexedRHS_->replaceMap(contigRowMap); // also replace row-map of OriginalRHS
      }
  }

//==============================================================================

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsReindexFilter_LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  rvs()
  {
  }

} //namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#define TPETRA_CRSREINDEXFILTER_INSTANT(SCALAR,LO,GO,NODE) \
  template class CrsReindexFilter_LinearProblem< SCALAR , LO , GO , NODE >;


#endif //  TPETRA_CRSREINDEXFILTER_LINEARPROBLEM_DEF_HPP

