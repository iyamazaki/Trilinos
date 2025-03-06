// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef GDSW_PROXY_DECL_HPP
#define GDSW_PROXY_DECL_HPP

#include <vector>
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Distributor.hpp"


namespace FROSch {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class TpetraFunctions
{

  using SC = Scalar;
  using LO = LocalOrdinal;
  using GO = GlobalOrdinal;
  using NT = Node;
  using Import = Tpetra::Import<LO,GO,NT>;
  using xMap = Xpetra::Map<LO,GO,NT>;
  using tMap = Tpetra::Map<LO,GO,NT>;
  using tCrsMatrix = Tpetra::CrsMatrix<SC,LO,GO,NT>;
  using ValuesViewT  = typename tCrsMatrix::values_host_view_type;
  using IndicesViewT = typename tCrsMatrix::local_inds_host_view_type;
  using GlobalIndicesViewT = typename tCrsMatrix::global_inds_host_view_type;
  
public:

    TpetraFunctions() = default;

    ~TpetraFunctions() = default;

    template<class local_map_device_type,
            class local_ptrs_device_const, class local_inds_device_const, class local_vals_device_const,
            class local_ptrs_device_type,  class local_inds_device_type,  class local_vals_device_type>
    struct TpetraFunctor_insert_with_map
    {
        TpetraFunctor_insert_with_map()
        {}

        TpetraFunctor_insert_with_map(local_map_device_type inputRowMap,
                                      local_map_device_type inputColMap,
                                      local_map_device_type outputRowMap,
                                      local_ptrs_device_const     localInputPtrs,
                                      local_inds_device_const     localInputInds,
                                      local_vals_device_const     localInputVals,
                                      local_ptrs_device_const    localOutputPtrs,
                                      local_inds_device_const    localOutputInds) :
        inputRowMap_ (inputRowMap),
        inputColMap_ (inputColMap),
        outputRowMap_ (outputRowMap)
        {}

        TpetraFunctor_insert_with_map(local_map_device_type inputRowMap,
                                      local_map_device_type inputColMap,
                                      local_map_device_type outputRowMap,
                                      local_ptrs_device_const     localInputPtrs,
                                      local_inds_device_const     localInputInds,
                                      local_vals_device_const     localInputVals,
                                      local_ptrs_device_const    localOutputPtrs,
                                      local_inds_device_const    localOutputInds,
                                      local_vals_device_type     localOutputVals) :
        inputRowMap_ (inputRowMap),
        inputColMap_ (inputColMap),
        outputRowMap_ (outputRowMap),
        localInputPtrs_ (localInputPtrs),
        localInputInds_ (localInputInds),
        localInputVals_ (localInputVals),
        localOutputPtrs_ (localOutputPtrs),
        localOutputInds_ (localOutputInds),
        localOutputVals_ (localOutputVals)
        {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const {
            //LO ILO = Teuchos::OrdinalTraits<LO>::invalid();
            GO globalIndex_i = inputRowMap_.getGlobalElement(i);
            LO localRowTarget = outputRowMap_.getLocalElement(globalIndex_i);
            for (size_t j=localInputPtrs_(i); j<localInputPtrs_(i+1); j++) {
                LO colIndex_In = localInputInds_(j);
                GO globalIndex_j = inputColMap_.getGlobalElement(colIndex_In);
                LO localIndex_j  = outputRowMap_.getLocalElement(globalIndex_j);
                if (localIndex_j >= 0) {
                    // look for the same column index in localSubdomainMatrix
                    for (size_t k=localOutputPtrs_(localRowTarget); k<localOutputPtrs_(localRowTarget+1); k++)
                    {
                        if (localOutputInds_(k) == localIndex_j)
                        {
                            localOutputVals_(k) = localInputVals_(j);
                            break;
                        }
                    }
                }
            }
        }
        /*KOKKOS_INLINE_FUNCTION
        void operator()(const int k) const {
            if (localOutputInds_dev(k) == localIndex_j)
            {
                localOutputVals_dev[k] = localInputVals_dev[j];
            }
        }
        int localIndex_j;
        int j;*/
        local_map_device_type inputRowMap_;
        local_map_device_type inputColMap_;
        local_map_device_type outputRowMap_;

        local_ptrs_device_const localInputPtrs_;
        local_inds_device_const localInputInds_;
        local_vals_device_const localInputVals_;

        local_ptrs_device_type localOutputPtrs_;
        local_inds_device_type localOutputInds_;
        local_vals_device_type localOutputVals_;
    };

    template<class local_map_device_type,
            class local_ptrs_device_const, class local_inds_device_const, class local_vals_device_const,
            class local_ptrs_device_type,  class local_inds_device_type,  class local_vals_device_type>
    struct TpetraFunctor_insert
    {
    };

  // --------------------------------------------------------------------------- //
  Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  importSquareMatrix_build(Teuchos::RCP<const tCrsMatrix> inputMatrix,
                           Teuchos::RCP<const xMap> outputXRowMap,
                           Teuchos::RCP<const tMap> outputTRowMap,
                           Teuchos::RCP<Tpetra::Distributor> & distributor,
                           size_t &numTerms,
                           size_t &locCount,
                           std::vector<size_t> & sourceSize,
                           std::vector<size_t> & targetSize,
                           Teuchos::ArrayRCP<size_t> & rowCount,
                           std::vector<GO> & targetMapGIDs,
                           std::vector<LO> & targetMapGIDsBegin,
                           std::vector<GO> & ownedRowGIDs,
                           std::vector<LO> & localRowsSend,
                           std::vector<LO> & localRowsSendBegin,
                           std::vector<LO> & localRowsRecv,
                           std::vector<LO> & localRowsRecvBegin,
                           std::vector<LO> & columnsRecv);

  void 
  importSquareMatrix_import(Teuchos::RCP<const tCrsMatrix> inputMatrix,
                            Teuchos::RCP<const tMap> outputRowMap,
                            Teuchos::RCP<Tpetra::Distributor> distributor,
                            const size_t numTerms,
                            const size_t locCount,
                            const std::vector<size_t> & sourceSize,
                            const std::vector<size_t> & targetSize,
                            const Teuchos::ArrayRCP<size_t> & rowCount,
                            const std::vector<GO> & targetMapGIDs,
                            const std::vector<LO> & targetMapGIDsBegin,
                            const std::vector<GO> & ownedRowGIDs,
                            const std::vector<LO> & localRowsSend,
                            const std::vector<LO> & localRowsSendBegin,
                            const std::vector<LO> & localRowsRecv,
                            const std::vector<LO> & localRowsRecvBegin,
                            const std::vector<LO> & columnsRecv,
                            Teuchos::RCP<tCrsMatrix> & outputMatrix,
                            bool replaceVals = false);
  // --------------------------------------------------------------------------- //
private:
  // --------------------------------------------------------------------------- //
  Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  communicateMatrixData_build(Teuchos::RCP<const tCrsMatrix> inputMatrix, 
                              Teuchos::RCP<const xMap> rowXMap,
                              Teuchos::RCP<const tMap> rowTMap,
                              Teuchos::RCP<Tpetra::Distributor> distributor,
                              std::vector<GO> & targetMapGIDs, 
                              std::vector<LO> & targetMapGIDsBegin,
                              std::vector<GO> & ownedRowGIDs,
                              std::vector<LO> & localRowsSend,
                              std::vector<LO> & localRowsSendBegin,
                              std::vector<LO> & localRowsRecv,
                              std::vector<LO> & localRowsRecvBegin,
                              size_t &numTerms, size_t &locCount,
                              std::vector<size_t> & sourceSize,
                              std::vector<size_t> & targetSize,
                              Teuchos::ArrayRCP<size_t> & rowCount,
                              std::vector<LO> & columnsRecv);

  void communicateMatrixData_import(Teuchos::RCP<const tCrsMatrix> inputMatrix, 
                                    Teuchos::RCP<const tMap> rowMap,
                                    Teuchos::RCP<Tpetra::Distributor> distributor,
                                    const size_t numTerms,
                                    const size_t locCount,
                                    const std::vector<size_t> & sourceSize,
                                    const std::vector<size_t> & targetSize,
                                    const Teuchos::ArrayRCP<size_t> & rowCount,
                                    const std::vector<GO> & targetMapGIDs, 
                                    const std::vector<LO> & targetMapGIDsBegin,
                                    const std::vector<GO> & ownedRowGIDs,
                                    const std::vector<LO> & localRowsSend,
                                    const std::vector<LO> & localRowsSendBegin,
                                    const std::vector<LO> & localRowsRecv,
                                    const std::vector<LO> & localRowsRecvBegin,
                                    const std::vector<LO> & columnsRecv,
                                    Teuchos::RCP<tCrsMatrix> & outputMatrix,
                                    bool replaceVals);
  // --------------------------------------------------------------------------- //
    void communicateRowMap(Teuchos::RCP<const tMap> rowMap, 
                           Teuchos::RCP<Tpetra::Distributor> distributor, 
                           std::vector<GO> & rowMapGIDs, 
                           std::vector<LO> & rowMapGIDsBegin);

    void constructDistributor(Teuchos::RCP<const tCrsMatrix> inputMatrix, 
                              Teuchos::RCP<const tMap> rowMap, 
                              Teuchos::RCP<Tpetra::Distributor> & distributor,
                              std::vector<GO> & ownedRowGIDs,
                              std::vector<LO> & localRowsSend,
                              std::vector<LO> & localRowsSendBegin,
                              std::vector<LO> & localRowsRecv,
                              std::vector<LO> & localRowsRecvBegin);

    void getUniqueEntries(const std::vector<int> & vector, 
                          std::vector<int> & vectorUnique);

};

}

#endif // GDSW_PROXY_HPP
