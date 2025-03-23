// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef GDSW_DEF_HPP
#define GDSW_DEF_HPP
#include "GDSW_Proxy_decl.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Tpetra_Details_PackTraits.hpp"
#include "MatrixMarket_Tpetra.hpp"

namespace FROSch {

using Teuchos::RCP;

// ---------------------------------------------------------------------------------------------------------- //
template <class SC, class LO, class GO, class NO>
Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>>
TpetraFunctions<SC,LO,GO,NO>::
importSquareMatrix_build(RCP<const tCrsMatrix> inputMatrix, 
                         RCP<const xMap> outputXMap,
                         RCP<const tMap> outputTMap,
                         RCP<Tpetra::Distributor> & distributor,
                         size_t &numTerms, size_t & locCount,
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
                         std::vector<LO> & columnsRecv)
{
    // ownedRowGIDs = rows of inputMatrix contained in outputRowMap
    // localRowsSend[indicesSend] = localRows of sending process i corresponding to
    //                              localRowsRecv[indicesRecv], where
    //               indicesSend = localRowsSendBegin[i]:localRowsSendBegin[i+1]-1
    //               indicesRecv = localRowsRecvBegin[i]:localRowsRecvBegin[i+1]-1
    constructDistributor(inputMatrix, outputTMap, distributor, ownedRowGIDs,
                         localRowsSend, localRowsSendBegin,
                         localRowsRecv, localRowsRecvBegin);
    // targetMapGIDs[indices] = globalIDs of outputRowMap for the i'th process
    //                          receiving matrix data
    //         indices = targetMapGIDsBegin[i]:targetMapGIDsBegin[i+1]-1;
    // Note: length of targetMapGIDsBegin is the number of processes receiving 
    //       matrix data plus 1
    communicateRowMap(outputTMap, distributor, targetMapGIDs, targetMapGIDsBegin);
    {
        auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_build");
        Teuchos::TimeMonitor CaseTimer( *caseTimer );

        return
        communicateMatrixData_build(inputMatrix, outputXMap, outputTMap, distributor, targetMapGIDs, 
                                    targetMapGIDsBegin, ownedRowGIDs,
                                    localRowsSend, localRowsSendBegin, localRowsRecv,
                                    localRowsRecvBegin, numTerms, locCount, sourceSize, targetSize,rowCount,
                                    columnsRecv);
    }
}


template <class SC, class LO, class GO, class NO>
void TpetraFunctions<SC,LO,GO,NO>::
importSquareMatrix_import(RCP<const tCrsMatrix> inputMatrix,
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
                          ValuesViewT_dev  &valuesRecv_d,
                          PointerViewT_dev &mapRecvToLocal,
                          RCP<tCrsMatrix> & outputMatrix,
                          bool replaceVals)
{
    auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: importMatrixData");
    Teuchos::TimeMonitor CaseTimer( *caseTimer );

    communicateMatrixData_import(inputMatrix, outputRowMap, distributor, 
                                 numTerms, locCount, sourceSize, targetSize, rowCount,
                                 targetMapGIDs, targetMapGIDsBegin, ownedRowGIDs,
                                 localRowsSend, localRowsSendBegin, localRowsRecv,
                                 localRowsRecvBegin, columnsRecv, valuesRecv_d, mapRecvToLocal,
                                 outputMatrix, replaceVals);
}

template <class SC, class LO, class GO, class NO>
void TpetraFunctions<SC,LO,GO,NO>::
extractMapRecvToLocal(Teuchos::RCP<Tpetra::Distributor> distributor,
                      const size_t numTerms,
                      const Teuchos::ArrayRCP<size_t> & rowCount,
                      const std::vector<LO> & localRowsRecv,
                      const std::vector<LO> & localRowsRecvBegin,
                      const std::vector<LO> & columnsRecv,
                      RCP<tCrsMatrix> & outputMatrix,
                      PointerViewT_dev &mapRecvToLocal)
{
    auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: extractMapRecvToLocal");
    Teuchos::TimeMonitor CaseTimer( *caseTimer );

    auto localOutputPtrs_h = outputMatrix->getLocalRowPtrsHost();
    auto localOutputInds_h = outputMatrix->getLocalIndicesHost();

    //printf( " extractMapRecvToLocal(%d -> %d : %d,%d)\n",mapRecvToLocal.extent(0),numTerms, localOutputPtrs_h.extent(0),localOutputInds_h.extent(0) );
    const size_t numRecvs = distributor->getNumReceives();
    Kokkos::resize(mapRecvToLocal, numTerms);
    auto mapRecvToLocal_h = Kokkos::create_mirror_view(mapRecvToLocal);
    size_t countTerms = 0;
    for (LO j=0; j<localRowsRecvBegin[numRecvs]; j++) {
        const LO localRow = localRowsRecv[j];
        for (size_t i=0; i<rowCount[localRow]; i++) {
            for (size_t k=localOutputPtrs_h(localRow); k<localOutputPtrs_h(localRow+1); k++) {
                if (localOutputInds_h(k) == columnsRecv[countTerms]) {
                    mapRecvToLocal_h(countTerms) = k;
                    countTerms++;
                    break;
                }
            }
        }
    }
    //printf( " extractMapRecvToLocal(%d vs %d)\n",numTerms,countTerms );
    Kokkos::deep_copy(mapRecvToLocal, mapRecvToLocal_h);
}
// ------------------------------------------------------------------------------- //


// ------------------------------------------------------------------------------- //
template <class SC, class LO, class GO, class NO>
Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>>
TpetraFunctions<SC,LO,GO,NO>::
communicateMatrixData_build(RCP<const tCrsMatrix> inputMatrix, 
                      RCP<const xMap> rowXMap,
                      RCP<const tMap> rowMap,
                      RCP<Tpetra::Distributor> distributor,
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
                      std::vector<LO> & columnsRecv)
{
    const size_t numSends = distributor->getNumSends();
    const size_t numRecvs = distributor->getNumReceives();
    TEUCHOS_TEST_FOR_EXCEPTION(numSends != localRowsSendBegin.size()-1,std::runtime_error,
                    "invalid size of localRowsSendBegin");
    TEUCHOS_TEST_FOR_EXCEPTION(numRecvs != localRowsRecvBegin.size()-1,std::runtime_error,
                    "invalid size of localRowsRecvBegin");

    const size_t numRowsSendTotal = localRowsSend.size();
    std::vector<size_t> count(numRowsSendTotal, 0);
    IndicesViewT indices;
    ValuesViewT values;
    RCP<const tMap> columnMap = inputMatrix->getColMap();
    auto IGO = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto ILO = Teuchos::OrdinalTraits<LO>::invalid();
    RCP<const Teuchos::Comm<int> > serialComm = RCP( new Teuchos::SerialComm<int>() );
    for (size_t i=0; i<numSends; i++) {
        const GO* targetGIDs = &targetMapGIDs[targetMapGIDsBegin[i]];
        const size_t numTargetGIDs = targetMapGIDsBegin[i+1] - targetMapGIDsBegin[i];
        RCP<const tMap> targetMap = 
            RCP( new tMap(IGO, Teuchos::ArrayView<const GO>(targetGIDs, numTargetGIDs),
                          0, serialComm) );
        for (LO j=localRowsSendBegin[i]; j<localRowsSendBegin[i+1]; j++) {
            const LO localRow = localRowsSend[j];
            inputMatrix->getLocalRowView(localRow, indices, values);
            const int sz = indices.size();
            for (int k=0; k<sz; k++) {
                const GO colGlobalID = columnMap->getGlobalElement(indices[k]);
                LO targetIndex = targetMap->getLocalElement(colGlobalID);
                if (targetIndex != ILO) count[j]++;
            }
        }
    }

    locCount = 0;
    for (size_t i=0; i<count.size(); i++) locCount += count[i];
    std::vector<LO> localColIDs(locCount);
    size_t sumCount = 0;
    for (size_t i=0; i<numSends; i++) {
        const GO* targetGIDs = &targetMapGIDs[targetMapGIDsBegin[i]];
        const size_t numTargetGIDs = targetMapGIDsBegin[i+1] - targetMapGIDsBegin[i];
        RCP<const tMap> targetMap = 
            RCP( new tMap(IGO, Teuchos::ArrayView<const GO>(targetGIDs, numTargetGIDs),
                          0, serialComm) );
        for (LO j=localRowsSendBegin[i]; j<localRowsSendBegin[i+1]; j++) {
            const LO localRow = localRowsSend[j];
            inputMatrix->getLocalRowView(localRow, indices, values);
            const int sz = indices.size();
            for (int k=0; k<sz; k++) {
                const GO colGlobalID = columnMap->getGlobalElement(indices[k]);
                LO targetIndex = targetMap->getLocalElement(colGlobalID);
                if (targetIndex != ILO) {
                    localColIDs[sumCount++] = targetIndex;
                }
            }
        }
    }
    const size_t numRowsRecv = localRowsRecvBegin[numRecvs];
    std::vector<size_t> recvSize(numRecvs);
    std::vector<size_t> recvRowNumNonzeros(numRowsRecv);
    for (size_t i=0; i<numRecvs; i++)  {
        recvSize[i] = localRowsRecvBegin[i+1] - localRowsRecvBegin[i];
    }
    std::vector<size_t> sendSize(numSends);
    for (size_t i=0; i<numSends; i++) {
        sendSize[i] = localRowsSendBegin[i+1] - localRowsSendBegin[i];
    }
    distributor->doPostsAndWaits(Kokkos::View<const size_t*, Kokkos::HostSpace>(count.data(), count.size()),
                                 Teuchos::ArrayView<const size_t>(sendSize),
                                 Kokkos::View<size_t*, Kokkos::HostSpace>(recvRowNumNonzeros.data(), recvRowNumNonzeros.size()),
                                 Teuchos::ArrayView<const size_t>(recvSize));

    /*
      const int myPID = rowMap->getComm()->getRank();
      Teuchos::ArrayView<const int> procsFrom = distributor->getProcsFrom();
      for (size_t i=0; i<numRecvs; i++) {
      std::cout << "row globalIDs and counts for proc " << myPID << " received from proc "
      << procsFrom[i] << std::endl;
      for (LO j=localRowsRecvBegin[i]; j<localRowsRecvBegin[i+1]; j++) {
      std::cout << rowMap->getGlobalElement(localRowsRecv[j]) << " "
      << recvRowNumNonzeros[j] << std::endl;
      }
      }
    */ 
    sourceSize.resize(numSends);
    for (size_t i=0; i<numSends; i++) {
        size_t procCount(0);
        for (LO j=localRowsSendBegin[i]; j<localRowsSendBegin[i+1]; j++) {
            procCount += count[j];
        }
        sourceSize[i] = procCount;
    }
    targetSize.resize(numRecvs);
    numTerms = 0;
    for (size_t i=0; i<numRecvs; i++) {
        size_t procCount(0);
        for (LO j=localRowsRecvBegin[i]; j<localRowsRecvBegin[i+1]; j++) {
            procCount += recvRowNumNonzeros[j];
        }
        targetSize[i] = procCount;
        numTerms += procCount;
    }

    columnsRecv.resize(numTerms);
#if 0
    if (columnMap->getComm()->getRank() == 2) {
        printf(" + localRowsRecv[i]=[\n");
        for (LO i=0; i<localRowsRecvBegin[numRecvs]; i++) printf("%d %d\n",i,localRowsRecv[i]);
        printf("];\n");
        printf( " %d %d\n",locCount,numTerms );
        printf(" + source=[\n");
        for (int i=0; i<sourceSize.size(); i++) printf( "%d %d\n",i,sourceSize[i]);
        printf("];\n");
        printf(" + target=[\n");
        for (int i=0; i<targetSize.size(); i++) printf( "%d %d\n",i,targetSize[i]);
        printf("];\n");
        printf(" + locCol[\n");
        for (int i=0; i<localColIDs.size(); i++) printf( "%d %d\n",i,localColIDs[i]);
        printf("];\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    distributor->doPostsAndWaits(Kokkos::View<const LO*, Kokkos::HostSpace>(localColIDs.data(), localColIDs.size()),
                                 Teuchos::ArrayView<const size_t>(sourceSize),
                                 Kokkos::View<LO*, Kokkos::HostSpace>(columnsRecv.data(), columnsRecv.size()),
                                 Teuchos::ArrayView<const size_t>(targetSize));
    
    RCP<const tMap> rowMap1to1 = inputMatrix->getRowMap();
    const size_t numRows = rowMap->getLocalNumElements();
    rowCount.resize(numRows);
    // rowCounts for owned rows
    RCP<const tMap> inputColMap = inputMatrix->getColMap();
    for (size_t i=0; i<ownedRowGIDs.size(); i++) {
        const LO localRowSource = rowMap1to1->getLocalElement(ownedRowGIDs[i]);
        const LO localRowTarget = rowMap->getLocalElement(ownedRowGIDs[i]);
        TEUCHOS_TEST_FOR_EXCEPTION(localRowSource == ILO, std::runtime_error,"globalID not found in rowMap1to1");
        TEUCHOS_TEST_FOR_EXCEPTION(localRowTarget == ILO, std::runtime_error,"globalID not found in rowMap");
        inputMatrix->getLocalRowView(localRowSource, indices, values);
        const int sz = indices.size();
        for (int j=0; j<sz; j++) {
            const GO colGID = inputColMap->getGlobalElement(indices[j]);
            LO localColTarget = rowMap->getLocalElement(colGID);
            if (localColTarget != ILO) rowCount[localRowTarget]++;
        }
    }
    // rowCounts for received rows
    for (LO i=0; i<localRowsRecvBegin[numRecvs]; i++) {
        const LO localRow = localRowsRecv[i];
        rowCount[localRow] = recvRowNumNonzeros[i];
    }
    //static RCP<Matrix> Build(const RCP<const Map>& rowMap, const RCP<const Map>& colMap, const ArrayRCP<const size_t>& NumEntriesPerRowToAlloc)
    // argument types are: (Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>, Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>, Teuchos::ArrayView<const unsigned long>)
    //return MatrixFactory<SC, LO, GO, NO>::Build(rowXMap, rowXMap, Teuchos::ArrayView<const size_t>(rowCount));
    return Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(rowXMap, rowXMap, rowCount);
}

template <class SC, class LO, class GO, class NO>
void TpetraFunctions<SC,LO,GO,NO>::
communicateMatrixData_import(RCP<const tCrsMatrix> inputMatrix, 
                      RCP<const tMap> outputRowMap, // TODO: outputMatrix->getRowMap() ?
                      RCP<Tpetra::Distributor> distributor,
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
                      ValuesViewT_dev  &valuesRecv_d,
                      PointerViewT_dev &mapRecvToLocal,
                      RCP<tCrsMatrix> & outputMatrix,
                      bool replaceVals)
{
    auto IGO = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    auto ILO = Teuchos::OrdinalTraits<LO>::invalid();
    RCP<const Teuchos::Comm<int> > serialComm = RCP( new Teuchos::SerialComm<int>() );

    size_t numRows = outputRowMap->getLocalNumElements();
    RCP<const tMap> inputRowMap = inputMatrix->getRowMap();
    RCP<const tMap> inputColMap = inputMatrix->getColMap();

    // Prepare : Pack non-zero values to send
    const size_t numSends = distributor->getNumSends();
    const size_t numRecvs = distributor->getNumReceives();
    std::vector<SC> matrixValues(locCount);
    size_t sumCount = 0;
    IndicesViewT indices;
    ValuesViewT values;
    {
      auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (prep)");
      Teuchos::TimeMonitor CaseTimer( *caseTimer );
      for (size_t i=0; i<numSends; i++) {
          const GO* targetGIDs = &targetMapGIDs[targetMapGIDsBegin[i]];
          const size_t numTargetGIDs = targetMapGIDsBegin[i+1] - targetMapGIDsBegin[i];
          RCP<const tMap> targetMap = 
              RCP( new tMap(IGO, Teuchos::ArrayView<const GO>(targetGIDs, numTargetGIDs),
                            0, serialComm) );
          for (LO j=localRowsSendBegin[i]; j<localRowsSendBegin[i+1]; j++) {
              const LO localRow = localRowsSend[j];
              inputMatrix->getLocalRowView(localRow, indices, values);
              const int sz = indices.size();
              for (int k=0; k<sz; k++) {
                  const GO colGlobalID = inputColMap->getGlobalElement(indices[k]);
                  LO targetIndex = targetMap->getLocalElement(colGlobalID);
                  if (targetIndex != ILO) {
                      matrixValues[sumCount++] = values[k];
                  }
              }
          }
      }
      // resized for the first call (in symbolic)
      Kokkos::resize(valuesRecv_d, numTerms);
    }
    //MPI_Barrier(MPI_COMM_WORLD); printf( " >> communicateMatrixData_import(1) <<\n" ); fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);

    // Communicate
    // MPI through host CPU, create buffer on host
    using KSX = typename Kokkos::ArithTraits<SC>::val_type;
    auto valuesRecv = Kokkos::create_mirror_view(valuesRecv_d);
    const KSX* matrixValues_K = reinterpret_cast<const KSX*>(matrixValues.data());
    KSX* valuesRecv_K = reinterpret_cast<KSX*>(valuesRecv.data());
    const size_t sizeSend = matrixValues.size();
    const size_t sizeRecv = valuesRecv.extent(0);
#if 1
    {
        int numProcs = inputRowMap->getComm()->getSize();
        int sendcounts[  numProcs];
        int senddispls[1+numProcs];
        int recvcounts[  numProcs];
        int recvdispls[1+numProcs];
        {
            auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (communicate:alltoallv:prep)");
            Teuchos::TimeMonitor CaseTimer( *caseTimer );
            senddispls[0] = 0;
            recvdispls[0] = 0;
            for (int i=0; i<numProcs; i++) {
                sendcounts[i] = 0;
                recvcounts[i] = 0;
            }
            auto procsFrom = distributor->getProcsFrom();
            for (size_t i=0; i<distributor->getNumReceives(); i++) {
                recvcounts[procsFrom[i]] = targetSize[i];
            }
            auto procsTo = distributor->getProcsTo();
            for (size_t i=0; i<distributor->getNumSends(); i++) {
                sendcounts[procsTo[i]] = sourceSize[i];
            }
            for (int i=0; i<numProcs; i++) {
                senddispls[i+1] = senddispls[i]+sendcounts[i];
                recvdispls[i+1] = recvdispls[i]+recvcounts[i];
                //printf( " %d/%d: sendcount[%d/%d] = %d, recvcount[%d/%d] = %d\n",inputRowMap->getComm()->getRank(),inputRowMap->getComm()->getSize(), i,numProcs,sendcounts[i], i,numProcs,recvcounts[i] );
            }
            //fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);
        }
        {
            auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (communicate:alltoallv:mpi)");
            Teuchos::TimeMonitor CaseTimer( *caseTimer );
            MPI_Alltoallv(matrixValues_K, sendcounts, senddispls, MPI_DOUBLE,
                          valuesRecv_K, recvcounts, recvdispls, MPI_DOUBLE,
                          Teuchos::getRawMpiComm(*(inputRowMap->getComm())));
        }
    }
#else
    {
        auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (communicate:doPostAndWait)");
        Teuchos::TimeMonitor CaseTimer( *caseTimer );
        distributor->doPostsAndWaits(Kokkos::View<const KSX*, Kokkos::HostSpace>(matrixValues_K, sizeSend),
                                     Teuchos::ArrayView<const size_t>(sourceSize),
                                     Kokkos::View<KSX*, Kokkos::HostSpace>(valuesRecv_K, sizeRecv),
                                     Teuchos::ArrayView<const size_t>(targetSize));
    }
#endif
    //MPI_Barrier(MPI_COMM_WORLD); printf( " >> communicateMatrixData_import(2) <<\n" ); fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);

    // Insert numerical values into local matrix
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (replaceVals) {
        auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (replace)");
        Teuchos::TimeMonitor CaseTimer( *caseTimer );

        // --------------------------------- //
        // insert ** OWNED ** constributions //
        // --------------------------------- //
        std::vector<LO> indicesVec(numRows);
        std::vector<SC> valuesVec(numRows);
        {
            //#define REPLACE_LOCAL_MATRIX
            #define REPLACE_LOCAL_MATRIX_ON_DEVICE
            #ifdef REPLACE_LOCAL_MATRIX
            //auto localOutputMatrix  = outputMatrix->getLocalMatrixHost();  // target
            auto localOutputPtrs = outputMatrix->getLocalRowPtrsHost();
            auto localOutputInds = outputMatrix->getLocalIndicesHost();
            auto localOutputVals = outputMatrix->getLocalValuesHost(Tpetra::Access::OverwriteAll /*Tpetra::Access::ReadWrite*/);
            #elif defined(REPLACE_LOCAL_MATRIX_ON_DEVICE)
            //auto localOutputPtrs = outputMatrix->getLocalRowPtrsHost();
            auto localOutputPtrs_dev = outputMatrix->getLocalRowPtrsDevice();
            auto localOutputInds_dev = outputMatrix->getLocalIndicesDevice();
            auto localOutputVals_dev = outputMatrix->getLocalValuesDevice(Tpetra::Access::OverwriteAll /*Tpetra::Access::ReadWrite*/);
            auto localInputPtrs_dev = inputMatrix->getLocalRowPtrsDevice();
            auto localInputInds_dev = inputMatrix->getLocalIndicesDevice();
            auto localInputVals_dev = inputMatrix->getLocalValuesDevice(Tpetra::Access::ReadOnly);
            #endif
            #if defined(REPLACE_LOCAL_MATRIX_ON_DEVICE)
            auto localInputRowMap  = inputRowMap->getLocalMap();
            auto localInputColMap  = inputColMap->getLocalMap();
            auto localOutputRowMap = outputRowMap->getLocalMap();
            //auto localInputMatrix  = inputMatrix->getLocalMatrixHost();  // source
            #else
            auto localInputMatrix  = inputMatrix->getLocalMatrixHost();  // source
            outputMatrix->resumeFill();
            #endif
            LO numRows = inputMatrix->getLocalNumRows();
            //printf( " %d: (nRows=%d,%d): non-overlap localInputMat(%d, %d,%d) %s, overlap localOutputMat (%d, %d,%d), (%d, %d,%d) %s\n",inputColMap->getComm()->getRank(),numRows,outputMatrix->getLocalNumRows(),
            //                localInputMatrix.graph.row_map.extent(0), localInputMatrix.graph.entries.extent(0),localInputMatrix.values.extent(0),(inputMatrix->isLocallyIndexed() ? "local" : "global"),
            //                localOutputPtrs.extent(0),localOutputInds.extent(0),localOutputVals.extent(0),
            //                localOutputMatrix.graph.row_map.extent(0), localOutputMatrix.graph.entries.extent(0),localOutputMatrix.values.extent(0),(outputMatrix->isLocallyIndexed() ? "local" : "global"));

#if defined(REPLACE_LOCAL_MATRIX_ON_DEVICE)
            using execution_space = typename tMap::local_map_type::execution_space;
            using local_map_device_type = typename tCrsMatrix::map_type::local_map_type;
            using local_ptrs_device_type = typename tCrsMatrix::row_ptrs_device_view_type;
            using local_inds_device_type = typename tCrsMatrix::local_inds_device_view_type;
            using local_vals_device_type = typename tCrsMatrix::local_matrix_device_type::values_type;
            using local_ptrs_device_const = typename local_ptrs_device_type::const_type;
            using local_inds_device_const = typename local_inds_device_type::const_type;
            using local_vals_device_const = typename local_vals_device_type::const_type;
            {
                auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (replaceLocalVals owned)");
                Teuchos::TimeMonitor CaseTimer( *caseTimer );
                TpetraFunctor_insert_with_map<local_map_device_type, local_ptrs_device_const, local_inds_device_const, local_vals_device_const,
                                                                     local_ptrs_device_type,  local_inds_device_type,  local_vals_device_type>
                        tpetra_functor(localInputRowMap, localInputColMap, localOutputRowMap,
                                       localInputPtrs_dev, localInputInds_dev, localInputVals_dev,
                                       localOutputPtrs_dev, localOutputInds_dev, localOutputVals_dev);
                Kokkos::RangePolicy<execution_space> policy_replace (0, numRows);
                Kokkos::parallel_for(
                    "FROSch_GDSW_Proxy::replace", policy_replace, tpetra_functor);
                Kokkos::fence();
            }
#else
            //MPI_Barrier(MPI_COMM_WORLD); printf( " >> communicateMatrixData_import(3) <<\n" ); fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);
            for(LO i = 0; i < numRows; i ++) {
                auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (replaceLocalVals owned)");
                Teuchos::TimeMonitor CaseTimer( *caseTimer );
                GO globalIndex_i = inputRowMap->getGlobalElement(i);
                LO localRowTarget = outputRowMap->getLocalElement(globalIndex_i);
                if (localRowTarget == ILO) {
                    printf( " %d: localRowTarget = %d\n",inputRowMap->getComm()->getRank(),localRowTarget );
                    break;
                }

                #ifndef REPLACE_LOCAL_MATRIX
                int sz = 0;
                indicesVec.resize(rowCount[localRowTarget]);
                valuesVec.resize(rowCount[localRowTarget]);
                #endif
                for (size_t j=localInputMatrix.graph.row_map(i); j<localInputMatrix.graph.row_map(i+1); j++) {
                    LO colIndex_In = localInputMatrix.graph.entries(j);
                    GO globalIndex_j = inputColMap->getGlobalElement(colIndex_In);
                    LO localIndex_j  = outputRowMap->getLocalElement(globalIndex_j);
                    if (localIndex_j != ILO) {
                        #ifdef REPLACE_LOCAL_MATRIX
                        // look for the same column index in localSubdomainMatrix
                        for (LO k=localOutputPtrs(localRowTarget); k<localOutputPtrs(localRowTarget+1); k++) 
                        {
                            if (localOutputInds(k) == localIndex_j) 
                            {
                                localOutputVals[k] = localInputMatrix.values[j];
                                break;
                            }
                        }
                        #else
                        indicesVec[sz] = localIndex_j;
                        valuesVec[sz] = localInputMatrix.values[j];
                        sz++;
                        #endif
                    }
                }
                #ifndef REPLACE_LOCAL_MATRIX
                {
                    //if (inputRowMap->getComm()->getRank() == 0) printf( " replace (%d,%d,%d) : %d,%d\n",i,globalIndex_i,localRowTarget, sz,rowCount[localRowTarget] );
                    outputMatrix->replaceLocalValues(localRowTarget, 
                                                     Teuchos::ArrayView<const LO>(indicesVec),
                                                     Teuchos::ArrayView<const SC>(valuesVec));
                }
                #endif
            }
#endif
        }
        //MPI_Barrier(MPI_COMM_WORLD); printf( " >> communicateMatrixData_import(4) <<\n" ); fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);
        // ----------------------------------- //
        // insert ** RECEIVED ** contributions //
        // ----------------------------------- //
        if (true) {
            auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (replaceLocalVals_localView received)");
            Teuchos::TimeMonitor CaseTimer( *caseTimer );
            if (mapRecvToLocal.extent(0) < numTerms) {
                #if 1
                extractMapRecvToLocal(distributor, numTerms, rowCount,
                                      localRowsRecv, localRowsRecvBegin, columnsRecv,
                                      outputMatrix, mapRecvToLocal);
                #else
                auto localOutputPtrs_h = outputMatrix->getLocalRowPtrsHost();
                auto localOutputInds_h = outputMatrix->getLocalIndicesHost();
                // create a map from valuesRecv to localOutputVals
                size_t countTerms = 0;
                Kokkos::resize(mapRecvToLocal, numTerms);
                auto mapRecvToLocal_h = Kokkos::create_mirror_view(mapRecvToLocal);
                for (LO j=0; j<localRowsRecvBegin[numRecvs]; j++) {
                    const LO localRow = localRowsRecv[j];
                    //printf( " %d: localRowsRecv[%d] = %d\n",myRank,j,localRow );
                    for (size_t i=0; i<rowCount[localRow]; i++) {
                        for (size_t k=localOutputPtrs_h(localRow); k<localOutputPtrs_h(localRow+1); k++) {
                            if (localOutputInds_h(k) == columnsRecv[countTerms]) {
                                mapRecvToLocal_h(countTerms) = k;
                                countTerms++;
                                break;
                            }
                        }
                    }
                }
                Kokkos::deep_copy(mapRecvToLocal, mapRecvToLocal_h);
                #endif
            }
            // copy received non-zero values to local matrix
            #if 1
             #if 1
             {
                 Kokkos::deep_copy(valuesRecv_d, valuesRecv);
                 auto localOutputVals_d = outputMatrix->getLocalValuesDevice(Tpetra::Access::OverwriteAll);

                 using execution_space = typename tMap::local_map_type::execution_space;
                 Kokkos::RangePolicy<execution_space> policy_row (0, valuesRecv.extent(0));
                 TpetraFunctor_insert tpetra_functor(mapRecvToLocal, valuesRecv_d, localOutputVals_d);
                 Kokkos::RangePolicy<execution_space> policy_replace (0, valuesRecv.extent(0));
                 Kokkos::parallel_for(
                     "FROSch::communicateMatrixData_imports::readMap", policy_replace, tpetra_functor);
                 Kokkos::fence();
             }
             #else
             {
                 auto localOutputVals_h = outputMatrix->getLocalValuesHost(Tpetra::Access::OverwriteAll);
                 auto mapRecvToLocal_h = Kokkos::create_mirror_view(mapRecvToLocal);
                 for (size_t i=0; i<valuesRecv.extent(0); i++) {
                     localOutputVals_h(mapRecvToLocal_h(i)) = valuesRecv(i);
                 }
             }
             #endif
            #else
             countTerms = 0;
             for (LO j=0; j<localRowsRecvBegin[numRecvs]; j++) {
                 const LO localRow = localRowsRecv[j];
                 //printf( " %d: localRowsRecv[%d] = %d\n",myRank,j,localRow );
                 for (size_t i=0; i<rowCount[localRow]; i++) {
                     for (size_t k=localOutputPtrs_h(localRow); k<localOutputPtrs_h(localRow+1); k++) {
                         if (localOutputInds_h(k) == columnsRecv[countTerms]) {
                             localOutputVals_h(k) = valuesRecv(countTerms);
                             countTerms++;
                             break;
                         }
                     }
                 }
             }
            #endif
        } else
        {
            auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (replaceLocalVals received)");
            Teuchos::TimeMonitor CaseTimer( *caseTimer );
            size_t countTerms = 0;
            for (LO j=0; j<localRowsRecvBegin[numRecvs]; j++) {
                const LO localRow = localRowsRecv[j];
                indicesVec.resize(rowCount[localRow]);
                valuesVec.resize(rowCount[localRow]);
                for (size_t k=0; k<rowCount[localRow]; k++) {
                    indicesVec[k] = columnsRecv[countTerms];
                    valuesVec[k] = valuesRecv(countTerms++);
                }
                outputMatrix->replaceLocalValues(localRow, 
                                                 Teuchos::ArrayView<const LO>(indicesVec),
                                                 Teuchos::ArrayView<const SC>(valuesVec));
            }
        }
        //if (myRank == 1)
        //{ printf( " >> outputMatrix <<\n" ); RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(std::cout)); outputMatrix->describe(*fancy,VERB_EXTREME); }
        /*{
            printf( " >> outputMatrix(%d) <<\n",myRank );
            typedef Tpetra::MatrixMarket::Writer<tCrsMatrix> writer_type;
            char filename[200];
            sprintf(filename,"A%d.dat",myRank);
            writer_type::writeSparseFile (filename, outputMatrix);
            printf( " >> outputMatrix(%d) done <<\n",myRank );
            MPI_Barrier(MPI_COMM_WORLD); exit(0);
        }*/
    } else {
        // First time inserting column indices (and numberical values), (from symbolic)
        auto caseTimer = Teuchos::TimeMonitor::getNewTimer("GDSW test: communicateMatrixData_import (insert)");
        Teuchos::TimeMonitor CaseTimer( *caseTimer );

        std::vector<GO> indicesVec(numRows);
        std::vector<SC> valuesVec(numRows);
        // insert owned constributions
        for (size_t i=0; i<ownedRowGIDs.size(); i++) {
            indicesVec.resize(0);
            valuesVec.resize(0);
            const GO globalRow = ownedRowGIDs[i];
            const LO localRowSource = inputRowMap->getLocalElement(globalRow);
            //const LO localRowTarget = outputRowMap->getLocalElement(globalRow);
            //if (inputColMap->getComm()->getRank() == 2) printf( " localRow : %d -> %d,%d\n",globalRow,localRowSource,localRowTarget);
            inputMatrix->getLocalRowView(localRowSource, indices, values);
            const int sz = indices.size();
            for (int j=0; j<sz; j++) {
                const GO colGID = inputColMap->getGlobalElement(indices[j]);
                LO localColTarget = outputRowMap->getLocalElement(colGID);
                //if (inputColMap->getComm()->getRank() == 2) printf( " + %d(%d) %e\n",colGID,localColTarget,values[j] );
                if (localColTarget != ILO) {
                    indicesVec.push_back(colGID);
                    valuesVec.push_back(values[j]);
                }
            }
            /*{
                GlobalIndicesViewT gIndices;
                outputMatrix->getGlobalRowView(globalRow, gIndices, values);
                if (inputColMap->getComm()->getRank() == 2) {
                    printf( " > current size = %d\n",gIndices.size() );
                    for (int j=0; j<gIndices.size(); j++) printf( " current %d: %d, %e\n",j,gIndices[j],values[j] );
                }
            }*/
            //if (inputColMap->getComm()->getRank() == 2) printf( " ==> %d(%d): %d\n",globalRow,localRowTarget,indicesVec.size()); fflush(stdout);
            outputMatrix->insertGlobalValues(globalRow, 
                                             Teuchos::ArrayView<const GO>(indicesVec),
                                             Teuchos::ArrayView<const SC>(valuesVec));
        }
        // insert received contributions
        size_t countTerms = 0;
        for (LO j=0; j<localRowsRecvBegin[numRecvs]; j++) {
            const LO localRow = localRowsRecv[j];
            const GO globalRow = outputRowMap->getGlobalElement(localRow);
            indicesVec.resize(rowCount[localRow]);
            valuesVec.resize(rowCount[localRow]);
            for (size_t k=0; k<rowCount[localRow]; k++) {
                //if (inputColMap->getComm()->getRank() == 2) printf( " * %d(%d) %e\n",outputRowMap->getGlobalElement(columnsRecv[countTerms]),valuesRecv(countTerms) );
                indicesVec[k] = outputRowMap->getGlobalElement(columnsRecv[countTerms]);
                valuesVec[k] = valuesRecv(countTerms++);
            }
            /*{
                GlobalIndicesViewT gIndices;
                outputMatrix->getGlobalRowView(globalRow, gIndices, values);
                if (inputColMap->getComm()->getRank() == 2) {
                    printf( " > current size = %d\n",gIndices.size() );
                    for (int j=0; j<gIndices.size(); j++) printf( " current %d: %d, %e\n",j,gIndices[j],values[j] );
                }
            }*/
            //if (inputColMap->getComm()->getRank() == 2) printf( " ==> %d(%d): %d\n",globalRow,localRow,indicesVec.size()); fflush(stdout);
            outputMatrix->insertGlobalValues(globalRow,
                                             Teuchos::ArrayView<const GO>(indicesVec),
                                             Teuchos::ArrayView<const SC>(valuesVec));
        }
        if (false) {
            outputMatrix->fillComplete();
            // create a map from valuesRecv to localOutputVals
            auto localOutputPtrs_h = outputMatrix->getLocalRowPtrsHost();
            auto localOutputInds_h = outputMatrix->getLocalIndicesHost();

            Kokkos::resize(mapRecvToLocal, numTerms);
            auto mapRecvToLocal_h = Kokkos::create_mirror_view(mapRecvToLocal);
            size_t countTerms = 0;
            for (LO j=0; j<localRowsRecvBegin[numRecvs]; j++) {
                const LO localRow = localRowsRecv[j];
                //printf( " %d: localRowsRecv[%d] = %d\n",myRank,j,localRow );
                for (size_t i=0; i<rowCount[localRow]; i++) {
                    for (size_t k=localOutputPtrs_h(localRow); k<localOutputPtrs_h(localRow+1); k++) {
                        if (localOutputInds_h(k) == columnsRecv[countTerms]) {
                            mapRecvToLocal_h(countTerms) = k;
                            countTerms++;
                            break;
                        }
                    }
                }
            }
            Kokkos::deep_copy(mapRecvToLocal, mapRecvToLocal_h);
            //outputMatrix->resumeFill();
        } else {
            Kokkos::resize(mapRecvToLocal, 0);
        }
    }
    //if (outputMatrix->isLocallyIndexed ()) printf( " isLocal(b)\n" );
    //else printf( " not isLocal(b)\n" ); fflush(stdout);

    // TODO : when need to fillComplete?
    // This is the overlapping "global" matrix (not local sequential one)
    //RCP<ParameterList> fillCompleteParams(new ParameterList);
    //fillCompleteParams->set("No Nonlocal Changes", true);
    //outputMatrix->fillComplete(rowMap1to1, rowMap1to1, fillCompleteParams);
    //outputMatrix->fillComplete(rowMap1to1, rowMap1to1);
}
// ------------------------------------------------------------------------------- //



/****************************************************************************************************/
/****************************************************************************************************/
/****************************************************************************************************/

template <class SC, class LO, class GO, class NO>
void TpetraFunctions<SC,LO,GO,NO>::
communicateRowMap(RCP<const tMap> rowMap, 
                  RCP<Tpetra::Distributor> distributor, 
                  std::vector<GO> & rowMapGIDs, 
                  std::vector<LO> & rowMapGIDsBegin)
{
    size_t numSends = distributor->getNumSends();
    size_t numRecvs = distributor->getNumReceives();
    size_t numRows = rowMap->getLocalNumElements();
    std::vector<size_t> targetValues(numSends);
    std::vector<size_t> targetSizes(numSends, 1);
    std::vector<size_t> sourceValues(numRecvs, numRows);
    std::vector<size_t> sourceSizes(numRecvs, 1); 

    // CMS: Reverse sends the # of rows on each proc to all neighbors (I think)
    distributor->doReversePostsAndWaits(Kokkos::View<const size_t*, Kokkos::HostSpace>(sourceValues.data(), sourceValues.size()),
                                        1,
                                        Kokkos::View<size_t*, Kokkos::HostSpace>(targetValues.data(), targetValues.size()));
    // CMS: Compute the total number of rows on reverse neighbors of this rank
    int numTerms(0);
    for (size_t i=0; i<targetValues.size(); i++) numTerms += targetValues[i];
    rowMapGIDs.resize(numTerms);

    // CMS: For each recv, record each of my GIDs
    std::vector<GO> globalIDsSource(numRecvs*numRows);
    numTerms = 0;
    for (size_t i=0; i<numRecvs; i++) {
        for (size_t j=0; j<numRows; j++) {
            globalIDsSource[numTerms++] = rowMap->getGlobalElement(j);
        }
    }

    // CMS: Reverse all of the GIDs owned by the neighboring proc
    distributor->doReversePostsAndWaits(Kokkos::View<const GO*, Kokkos::HostSpace>(globalIDsSource.data(), globalIDsSource.size()),
                                        Teuchos::ArrayView<const size_t>(sourceValues),
                                        Kokkos::View<GO*, Kokkos::HostSpace>(rowMapGIDs.data(), rowMapGIDs.size()),
                                        Teuchos::ArrayView<const size_t>(targetValues));

    // CMS: Note the row beginnings of each reverse neighboring row
    rowMapGIDsBegin.resize(numSends+1, 0);
    for (size_t i=0; i<numSends; i++) {
        rowMapGIDsBegin[i+1] = rowMapGIDsBegin[i] + targetValues[i];
    }
    /*
      const int myPID = rowMap->getComm()->getRank();
      std::cout << "map globalIDs for myPID = " << myPID << std::endl;
      for (size_t i=0; i<numSends; i++) {
      for (LO j=rowMapGIDsBegin[i]; j<rowMapGIDsBegin[i+1]; j++) {
      std::cout << rowMapGIDs[j] << " ";
      }
      std::cout << std::endl;
      }
    */
}





template <class SC, class LO, class GO, class NO>
void TpetraFunctions<SC,LO,GO,NO>::
constructDistributor(RCP<const tCrsMatrix> inputMatrix, 
                     RCP<const tMap> rowMap, 
                     RCP<Tpetra::Distributor> & distributor,
                     std::vector<GO> & ownedRowGIDs,
                     std::vector<LO> & localRowsSend,
                     std::vector<LO> & localRowsSendBegin,
                     std::vector<LO> & localRowsRecv,
                     std::vector<LO> & localRowsRecvBegin)
{
    const LO numRows = rowMap->getLocalNumElements();
    std::vector<GO> globalIDs(numRows);
    for (LO i=0; i<numRows; i++) globalIDs[i] = rowMap->getGlobalElement(i);
    std::vector<int> remotePIDs(numRows);
    std::vector<LO> remoteLocalRows(numRows);
    RCP<const tMap> rowMap1to1 = inputMatrix->getRowMap();
    rowMap1to1->getRemoteIndexList(Teuchos::ArrayView<GO>(globalIDs),
                                   Teuchos::ArrayView<int>(remotePIDs),
                                   Teuchos::ArrayView<LO>(remoteLocalRows));
    std::vector<int> offProcessorPIDs(numRows);
    const int myPID = rowMap->getComm()->getRank();
    size_t numOffProcessorRows(0), numOnProcessorRows(0);
    ownedRowGIDs.resize(numRows);
    for (LO i=0; i<numRows; i++) {
        if (remotePIDs[i] != myPID) {
            // Remote
            globalIDs[numOffProcessorRows] = globalIDs[i];
            remotePIDs[numOffProcessorRows] = remotePIDs[i];
            remoteLocalRows[numOffProcessorRows++] = remoteLocalRows[i];
        }
        else {
            // Owned
            ownedRowGIDs[numOnProcessorRows++] = globalIDs[i];
        }
    }
    remotePIDs.resize(numOffProcessorRows);
    remoteLocalRows.resize(numOffProcessorRows);
    ownedRowGIDs.resize(numOnProcessorRows);
    // Find unique IDs, and sort
    std::vector<int> recvPIDs;
    getUniqueEntries(remotePIDs, recvPIDs);
    // Create recv GIDs list
    size_t numRecvs = recvPIDs.size();
    std::map<int,int> offProcessorMap;
    for (size_t i=0; i<numRecvs; i++) {
        offProcessorMap.emplace(recvPIDs[i], i);
    }
    std::vector<GO> recvGIDs(numRecvs);
    std::vector<size_t> count(numRecvs, 0);
    for (size_t i=0; i<numOffProcessorRows; i++) {
        auto iter = offProcessorMap.find(remotePIDs[i]);
        recvGIDs[iter->second] = globalIDs[i];
        count[iter->second]++;
    }
    localRowsRecvBegin.resize(numRecvs+1, 0);
    for (size_t i=0; i<numRecvs; i++) {
        localRowsRecvBegin[i+1] = localRowsRecvBegin[i] + count[i];
        count[i] = 0;
    }
    localRowsRecv.resize(numOffProcessorRows);
    for (size_t i=0; i<numOffProcessorRows; i++) {
        auto iter = offProcessorMap.find(remotePIDs[i]);
        const int index = localRowsRecvBegin[iter->second] + count[iter->second];
        localRowsRecv[index] = remoteLocalRows[i];
        count[iter->second]++;
    }
    /*
      for (size_t i=0; i<numRecvs; i++) {
      std::cout << "proc " << myPID << " local rows to be received from proc " << recvPIDs[i]
      << std::endl;
      for (int j=localRowsRecvBegin[i]; j<localRowsRecvBegin[i+1]; j++) {
      std::cout << localRowsRecv[j] << " ";
      }
      std::cout << std::endl;
      }
    */
    // Create distributor
    Teuchos::Array<GO> sendGIDs;
    Teuchos::Array<int> sendPIDs;
    distributor = RCP( new Tpetra::Distributor(rowMap->getComm()) );
    distributor->createFromRecvs(Teuchos::ArrayView<const GO>(recvGIDs),
                                 Teuchos::ArrayView<const int>(recvPIDs),
                                 sendGIDs, sendPIDs);
    TEUCHOS_TEST_FOR_EXCEPTION(distributor->hasSelfMessage() == true,std::runtime_error,
                               "distributor hasSelfMessage error");
    TEUCHOS_TEST_FOR_EXCEPTION(distributor->getNumReceives() != numRecvs,std::runtime_error,
                               "inconsistent numRecvs");
    /*
      Teuchos::ArrayView<const int> procsTo = distributor->getProcsTo();
      Teuchos::ArrayView<const int> procsFrom = distributor->getProcsFrom();
      Teuchos::ArrayView<const size_t> lengthsFrom = distributor->getLengthsFrom();
      Teuchos::ArrayView<const size_t> lengthsTo = distributor->getLengthsTo();
      std::cout << "procsTo for myPID = " << myPID << ": ";
      for (int i=0; i<procsTo.size(); i++) std::cout << procsTo[i] << " ";
      std::cout << std::endl;
      std::cout << "procsFrom for myPID = " << myPID << ": ";
      for (int i=0; i<procsFrom.size(); i++) std::cout << procsFrom[i] << " ";
      std::cout << std::endl;
      std::cout << "lengthsFrom for myPID = " << myPID << ": ";
      for (int i=0; i<lengthsFrom.size(); i++) std::cout << lengthsFrom[i] << " ";
      std::cout << std::endl;
      std::cout << "lengthsTo for myPID = " << myPID << ": ";
      for (int i=0; i<lengthsTo.size(); i++) std::cout << lengthsTo[i] << " ";
      std::cout << std::endl;
    */
    size_t numSends = distributor->getNumSends();
    std::vector<size_t> targetValues(numSends);
    std::vector<size_t> targetSizes(numSends, 1);
    std::vector<size_t> sourceSizes(numRecvs, 1); 
    distributor->doReversePostsAndWaits(Kokkos::View<const size_t*, Kokkos::HostSpace>(count.data(), count.size()),
                                        1,
                                        Kokkos::View<size_t*, Kokkos::HostSpace>(targetValues.data(), targetValues.size()));
    localRowsSendBegin.resize(numSends+1, 0);
    for (size_t i=0; i<numSends; i++) {
        localRowsSendBegin[i+1] = localRowsSendBegin[i] + targetValues[i];
    }
    int numTerms = localRowsSendBegin[numSends];
    localRowsSend.resize(numTerms);
    distributor->doReversePostsAndWaits(Kokkos::View<const int*, Kokkos::HostSpace>(localRowsRecv.data(), localRowsRecv.size()),
                                        Teuchos::ArrayView<const size_t>(count),
                                        Kokkos::View<int*, Kokkos::HostSpace>(localRowsSend.data(), localRowsSend.size()),
                                        Teuchos::ArrayView<const size_t>(targetValues));
    /*
      Teuchos::ArrayView<const int> procsTo = distributor->getProcsTo();
      for (size_t i=0; i<numSends; i++) {
      std::cout << "proc " << myPID << " globalIDs for rows of matrix to be sent to proc " 
      << procsTo[i] << std::endl;
      for (int j=localRowsSendBegin[i]; j<localRowsSendBegin[i+1]; j++) {
      std::cout << rowMap1to1->getGlobalElement(localRowsSend[j]) << " ";
      }
      std::cout << std::endl;
      }
    */
    // switch localRowsRecv to on-processor rather than off-processor localRows
    for (size_t i=0; i<numRecvs; i++) count[i] = 0;
    for (size_t i=0; i<numOffProcessorRows; i++) {
        auto iter = offProcessorMap.find(remotePIDs[i]);
        const int index = localRowsRecvBegin[iter->second] + count[iter->second];
        localRowsRecv[index] = rowMap->getLocalElement(globalIDs[i]);
        count[iter->second]++;
    }
}


// Get unique IDs and sort them
template <class SC, class LO, class GO, class NO>
void TpetraFunctions<SC,LO,GO,NO>::
getUniqueEntries(const std::vector<int> & vector, 
                 std::vector<int> & vectorUnique)
{
    vectorUnique = vector;
    std::sort(vectorUnique.begin(), vectorUnique.end());
    auto iter = std::unique(vectorUnique.begin(), vectorUnique.end());
    vectorUnique.erase(iter, vectorUnique.end());
}

} // namespace

#endif
