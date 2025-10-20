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
#include <sys/time.h>
#include <sys/resource.h>
#include <cstdlib>
#include <unordered_set>

#include "d3_solver.h"

// Direct Domain Decomposition Solver, a sparse distributed memory direct solver based on
// domain decomposition concepts.
// Author: Clark R. Dohrmann

D3Solver::D3Solver(MPI_Comm commIn,
                   const int msg_levelIn,
                   const int num_threadsIn,
                   const int reorder_optionIn,
                   const int debug_levelIn) :
  comm(commIn),
  debug_level(debug_levelIn)
{
#ifndef USE_INTEL_PARDISO
  ThrowAssert(0, "d3_solver currently requires an Intel build with MKL/Pardiso");
#endif
  msg_level = msg_levelIn;
  num_threads = num_threadsIn;
  debug_level = debug_levelIn;
  reorder_option = reorder_optionIn;
  MPI_Comm_rank(comm, &myPID);
  int numProc;
  MPI_Comm_size(comm, &numProc);
  ThrowAssert(numProc > 1, "d3_solver currently must be run on at least 2 MPI processes");
}

D3Solver::~D3Solver()
{
  #ifdef USE_INTEL_PARDISO
    pardiso_solver.cleanup();
  #endif
  for (int i=0; i<num_level; i++) {
    if (comm_level[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_level[i]);
    }
  }
}

int D3Solver::getLocalID_unsorted(const int gID,
                                  const std::vector<int> & vec) const
{
  int index = -1;
  for (size_t i=0; i<vec.size(); i++) {
    if (gID == vec[i]) {
      index = i;
      break;
    }
  }
  ThrowAssert(index != -1, "index not found");
  return index;
}

int D3Solver::getLocalID(const int gID,
                         const std::vector<int> & svec,
                         const bool do_not_throw) const
{
  /*
  auto it = std::lower_bound(svec.begin(), svec.end(), gID);
  const bool valid = (it != svec.end()) && (*it == gID);
  if ((valid == false) && do_not_throw) {
    return -1;
  }
  ThrowAssert(valid, "index not found");
  return std::distance(svec.begin(), it);
  */
  return getLocalID(gID, svec.data(), svec.size(), do_not_throw);
}

int D3Solver::getLocalID(const int gID,
                         const int* array,
                         const int length,
                         const bool do_not_throw) const
{
  auto it = std::lower_bound(array, array+length, gID);
  const bool valid = (it != array+length) && (*it == gID);
  if ((valid == false) && do_not_throw) {
    return -1;
  }
  if (valid == false) {
    std::cout << "myPID, gID = " << myPID << " " << gID << std::endl;
  }
  ThrowAssert(valid, "index not found");
  return std::distance(array, it);
}

void D3Solver::gatherScatterSol(std::vector<double> & sol,
                                std::vector<double> & solAll) const
{
  const int numRows = sol.size();
  ThrowAssert(numRows == numRows_proc, "incompatible number of rows");
  int numProc;
  MPI_Comm_size(comm, &numProc);
  std::vector<int> numRowsProc;
  const int root = 0;
  if (myPID == root) {
    numRowsProc.resize(numProc);
  }
  MPI_Gather(&numRows, 1, MPI_INT, numRowsProc.data(), 1, MPI_INT, root, comm);
  int numRowsRoot(0);
  for (size_t i=0; i<numRowsProc.size(); i++) {
    numRowsRoot += numRowsProc[i];
  }
  MPI_Bcast(&numRowsRoot, 1, MPI_INT, root, comm);
  std::vector<int> displs;
  getDispls(numRowsProc, displs);
  solAll.resize(numRowsRoot);
  MPI_Gatherv(sol.data(), numRows, MPI_DOUBLE, solAll.data(), numRowsProc.data(),
              displs.data(), MPI_DOUBLE, root, comm);
  MPI_Bcast(solAll.data(), numRowsRoot, MPI_DOUBLE, root, comm);
}

void D3Solver::getDispls(const std::vector<int> & numEntriesProc,
                         std::vector<int> & displs) const
{
  const int numProc = numEntriesProc.size();
  displs.assign(numProc, 0);
  displs.resize(numProc, 0);
  for (int i=1; i<numProc; i++) {
    displs[i] = displs[i-1] + numEntriesProc[i-1];
  }
}

void D3Solver::getGraphForMetis(const std::vector<int> & rowBegin,
                                const std::vector<int> & columns,
                                std::vector<idx_t> & rowBeginMetis,
                                std::vector<idx_t> & columnsMetis,
                                std::vector<std::pair<int,int>> & additional_edges)
{
  // remove edges to self
  const int numRows = rowBegin.size() - 1;
  int numTerms = rowBegin[numRows];
  std::vector<int> rowBeginG(numRows+1, 0);
  std::vector<int> columnsG(numTerms);
  numTerms = 0;
  for (int i=0; i<numRows; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      if (col != i) {
        columnsG[numTerms++] = col;
      }
    }
    rowBeginG[i+1] = numTerms;
  }
  columnsG.resize(numTerms);
  // next, make sure graph is symmetric
  std::vector<int> rowBeginGT, columnsGT;
  getGraphTranspose(rowBeginG, columnsG, rowBeginGT, columnsGT);
  std::vector<int> count(numRows, 0);
  std::vector<bool> colFlag(numRows, false);
  int num_additional_edges = 0;
  for (int i=0; i<numRows; i++) {
    for (int j=rowBeginG[i]; j<rowBeginG[i+1]; j++) colFlag[columnsG[j]] = true;
    int num_cols = rowBeginG[i+1] - rowBeginG[i];
    for (int j=rowBeginGT[i]; j<rowBeginGT[i+1]; j++) {
      if (colFlag[columnsGT[j]] == false) {
        num_additional_edges++;
        num_cols++;
      }
    }
    count[i] = num_cols;
    for (int j=rowBeginG[i]; j<rowBeginG[i+1]; j++) colFlag[columnsG[j]] = false;
  }
  rowBeginMetis.resize(numRows+1, 0);
  for (int i=0; i<numRows; i++) {
    rowBeginMetis[i+1] = rowBeginMetis[i] + count[i];
  }
  additional_edges.resize(num_additional_edges);
  num_additional_edges = 0;
  numTerms = rowBeginMetis[numRows];
  columnsMetis.resize(numTerms);
  numTerms = 0;
  for (int i=0; i<numRows; i++) {
    for (int j=rowBeginG[i]; j<rowBeginG[i+1]; j++) {
      const int col = columnsG[j];
      colFlag[col] = true;
      columnsMetis[numTerms++] = columnsG[j];
    }
    for (int j=rowBeginGT[i]; j<rowBeginGT[i+1]; j++) {
      if (colFlag[columnsGT[j]] == false) {
        columnsMetis[numTerms++] = columnsGT[j];
        additional_edges[num_additional_edges++] = std::make_pair(i, columnsGT[j]);
      }
    }
    for (int j=rowBeginG[i]; j<rowBeginG[i+1]; j++) colFlag[columnsG[j]] = false;
  }
  // sort columns
  std::vector<int> sortedCols(numRows);
  for (int i=0; i<numRows; i++) {
    const int num_cols = rowBeginMetis[i+1] - rowBeginMetis[i];
    int index = rowBeginMetis[i];
    for (int j=0; j<num_cols; j++) sortedCols[j] = columnsMetis[index++];
    std::sort(sortedCols.begin(), sortedCols.begin() + num_cols);
    index = rowBeginMetis[i];
    for (int j=0; j<num_cols; j++) columnsMetis[index++] = sortedCols[j];
  }
  if (num_additional_edges > 0) {
    if (debug_level) {
      std::cout << "number of additional edges added to graph for Metis = "
                << num_additional_edges << std::endl;
    }
  }
}

void D3Solver::getLevelsAndLocations(const int numProc,
                                     std::vector<int> & level,
                                     std::vector<int> & location) const
{
  const int num_node_tree = 2*numProc - 1;
  level.resize(num_node_tree, 0);
  location.resize(num_node_tree, 0);
  std::vector<int> location_level(num_level+1, 0), count_level(num_level+1, 0);
  int curr_level = 0;
  int node = 0;
  while (node < num_node_tree) {
    location[node] = location_level[curr_level];
    level[node] = curr_level;
    node++;
    location_level[curr_level]++;
    count_level[curr_level]++;
    const int count = count_level[curr_level];
    if (count == 1) {
      if (curr_level > 0) {
        count_level[0] = 0;
      }
      curr_level = 0;
    }
    else if (count == 2) {
      count_level[curr_level] = 0;
      curr_level++;
    }
  }
}

void D3Solver::extractRowSubIDs(const std::vector<int> & node_begin,
                                const std::vector<int> & node_sub_id,
                                const std::vector<idx_t> & order,
                                std::vector<int> & row_sub_id) const
{
  const int numRows = order.size();
  row_sub_id.resize(numRows);
  const int numSubs = node_begin.size() - 1;
  for (int i=0; i<numRows; i++) {
    int id = -1;
    for (int j=0; j<numSubs; j++) {
      if ((order[i] >= node_begin[j]) && (order[i] < node_begin[j+1])) {
        id = j;
        break;
      }
    }
    ThrowAssert(id != -1, "row not found in bounds of node_begin");
    row_sub_id[i] = node_sub_id[id];
  }
}

void D3Solver::checkRowSubIDs(const std::vector<int> & rowSubIDs,
                              const std::vector<int> & rowBegin,
                              const std::vector<int> & columns) const
{
  int maxSubID = 0;
  for (size_t i=0; i<rowSubIDs.size(); i++) {
    const int sub = std::abs(rowSubIDs[i]);
    if (sub > maxSubID) maxSubID = sub;
  }
  const int num_group = maxSubID + 1;
  std::vector<std::vector<int>> subI(numProcSolver);
  for (size_t i=0; i<rowSubIDs.size(); i++) {
    const int sub = rowSubIDs[i];
    if (sub >= 0) {
      subI[sub].push_back(i);
    }
  }
  for (int i=0; i<numProcSolver; i++) {
    std::vector<int> adj_sub(num_group, 0);
    for (size_t j=0; j<subI[i].size(); j++) {
      const int row = subI[i][j];
      for (int k=rowBegin[row]; k<rowBegin[row+1]; k++) {
        const int col = columns[k];
        const int sub2 = std::abs(rowSubIDs[col]);
        adj_sub[sub2] = 1;
      }
    }
    for (int j=0; j<numProcSolver; j++) {
      if (j == i) {
        ThrowAssert(adj_sub[j] == 1, "subdomain has no interior unknowns");
      }
      else {
        ThrowAssert(adj_sub[j] == 0, "partitioning error");
      }
    }
  }
}

void D3Solver::get_separators(const std::vector<int> & rowSubIDs,
                              std::vector<int> & sepIDs,
                              std::vector<int> & sepBegin,
                              std::vector<int> & sepRows) const
{
  const int numRows = rowSubIDs.size();
  const int num_sep = numProcSolver - 1;
  sepIDs.resize(num_sep);
  for (int i=0; i<num_sep; i++) {
    sepIDs[i] = numProcSolver + i;
  }
  std::vector<int> count(2*numProcSolver, 0);
  for (int i=0; i<numRows; i++) {
    if (rowSubIDs[i] < 0) {
      const int sepID = -rowSubIDs[i];
      count[sepID]++;
    }
  }
  if (debug_level) {
    std::cout << "number of separators = " << num_sep << std::endl;
  }
  sepBegin.resize(num_sep+1, 0);
  for (int i=0; i<num_sep; i++) {
    const int sep = sepIDs[i];
    sepBegin[i+1] = sepBegin[i] + count[sep];
    count[i] = 0;
  }
  const int num_terms = sepBegin[num_sep];
  sepRows.resize(num_terms);
  for (int i=0; i<numRows; i++) {
    if (rowSubIDs[i] < 0) {
      const int sepID = -rowSubIDs[i];
      const int index = getLocalID(sepID, sepIDs);
      const int index2 = sepBegin[index] + count[index];
      sepRows[index2] = i;
      count[index]++;
    }
  }
}
                    
void D3Solver::getGraphTranspose(const std::vector<int> & rowBegin,
                                 const std::vector<int> & columns,
                                 std::vector<int> & rowBeginT,
                                 std::vector<int> & columnsT) const
{
  // Assumption: number of columns equals number of rows
  const int numRows = rowBegin.size() - 1;
  std::vector<int> count(numRows, 0);
  for (int i=0; i<numRows; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      ThrowAssert(col < numRows, "graph not square");
      count[col]++;
    }
  }
  rowBeginT.resize(numRows+1, 0);
  for (int i=0; i<numRows; i++) {
    rowBeginT[i+1] = rowBeginT[i] + count[i];
    count[i] = 0;
  }
  const int num_terms = rowBeginT[numRows];
  columnsT.resize(num_terms);
  for (int i=0; i<numRows; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      const int index = rowBeginT[col] + count[col];
      columnsT[index] = i;
      count[col]++;
    }
  }
}

int D3Solver::get_proc_for_row(const int row,
                               const std::vector<int> & numRowsAll,
                               const int numProc,
                               int & first_proc,
                               int & first_row) const
{
  int proc = -1;
  while (first_proc < numProc) {
    const bool cond1 = (row >= first_row) && (row < first_row+numRowsAll[first_proc]);
    const bool cond2 = numRowsAll[first_proc] != 0;
    if (cond1 && cond2) {
      proc = first_proc;
      break;
    }
    else {
      first_row += numRowsAll[first_proc];
      first_proc++;
    }
  }
  ThrowAssert(proc != -1, "processor not found");
  return proc;
}

void D3Solver::assign_graph(const std::vector<int> & rowBegin,
                            const std::vector<int> & columns,
                            const std::vector<int> & extraEdges)
{
  num_extra_edges = extraEdges.size() / 2;
  if (num_extra_edges == 0) {
    rowBeginPtr = &rowBegin;
    columnsPtr = &columns;
  }
  else {
    update_graph(rowBegin, columns, extraEdges);
    rowBeginPtr = &rowBeginUse;
    columnsPtr = &columnsUse;
    rowBeginOrig = rowBegin;
  }
}

void D3Solver::scatter_additional_edges(const std::vector<std::pair<int,int>> & additional_edges,
                                        std::vector<int> & extraEdges)
{
  int numProc, root(0);
  MPI_Comm_size(comm, &numProc);
  std::vector<int> numRowsAll, count, displs;
  if (myPID == root) {
    numRowsAll.resize(numProc);
    count.resize(numProc, 0);
    displs.resize(numProc, 0);
  }
  MPI_Gather(&numRows_proc, 1, MPI_INT, numRowsAll.data(), 1, MPI_INT, root, comm);
  std::vector<int> row_col_pair_send;
  if (myPID == root) {
    for (int i=1; i<numProc; i++) {
      displs[i] = displs[i-1] + numRowsAll[i-1];
    }
    const int num_additional_edges = additional_edges.size();
    row_col_pair_send.resize(2*num_additional_edges);
    int first_proc(0), first_row(0), index(0);
    for (int i=0; i<num_additional_edges; i++) {
      const int row = additional_edges[i].first;
      const int col = additional_edges[i].second;
      row_col_pair_send[index++] = row;
      row_col_pair_send[index++] = col;
      const int proc = get_proc_for_row(row, numRowsAll, numProc, first_proc, first_row);
      count[proc] += 2;
    }
  }
  int num_extra;
  MPI_Scatter(count.data(), 1, MPI_INT, &num_extra, 1, MPI_INT, root, comm);
  extraEdges.resize(num_extra);
  if (myPID == root) {
    for (int i=1; i<numProc; i++) {
      displs[i] = displs[i-1] + count[i-1];
    }
  }
  MPI_Scatterv(row_col_pair_send.data(), count.data(), displs.data(), MPI_INT,
               extraEdges.data(), num_extra, MPI_INT, root, comm);
  num_extra /= 2;
}

void D3Solver::update_graph(const std::vector<int> & rowBegin,
                            const std::vector<int> & columns,
                            const std::vector<int> & extraEdges)
{
  std::vector<int> count(numRows_proc, 0);
  for (int i=0; i<numRows_proc; i++) {
    count[i] = rowBegin[i+1] - rowBegin[i];
  }
  int index = 0;
  const int num_extra = extraEdges.size() / 2;
  for (int i=0; i<num_extra; i++) {
    const int row = extraEdges[index];
    index += 2;
    const int local_row = row - startGID;
    ThrowAssert((local_row >= 0) && (local_row < numRows_proc), "row is out of range");
    count[local_row]++;
  }
  rowBeginUse.resize(numRows_proc+1, 0);
  for (int i=0; i<numRows_proc; i++) {
    rowBeginUse[i+1] = rowBeginUse[i] + count[i];
    count[i] = 0;
  }
  const int numTerms = rowBeginUse[numRows_proc];
  columnsUse.resize(numTerms);
  valuesUse.resize(numTerms, 0);
  for (int i=0; i<numRows_proc; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      const int index = rowBeginUse[i] + count[i];
      columnsUse[index] = col;
      count[i]++;
    }
  }
  index = 0;
  for (int i=0; i<num_extra; i++) {
    const int row = extraEdges[index++];
    const int col = extraEdges[index++];
    const int local_row = row - startGID;
    const int index = rowBeginUse[local_row] + count[local_row];
    columnsUse[index] = col;
    count[local_row]++;
  }
}

void D3Solver::getRowSubIDs(const std::vector<int> & rowBegin,
                            const std::vector<int> & columns)
{
  GatherToRootSimple gatherer(rowBegin, columns, comm);
  gatherer.initialize();
  const std::vector<int> & rowBeginRoot = gatherer.getRowBeginRoot();
  const std::vector<int> & columnsRoot = gatherer.getColumnsRoot();
  std::vector<std::pair<int,int>> additional_edges;
  int numRows, numSep, numTerms, numRowsB;
  if (myPID == 0) {
    std::vector<idx_t> rowBeginMetis, columnsMetis;
    getGraphForMetis(rowBeginRoot, columnsRoot, rowBeginMetis, columnsMetis, additional_edges);
    numRows = rowBeginRoot.size() - 1;
    std::vector<idx_t> options(METIS_NOPTIONS), perm(numRows), iperm(numRows), sizes(2*numProcSolver);
    int info = METIS_SetDefaultOptions(options.data());
    ThrowAssert(info == METIS_OK, "METIS_SetDefaultOptions failed");
    idx_t* vwgt = nullptr;
    METIS_NodeNDP(numRows, rowBeginMetis.data(), columnsMetis.data(), vwgt,
                  numProcSolver, options.data(), perm.data(), iperm.data(), 
                  sizes.data());
    int lowerIndex = 2*numProcSolver - 2;
    int numSepLevel(1), interfaceSize(0);
    const int numLevel = std::log2(numProcSolver) + 1;
    for (int level=0; level<numLevel; level++) {
      if (debug_level) {
        if (level < numLevel-1) {
          std::cout << "separator sizes for level " << level << ": \n";
        }
        else {
          std::cout << "subdomain interior sizes:\n";
        }
      }
      for (int i=0; i<numSepLevel; i++) {
        const int sepSize = sizes[lowerIndex--];
        if (debug_level) std::cout << sepSize << std::endl;
        if (level < numLevel-1) interfaceSize += sepSize;
      }
      numSepLevel *= 2;
    }
    if (debug_level) std::cout << "total interface size = " << interfaceSize << std::endl;
    std::vector<int> level, location;
    getLevelsAndLocations(numProcSolver, level, location);
    const int num_level_nd = std::log2(numProcSolver) + 1;
    std::vector<int> start_level(num_level_nd);
    int delta = numProcSolver;
    for (int i=1; i<num_level_nd; i++) {
      start_level[i] = start_level[i-1] + delta;
      delta /= 2;
    }
    const int num_node = location.size();
    std::vector<int> node_size(num_node), node_sub_id(num_node), node_begin(num_node+1, 0);
    for (int i=0; i<num_node; i++) {
      node_size[i] = sizes[start_level[level[i]] + location[i]];
      node_sub_id[i] = start_level[level[i]] + location[i];
      if (level[i] > 0) node_sub_id[i] *= -1;
      node_begin[i+1] = node_begin[i] + node_size[i];
    }
    extractRowSubIDs(node_begin, node_sub_id, iperm, rowSubIDs);
    checkRowSubIDs(rowSubIDs, rowBeginRoot, columnsRoot);
    get_separators(rowSubIDs, sepIDs, sepBegin, sepRows);
    //    getS2I(rowSubIDs, rowBeginRoot, columnsRoot, sepIDs, s2i, s2iBegin);
    numSep = sepIDs.size();
    numTerms = sepRows.size();
    // rowsB
    numRowsB = 0;
    for (int i=0; i<numRows; i++) {
      if (rowSubIDs[i] < 0) numRowsB++;
    }
    rowsB.resize(numRowsB);
    numRowsB = 0;
    for (int i=0; i<numRows; i++) {
      if (rowSubIDs[i] < 0) rowsB[numRowsB++] = i;
    }
    for (int i=0; i<numProcSolver/2; i++) {
      std::vector<bool> adjacent_sep(numProcSolver/2, false);
      for (int j=sepBegin[i]; j<sepBegin[i+1]; j++) {
        const int row = sepRows[j];
        for (int k=rowBeginRoot[row]; k<rowBeginRoot[row+1]; k++) {
          const int col = columnsRoot[k];
          const int sub = rowSubIDs[col];
          if (sub < 0) {
            const int sep2 = -sub - numProcSolver;
            if (sep2 < numProcSolver/2) adjacent_sep[sep2] = true;
          }
        }
      }
    }
  }
  MPI_Barrier(comm);
  std::vector<int> extraEdges; // extra edges on each proc
  scatter_additional_edges(additional_edges, extraEdges);
  assign_graph(rowBegin, columns, extraEdges);

  MPI_Bcast(&numRows, 1, MPI_INT, 0, comm);
  MPI_Bcast(&numSep, 1, MPI_INT, 0, comm);
  MPI_Bcast(&numTerms, 1, MPI_INT, 0, comm);
  MPI_Bcast(&numRowsB, 1, MPI_INT, 0, comm);
  if (myPID != 0) {
    rowSubIDs.resize(numRows);
    sepIDs.resize(numSep);
    sepRows.resize(numTerms);
    sepBegin.resize(numSep+1);
    rowsB.resize(numRowsB);
  }
  MPI_Bcast(rowSubIDs.data(), numRows, MPI_INT, 0, comm);
  MPI_Bcast(sepIDs.data(), numSep, MPI_INT, 0, comm);
  MPI_Bcast(sepRows.data(), numTerms, MPI_INT, 0, comm);
  MPI_Bcast(sepBegin.data(), numSep+1, MPI_INT, 0, comm);
  MPI_Bcast(rowsB.data(), numRowsB, MPI_INT, 0, comm);
}

std::vector<int> D3Solver::myReceives(const std::vector<int> & mySends)
{
  int numProc;
  MPI_Comm_size(comm, &numProc);
  std::vector<int> sendArray(numProc, 0);
  for (size_t i=0; i<mySends.size(); i++) {
    sendArray[targetMPIs[mySends[i]]] = 1;
  }
  const int n = numProc*numProc;
  std::vector<int> gatherArrayRoot(n);
  int root = 0;
  MPI_Gather(sendArray.data(), numProc, MPI_INT, gatherArrayRoot.data(), numProc,
             MPI_INT, root, comm);
  MPI_Bcast(gatherArrayRoot.data(), n, MPI_INT, root, comm);
  std::vector<int> myRecvs;
  for (int j=0; j<numProc; j++) {
    if (gatherArrayRoot[myPID+numProc*j] == 1) myRecvs.push_back(j);
  }
  return myRecvs;
}

void D3Solver::
communicateMatrixData(const std::vector<int> & activeSubs,
                      const std::vector<std::vector<int>> & num_rows_send,
                      const std::vector<std::vector<int>> & row_GIDs_send,
                      const std::vector<std::vector<int>> & column_counts_send,
                      const std::vector<std::vector<int>> & column_GIDs_send,
                      const std::vector<std::vector<double>> & values_send_here,
                      std::vector<std::vector<int>> & num_rows_recv,
                      std::vector<std::vector<int>> & row_GIDs_recv,
                      std::vector<std::vector<int>> & column_counts_recv,
                      std::vector<std::vector<int>> & column_GIDs_recv,
                      std::vector<std::vector<double>> & values_recv_here,
                      std::vector<int> & my_send_PIDs,
                      std::vector<int> & my_recv_PIDs)
{
  const int numActive = activeSubs.size();
  my_recv_PIDs = myReceives(activeSubs);
  my_send_PIDs.resize(numActive);
  for (int i=0; i<numActive; i++) {
    my_send_PIDs[i] = targetMPIs[activeSubs[i]];
  }
  
  const int num_recvs = my_recv_PIDs.size();
  // number of rows
  num_rows_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) num_rows_recv[i].resize(1);
  communicateData(num_rows_send, my_recv_PIDs, my_send_PIDs, num_rows_recv);
  // row numbers
  row_GIDs_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) row_GIDs_recv[i].resize(num_rows_recv[i][0]);
  communicateData(row_GIDs_send, my_recv_PIDs, my_send_PIDs, row_GIDs_recv);
  // column counts for rows
  column_counts_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) column_counts_recv[i].resize(num_rows_recv[i][0]);
  communicateData(column_counts_send, my_recv_PIDs, my_send_PIDs, column_counts_recv);
  // column GIDs and values
  column_GIDs_recv.resize(num_recvs);
  values_recv_here.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) {
    int num_terms = 0;
    for (int j=0; j<num_rows_recv[i][0]; j++) num_terms += column_counts_recv[i][j];
    column_GIDs_recv[i].resize(num_terms);
    values_recv_here[i].resize(num_terms);
  }
  communicateData(column_GIDs_send, my_recv_PIDs, my_send_PIDs, column_GIDs_recv);
}

template <typename T>
void D3Solver::communicateData(const std::vector<std::vector<T>> & data_send,
                               const std::vector<int> & my_recv_PIDs,
                               const std::vector<int> & my_send_PIDs,
                               std::vector<std::vector<T>> & data_recv,
                               const bool reverse_comm)
{
  MPI_Datatype MPI_type = MPI_INT;
  if constexpr (std::is_same_v<T, double>) MPI_type = MPI_DOUBLE;
  const int num_recvs = my_recv_PIDs.size();
  const int num_sends = my_send_PIDs.size();
  ThrowAssert(my_recv_PIDs.size() == data_recv.size(), "incompatible sizes");
  ThrowAssert(my_send_PIDs.size() == data_send.size(), "incompatible sizes");
  int numProc;
  MPI_Comm_size(comm, &numProc);
  const int tag = 0;
  std::vector<MPI_Request> send_requests(num_sends);
  std::vector<MPI_Request> recv_requests(num_recvs);
  std::vector<MPI_Status> statuses(numProc);
  // communicate data
  int actual_num_sends(0), actual_num_recvs(0);
  for (int i=0; i<num_recvs; i++) {
    // don't receive data from self
    if (my_recv_PIDs[i] != myPID) {
      const int count = data_recv[i].size();
      MPI_Irecv(data_recv[i].data(), count, MPI_type, my_recv_PIDs[i], tag, comm,
                &recv_requests[actual_num_recvs++]);
    }
  }
  for (int i=0; i<num_sends; i++) {
    // don't send data to self, but do copy over data
    if (my_send_PIDs[i] != myPID) {
      const int count = data_send[i].size();
      MPI_Isend(data_send[i].data(), count, MPI_type, my_send_PIDs[i], tag, comm,
                &send_requests[actual_num_sends++]);
    }
    else {
      int index;
      if (reverse_comm) {
        index = getLocalID_unsorted(myPID, my_recv_PIDs);
      }
      else {
        index = getLocalID(myPID, my_recv_PIDs);
      }
      for (size_t j=0; j<data_send[i].size(); j++) {
        data_recv[index][j] = data_send[i][j];
      }
    } 
  }
  MPI_Waitall(actual_num_sends, send_requests.data(), statuses.data());
  MPI_Waitall(actual_num_recvs, recv_requests.data(), statuses.data());
}

void D3Solver::phase1_rhs()
{
  const int numSubs = targetMPIs.size();
  std::vector<std::vector<int>> rhs_index(numSubs);
  for (int i=0; i<numRows_proc; i++) {
    const int gID = startGID + i;
    const int sub = rowSubIDs[gID];
    if (sub >= 0) { // row is in interior of a subdomain
      rhs_index[sub].push_back(i);
    }
  }
  std::vector<int> activeSubs;
  for (int i=0; i<numSubs; i++) {
    if (rhs_index[i].size() > 0) activeSubs.push_back(i);
  }
  const int numActive = activeSubs.size();
  std::vector<std::vector<int>> num_rows_send(numActive);
  rhs_index_send.resize(numActive);
  rhs_send.resize(numActive);
  for (int i=0; i<numActive; i++) {
    const int sub = activeSubs[i]; 
    const int num_rows_sub = rhs_index[sub].size();
    num_rows_send[i].resize(1, num_rows_sub);
    rhs_index_send[i] = rhs_index[sub];
    rhs_send[i].resize(num_rows_sub);
  }
  communicateRhsData(activeSubs, num_rows_send);
}

void D3Solver::communicateRhsData(const std::vector<int> & activeSubs,
                                  const std::vector<std::vector<int>> & num_rows_send_rhs)
{
  const int numActive = activeSubs.size();
  my_recv_PIDs_rhs = myReceives(activeSubs);
  my_send_PIDs_rhs.resize(numActive);
  for (int i=0; i<numActive; i++) {
    my_send_PIDs_rhs[i] = targetMPIs[activeSubs[i]];
  }
  const int num_recvs = my_recv_PIDs_rhs.size();
  // number of rows
  std::vector<std::vector<int>> num_rows_recv_rhs(num_recvs);
  for (int i=0; i<num_recvs; i++) num_rows_recv_rhs[i].resize(1);
  communicateData(num_rows_send_rhs, my_recv_PIDs_rhs, my_send_PIDs_rhs, num_rows_recv_rhs);
  // rhs values
  rhs_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) rhs_recv[i].resize(num_rows_recv_rhs[i][0]);
}

void D3Solver::phase1(const std::vector<int> & rowBegin,
                      const std::vector<int> & columns,
                      std::vector<std::vector<int>> & num_rows_recv,
                      std::vector<std::vector<int>> & row_GIDs_recv,
                      std::vector<std::vector<int>> & column_counts_recv,                      
                      std::vector<std::vector<int>> & column_GIDs_recv)
{
  // communicate [A_{II} A_{IB}; A_{BI}]
  const int numSubs = targetMPIs.size();
  std::vector<int> num_terms_sub(numSubs, 0);
  std::vector<std::vector<int>> row_GIDs(numSubs);
  const int numRows = rowBegin.size() - 1;
  // determine active subdomains, row_GIDs, and memory requirements first
  for (int i=0; i<numRows; i++) {
    const int gID = startGID + i;
    const int sub = rowSubIDs[gID];
    if (sub >= 0) { // row is in interior of a subdomain
      row_GIDs[sub].push_back(gID);
      num_terms_sub[sub] += rowBegin[i+1] - rowBegin[i];
    }
    else { // row is on the interface
      std::unordered_set<int> sub_set2;
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int gID2 = columns[j];
        const int sub2 = rowSubIDs[gID2];
        if (sub2 >= 0) {
          sub_set2.insert(sub2);
          num_terms_sub[sub2]++;
        }
      }
      for (const int& sub2 : sub_set2) {
        row_GIDs[sub2].push_back(gID);
      }
    }
  }
  std::vector<int> activeSubs;
  for (int i=0; i<numSubs; i++) {
    if (row_GIDs[i].size() > 0) activeSubs.push_back(i);
  }
  const int numActive = activeSubs.size();
  std::vector<std::vector<int>> num_rows_send(numActive), row_GIDs_send(numActive),
    column_GIDs_send(numActive), column_counts_send(numActive);
  values_send.resize(numActive);
  values_send_index.resize(numActive);
  for (int i=0; i<numActive; i++) {
    const int sub = activeSubs[i]; 
    const int num_rows_sub = row_GIDs[sub].size();
    num_rows_send[i].resize(1, num_rows_sub);
    row_GIDs_send[i] = row_GIDs[sub];
    column_counts_send[i].resize(num_rows_sub);
    column_GIDs_send[i].resize(num_terms_sub[sub]);
    values_send[i].resize(num_terms_sub[sub], 0.0);
    values_send_index[i].resize(num_terms_sub[sub]);
    num_terms_sub[i] = 0; // Note: okay since activeSubs is in ascending order
  }
  std::vector<int> num_rows_sub(numActive, 0);
  for (int i=0; i<numRows; i++) {
    const int gID = startGID + i;
    const int sub = rowSubIDs[gID];
    if (sub >= 0) { // row is in interior of a subdomain
      const int local_sub = getLocalID(sub, activeSubs);
      const int row = num_rows_sub[local_sub];
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int index = num_terms_sub[local_sub];
        column_GIDs_send[local_sub][index] = columns[j];
        values_send_index[local_sub][index] = j;
        num_terms_sub[local_sub]++;
        column_counts_send[local_sub][row]++;
      }
      num_rows_sub[local_sub]++;
    }
    else { // row is on the interface
      std::unordered_set<int> sub_set_local;
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int gID2 = columns[j];
        const int sub2 = rowSubIDs[gID2];
        if (sub2 >= 0) {
          const int local_sub = getLocalID(sub2, activeSubs);
          sub_set_local.insert(local_sub);
          const int row = num_rows_sub[local_sub];
          column_counts_send[local_sub][row]++;
          const int index = num_terms_sub[local_sub];
          column_GIDs_send[local_sub][index] = columns[j];
          values_send_index[local_sub][index] = j;
          num_terms_sub[local_sub]++;
        }
      }
      for (const int& local_sub : sub_set_local) {
        num_rows_sub[local_sub]++;
      }
    }
  }
  communicateMatrixData(activeSubs, num_rows_send, row_GIDs_send,
                        column_counts_send, column_GIDs_send, values_send,
                        num_rows_recv, row_GIDs_recv, column_counts_recv,
                        column_GIDs_recv, values_recv, my_send_PIDs_sub,
                        my_recv_PIDs_sub);
}

void D3Solver::communicateMatrixValues(const std::vector<double> & values)
{
  const int num_send = values_send.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<values_send[i].size(); j++) {
      values_send[i][j] = values[values_send_index[i][j]];
    }
  }
  communicateData(values_send, my_recv_PIDs_sub, my_send_PIDs_sub, values_recv);
}

bool D3Solver::determine_valid_row(const int gID,
                                   const std::vector<int> & separators) const
{
  const int sep = -rowSubIDs[gID];
  if (sep < separators[0]) return false;
  else return true;
}

void D3Solver::phase2(const int level,
                      const std::vector<int> & rowBegin,
                      const std::vector<int> & columns,
                      const std::vector<int> & separators,
                      const int delta_pid,
                      std::vector<std::vector<int>> & num_rows_recv,
                      std::vector<std::vector<int>> & row_GIDs_recv,
                      std::vector<std::vector<int>> & column_counts_recv,
                      std::vector<std::vector<int>> & column_GIDs_recv)
{
  // communicate [A_{SS} A_{SB}; A_{BS}]
  const int numSeps = separators.size();
  std::vector<int> num_terms_sep(numSeps, 0);
  std::vector<std::vector<int>> row_GIDs(numSeps);
  std::vector<bool> sep_flag(numSeps, false);
  const int numRows = rowBegin.size() - 1;
  // determine active subdomains, row_GIDs, and memory requirements first
  for (int i=0; i<numRows; i++) {
    const int gID = startGID + i;
    const int validRow = determine_valid_row(gID, separators);
    if (validRow) { // row is on the interface Gamma and not already eliminated
      const int sep = -rowSubIDs[gID];
      const bool do_not_throw = true;
      const int local_sep = getLocalID(sep, separators, do_not_throw);
      if (local_sep >= 0) { // row is on a separator in list of separators
        sep_flag[local_sep] = true;
        row_GIDs[local_sep].push_back(gID);
        for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
          const int gID2 = columns[j];
          const int validRow2 = determine_valid_row(gID2, separators);
          if (validRow2) num_terms_sep[local_sep]++;
        }
      }
      else { // row is on the interface but not on separator in separators list
        std::unordered_set<int> local_sep2_set;
        for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
          const int gID2 = columns[j];
          const int sep2 = -rowSubIDs[gID2];
          const int local_sep2 = getLocalID(sep2, separators, do_not_throw);
          if (local_sep2 >= 0) {
            sep_flag[local_sep2] = true;
            local_sep2_set.insert(local_sep2);
            num_terms_sep[local_sep2]++;
          }
        }
        for (const int& local_sep2 : local_sep2_set) {
          row_GIDs[local_sep2].push_back(gID);
        }
      }
    }
  }
  std::vector<int> activeSeps;
  for (int i=0; i<numSeps; i++) {
    if (sep_flag[i]) activeSeps.push_back(i);
  }
  const int numActive = activeSeps.size();
  std::vector<std::vector<int>> num_rows_send(numActive), row_GIDs_send(numActive),
    column_GIDs_send(numActive), column_counts_send(numActive);
  values_send_B[level].resize(numActive);
  values_send_B_index[level].resize(numActive);
  for (int i=0; i<numActive; i++) {
    const int local_sep = activeSeps[i]; 
    const int num_rows_sep = row_GIDs[local_sep].size();
    num_rows_send[i].resize(1, num_rows_sep);
    row_GIDs_send[i] = row_GIDs[local_sep];
    column_counts_send[i].resize(num_rows_sep);
    column_GIDs_send[i].resize(num_terms_sep[local_sep]);
    values_send_B[level][i].resize(num_terms_sep[local_sep]);
    values_send_B_index[level][i].resize(num_terms_sep[local_sep]);
    num_terms_sep[i] = 0; // Note: okay since activeSeps is in ascending order
  }
  std::vector<int> num_rows_sep(numActive, 0);
  for (int i=0; i<numRows; i++) {
    const int gID = startGID + i;
    const int validRow = determine_valid_row(gID, separators);
    if (validRow) {
      const int sep = -rowSubIDs[gID];
      const bool do_not_throw = true;
      const int local_sep = getLocalID(sep, separators, do_not_throw);
      if (local_sep >= 0) { // row is on a separator in list of separators
        const int local_sep_active = getLocalID(local_sep, activeSeps);
        const int row = num_rows_sep[local_sep_active];
        for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
          const int gID2 = columns[j];
          const int validRow2 = determine_valid_row(gID2, separators);
          if (validRow2) {
            const int index = num_terms_sep[local_sep_active];
            column_GIDs_send[local_sep_active][index] = columns[j];
            //            values_send_B[local_sep_active][index] = values[j];
            values_send_B_index[level][local_sep_active][index] = j;
            num_terms_sep[local_sep_active]++;
            column_counts_send[local_sep_active][row]++;
          }
        }
        num_rows_sep[local_sep_active]++;
      }
      else { // row is on the interface but not on separator in separators list
        std::unordered_set<int> local_sep2_set;
        for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
          const int gID2 = columns[j];
          const int sep2 = -rowSubIDs[gID2];
          const int local_sep2 = getLocalID(sep2, separators, do_not_throw);
          if (local_sep2 >= 0) {
            const int local_sep2_active = getLocalID(local_sep2, activeSeps);
            local_sep2_set.insert(local_sep2_active);
            const int row = num_rows_sep[local_sep2_active];
            column_counts_send[local_sep2_active][row]++;
            const int index = num_terms_sep[local_sep2_active];
            column_GIDs_send[local_sep2_active][index] = columns[j];
            //            values_send_B[local_sep2_active][index] = values[j];
            values_send_B_index[level][local_sep2_active][index] = j;
            num_terms_sep[local_sep2_active]++;
          }
        }
        for (const int& local_sep2_active : local_sep2_set) {
          num_rows_sep[local_sep2_active]++;
        }
      }
    }
  }
  std::vector<int> localPIDs(numActive);
  for (int i=0; i<numActive; i++) {
    localPIDs[i] = delta_pid * activeSeps[i];
  }
  communicateMatrixData(localPIDs, num_rows_send, row_GIDs_send,
                        column_counts_send, column_GIDs_send, values_send_B[level],
                        num_rows_recv, row_GIDs_recv, column_counts_recv,
                        column_GIDs_recv, values_recv_B[level], my_send_PIDs_B[level],
                        my_recv_PIDs_B[level]);
}

void D3Solver::phase2_rhs(const int level,
                          const std::vector<int> & separators,
                          const int delta_pid,
                          const int sep_number)
{
  // only communicate rhs for rows in separators
  const int numSeps = separators.size();
  std::vector<std::vector<int>> row_GIDs(numSeps);
  for (int i=0; i<numRows_proc; i++) {
    const int gID = startGID + i;
    const int validRow = determine_valid_row(gID, separators);
    if (validRow) { // row is on the interface Gamma and not already eliminated
      const int sep = -rowSubIDs[gID];
      const bool do_not_throw = true;
      const int local_sep = getLocalID(sep, separators, do_not_throw);
      if (local_sep >= 0) { // row is on a separator in list of separators
        row_GIDs[local_sep].push_back(gID);
      }
    }
  }
  std::vector<int> activeSeps;
  for (int i=0; i<numSeps; i++) {
    if (row_GIDs[i].size() > 0) activeSeps.push_back(i);
  }
  const int numActive = activeSeps.size();
  std::vector<std::vector<int>> num_rows_send(numActive), row_GIDs_send(numActive);
  rhs_send_sep[level].resize(numActive);
  rhs_send_sep_index[level].resize(numActive);
  for (int i=0; i<numActive; i++) {
    const int local_sep = activeSeps[i]; 
    const int num_rows_sep = row_GIDs[local_sep].size();
    num_rows_send[i].resize(1, num_rows_sep);
    row_GIDs_send[i] = row_GIDs[local_sep];
    rhs_send_sep[level][i].resize(num_rows_sep);
    rhs_send_sep_index[level][i].resize(num_rows_sep);
  }
  std::vector<int> num_rows_sep(numActive, 0);
  for (int i=0; i<numRows_proc; i++) {
    const int gID = startGID + i;
    const int validRow = determine_valid_row(gID, separators);
    if (validRow) {
      const int sep = -rowSubIDs[gID];
      const bool do_not_throw = true;
      const int local_sep = getLocalID(sep, separators, do_not_throw);
      if (local_sep >= 0) { // row is on a separator in list of separators
        const int local_sep_active = getLocalID(local_sep, activeSeps);
        const int row = num_rows_sep[local_sep_active];
        rhs_send_sep_index[level][local_sep_active][row] = i;
        num_rows_sep[local_sep_active]++;
      }
    }
  }
  std::vector<int> localPIDs(numActive);
  for (int i=0; i<numActive; i++) {
    localPIDs[i] = delta_pid * activeSeps[i];
  }
  std::vector<std::vector<int>> row_GIDs_recv;
  communicateRhsData(localPIDs, num_rows_send, row_GIDs_send, row_GIDs_recv,
                     my_send_PIDs_sep[level], my_recv_PIDs_sep[level]);
  std::vector<int> sepIDsLevel;
  if (sep_number != -1) {
    const int index = getLocalID(sep_number, sepIDs);
    for (int i=sepBegin[index]; i<sepBegin[index+1]; i++) {
      sepIDsLevel.push_back(sepRows[i]);
    }
  }
  const int num_recv = row_GIDs_recv.size();
  rhs_recv_sep[level].resize(num_recv);
  rhs_recv_sep_index[level].resize(num_recv);
  for (int i=0; i<num_recv; i++) {
    const int length = row_GIDs_recv[i].size();
    rhs_recv_sep[level][i].resize(length);
    rhs_recv_sep_index[level][i].resize(length);
    for (int j=0; j<length; j++) {
      const int index = getLocalID(row_GIDs_recv[i][j], sepIDsLevel);
      rhs_recv_sep_index[level][i][j] = index;
    }
  }
}

void D3Solver::
communicateRhsData(const std::vector<int> & activeSubs,
                   const std::vector<std::vector<int>> & num_rows_send,
                   const std::vector<std::vector<int>> & row_GIDs_send,
                   std::vector<std::vector<int>> & row_GIDs_recv,
                   std::vector<int> & my_send_PIDs,
                   std::vector<int> & my_recv_PIDs)
{
  const int numActive = activeSubs.size();
  my_recv_PIDs = myReceives(activeSubs);
  my_send_PIDs.resize(numActive);
  for (int i=0; i<numActive; i++) {
    my_send_PIDs[i] = targetMPIs[activeSubs[i]];
  }
  
  const int num_recvs = my_recv_PIDs.size();
  // number of rows
  std::vector<std::vector<int>> num_rows_recv(num_recvs);
  for (int i=0; i<num_recvs; i++) num_rows_recv[i].resize(1);
  communicateData(num_rows_send, my_recv_PIDs, my_send_PIDs, num_rows_recv);
  // row numbers
  row_GIDs_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) row_GIDs_recv[i].resize(num_rows_recv[i][0]);
  communicateData(row_GIDs_send, my_recv_PIDs, my_send_PIDs, row_GIDs_recv);
}

std::vector<int> D3Solver::getSubRows(const std::vector<std::vector<int>> & row_GIDs_recv) const
{
  const int num_recvs = row_GIDs_recv.size();
  int numRows = 0;
  for (int i=0; i<num_recvs; i++) numRows += row_GIDs_recv[i].size();
  std::vector<int> rows(numRows);
  numRows = 0;
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<row_GIDs_recv[i].size(); j++) {
      rows[numRows++] = row_GIDs_recv[i][j];
    }
  }
  return rows;
}

void D3Solver::generateSubMatrices(const std::vector<std::vector<int>> & row_GIDs_recv,
                                   const std::vector<std::vector<int>> & column_counts_recv,
                                   const std::vector<std::vector<int>> & column_GIDs_recv,
                                   std::vector<int> & rowBegin,
                                   std::vector<int> & columns,
                                   std::vector<double> & values,
                                   std::vector<int> & rowGIDs)
{
  rowGIDs = getSubRows(row_GIDs_recv); // these are in ascending order
  for (size_t i=1; i<rowGIDs.size(); i++) {
    ThrowAssert(rowGIDs[i] > rowGIDs[i-1], "rowGIDs not in ascending order");
  }
  int numRows = rowGIDs.size();
  for (int i=0; i<numRows; i++) {
    const int gID = rowGIDs[i];
    if (rowSubIDs[gID] < 0) rowsBSub.push_back(i);
    else rowsISub.push_back(i);
  }
  std::vector<int> count(numRows, 0);
  const int num_recvs = row_GIDs_recv.size();
  index_map_sub.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) {
    index_map_sub[i].resize(column_GIDs_recv[i].size());
    for (size_t j=0; j<row_GIDs_recv[i].size(); j++) {
      const int gID = row_GIDs_recv[i][j];
      const int local_row = getLocalID(gID, rowGIDs);
      count[local_row] += column_counts_recv[i][j];
    }
  }
  rowBegin.resize(numRows+1, 0);
  for (int i=0; i<numRows; i++) {
    rowBegin[i+1] = rowBegin[i] + count[i];
    count[i] = 0;
  }
  const int numTerms = rowBegin[numRows];
  columns.resize(numTerms);
  values.resize(numTerms);
  for (int i=0; i<num_recvs; i++) {
    int num_terms_data = 0;
    for (size_t j=0; j<row_GIDs_recv[i].size(); j++) {
      const int gID = row_GIDs_recv[i][j];
      const int local_row = getLocalID(gID, rowGIDs);
      for (size_t k=0; k<column_counts_recv[i][j]; k++) {
        const int col = column_GIDs_recv[i][num_terms_data];
        bool check = true;
        if ((rowSubIDs[gID] < 0) && (rowSubIDs[col] < 0)) {
          if (col != gID) check = false;
        }
        ThrowAssert(check, "should not be off-diagonal A_BB entries here");
        const int localCol = getLocalID(col, rowGIDs);
        const int index = rowBegin[local_row] + count[local_row];
        index_map_sub[i][num_terms_data++] = index;
        columns[index] = localCol;
        count[local_row]++;
      }
    }
  }
  // check
  for (size_t i=0; i<rowsBSub.size(); i++) {
    const int row = rowsBSub[i];
    const int rowGID = rowGIDs[row];
    ThrowAssert(rowSubIDs[rowGID] < 0, "logic error a");
    for (int j=rowBegin[row]; j<rowBegin[row+1]; j++) {
      const int col = columns[j];
      const int colGID = rowGIDs[col];
      if (colGID != rowGID) {
        ThrowAssert(rowSubIDs[colGID] >= 0, "logic error b");
      }
    }
  }
}

void D3Solver::sort_cols_and_indices(int* cols,
                                     int* indices,
                                     std::pair<int,int>* col_index_pairs,
                                     const int num_cols)
{
  for (int i=0; i<num_cols; i++) {
    col_index_pairs[i] = std::make_pair(cols[i], indices[i]);
  }
  std::sort(col_index_pairs, col_index_pairs + num_cols);
  for (int i=0; i<num_cols; i++) {
    cols[i] = col_index_pairs[i].first;
    indices[i] = col_index_pairs[i].second;
  }
}

void D3Solver::extractMatrix(const int level,
                             const std::vector<std::vector<int>> & row_GIDs_recv,
                             const std::vector<std::vector<int>> & column_counts_recv,
                             const std::vector<std::vector<int>> & column_GIDs_recv,
                             std::vector<int> & rowBegin,
                             std::vector<int> & columns,
                             std::vector<double> & values,
                             std::vector<int> & rowGIDs)
{
  rowGIDs = getSubRows(row_GIDs_recv); // these are in ascending order
  for (size_t i=1; i<rowGIDs.size(); i++) {
    ThrowAssert(rowGIDs[i] > rowGIDs[i-1], "rowGIDs not in ascending order");
  }
  int numRows = rowGIDs.size();
  std::vector<int> count(numRows, 0);
  const int num_recvs = row_GIDs_recv.size();
  std::vector<std::vector<int>> & index_map = index_map_B[level];
  index_map.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) {
    index_map[i].resize(column_GIDs_recv[i].size());
    for (size_t j=0; j<row_GIDs_recv[i].size(); j++) {
      const int gID = row_GIDs_recv[i][j];
      const int local_row = getLocalID(gID, rowGIDs);
      count[local_row] += column_counts_recv[i][j];
    }
  }
  rowBegin.resize(numRows+1, 0);
  for (int i=0; i<numRows; i++) {
    rowBegin[i+1] = rowBegin[i] + count[i];
    count[i] = 0;
  }
  const int numTerms = rowBegin[numRows];
  columns.resize(numTerms);
  values.resize(numTerms);
  for (int i=0; i<num_recvs; i++) {
    int num_terms_data = 0;
    for (size_t j=0; j<row_GIDs_recv[i].size(); j++) {
      const int gID = row_GIDs_recv[i][j];
      const int local_row = getLocalID(gID, rowGIDs);
      for (size_t k=0; k<column_counts_recv[i][j]; k++) {
        const int col = column_GIDs_recv[i][num_terms_data];
        //        const double value = values_recv_here[i][num_terms_data];
        const int localCol = getLocalID(col, rowGIDs);
        const int index = rowBegin[local_row] + count[local_row];
        index_map[i][num_terms_data++] = index;
        columns[index] = localCol;
        //        values[index] = value;
        count[local_row]++;
      }
    }
  }
}

void D3Solver::extractRhs(const int level)
{
  std::vector<double> & rhs = rhs_sep[level];
  const std::vector<std::vector<double>> & rhs_recv = rhs_recv_sep[level];
  const std::vector<std::vector<int>> & index_map = rhs_recv_sep_index[level];
  const int num_recvs = index_map.size();
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<index_map[i].size(); j++) {
      const int index = index_map[i][j];
      rhs[index] = rhs_recv[i][j];
    }
  }
}

void D3Solver::extractMatrix(const int level)
{
  const std::vector<std::vector<int>> & index_map = index_map_B[level];
  std::vector<double> & values = values_B[level];
  const std::vector<std::vector<double>> & vals_recv_B = values_recv_B[level];
  const int num_recvs = index_map.size();
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<index_map[i].size(); j++) {
      const int index = index_map[i][j];
      values[index] = vals_recv_B[i][j];
    }
  }
}

/*
void D3Solver::sortColumns(const std::vector<int> & rowBegin,
                           std::vector<int> & columns,
                           std::vector<double> & values)
{
  const int numRows = rowBegin.size() - 1;
  std::vector<std::pair<int, double>> sortedCols(numRows);
  for (int i=0; i<numRows; i++) {
    int index = 0;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      sortedCols[index++] = std::make_pair(columns[j], values[j]);
    }
    const int num_cols = rowBegin[i+1] - rowBegin[i];
    std::sort(sortedCols.begin(), sortedCols.begin() + num_cols);
    index = 0;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      columns[j] = sortedCols[index].first;
      values[j] = sortedCols[index++].second;
    }
  }
}
*/

void D3Solver::getSubMatrices(const std::vector<int> & rowBegin,
                              const std::vector<int> & columns,
                              std::vector<int> & rowBeginSub,
                              std::vector<int> & columnsSub,
                              std::vector<double> & valuesSub,
                              std::vector<int> & rowGIDsSub)
{
  // subdomain matrices [A_{II} A_{IB}; A_{BI} 0]
  std::vector<std::vector<int>> num_rows_recv, row_GIDs_recv,
    column_counts_recv, column_GIDs_recv;
  phase1(rowBegin, columns, num_rows_recv, row_GIDs_recv,
         column_counts_recv, column_GIDs_recv);
  generateSubMatrices(row_GIDs_recv, column_counts_recv, column_GIDs_recv,
                      rowBeginSub, columnsSub, valuesSub, rowGIDsSub);
  phase1_rhs();  
}

void D3Solver::output_rows(const std::string name,
                           const std::vector<int> & rows)
{
  std::string fname = name + std::to_string(myPID) + ".dat";
  std::ofstream fout;
  fout.open(fname);
  const int numRows = rows.size();
  for (int i=0; i<numRows; i++) {
    const int row = rows[i];
    fout << row+1 << std::endl;
  }
  fout.close();
}

void D3Solver::output_sub_matrices(const std::vector<int> & rowBegin,
                                   const std::vector<int> & columns,
                                   const std::vector<double> & values)
{
  if (debug_level < 2) return;
  const int numRows = rowsISub.size() + rowsBSub.size();
  if (numRows > 0) {
    output_rows("rowsI", rowsISub);
    output_rows("rowsB", rowsBSub);
    std::cout << "subdomain Schur complement size = " << rowsBSub.size() << std::endl;
    std::string fname = "Asub_" + std::to_string(myPID) + ".dat";
    std::ofstream fout;
    fout.open(fname);
    for (int i=0; i<numRows; i++) {
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int col = columns[j];
        fout << i+1 << " " << col+1 << " ";
        fout << std::setw(23) << std::setprecision(16) << values[j] << std::endl;
      }
    }
    fout.close();
  }
}

void D3Solver::resize_vectors()
{
  sep_map.resize(num_level);
  sep_map_recv.resize(num_level);
  AS.resize(num_level);
  A11.resize(num_level); A12.resize(num_level);
  A21.resize(num_level); A22.resize(num_level);
  AS_rhs.resize(num_level);
#ifdef USE_INTEL_PARDISO
  ipiv.resize(num_level);
#endif
  sc.resize(num_level);
  sc_recv.resize(num_level);
  rhs_sc.resize(num_level);
  rhs_sc_recv.resize(num_level);
  rhs_sep.resize(num_level);
  sc_GIDs.resize(num_level+1);
  rowBegin_B.resize(num_level);
  columns_B.resize(num_level);
  values_B.resize(num_level);
  values_send_B.resize(num_level);
  values_send_B_index.resize(num_level);
  values_recv_B.resize(num_level);
  rhs_send_sep.resize(num_level);
  rhs_send_sep_index.resize(num_level);
  rhs_recv_sep.resize(num_level);
  my_send_PIDs_B.resize(num_level);
  my_recv_PIDs_B.resize(num_level);
  comm_level.resize(num_level);
  sep_map_B.resize(num_level);
  index_map_B.resize(num_level);
  rhs_recv_sep_index.resize(num_level);
  my_send_PIDs_sep.resize(num_level);
  my_recv_PIDs_sep.resize(num_level);
  n1a.resize(num_level);
  n2a.resize(num_level);
  timer_factor.resize(num_level);
  timer_factor_dla.resize(num_level);
  timer_solve.resize(num_level);
  timer_solve_dla.resize(num_level);
}

void D3Solver::sort_and_add_zero_diags(std::vector<int> & rowBegin,
                                       std::vector<int> & columns)
{
  const int numRows = rowBegin.size() - 1;
  std::vector<bool> flagB(numRows, false);
  for (size_t i=0; i<rowsBSub.size(); i++) {
    const int row = rowsBSub[i];
    flagB[row] = true;
  }
  const int numTerms = rowBegin[numRows];
  old_to_new_indices.resize(numTerms, -1);
  
  int numTermsNew = rowBegin[numRows] + rowsBSub.size();
  std::vector<int> rowBeginNew(numRows+1, 0), columnsNew(numTermsNew);
  numTermsNew = 0;
  std::vector<std::pair<int,int>> cols_and_indices(numRows);
  for (int i=0; i<numRows; i++) {
    int numCols = rowBegin[i+1] - rowBegin[i];
    for (int j=0; j<numCols; j++) {
      const int orig_index = rowBegin[i] + j;
      const int col = columns[orig_index];
      cols_and_indices[j] = std::make_pair(col, orig_index);
      if (flagB[i]) {
        ThrowAssert(col != i, "diagonal on interface should not be there");
      }
    }
    if (flagB[i]) {
      cols_and_indices[numCols] = std::make_pair(i, -1);
      numCols++;
    }
    std::sort(cols_and_indices.begin(), cols_and_indices.begin() + numCols);
    for (int j=0; j<numCols; j++) {
      const int orig_index = cols_and_indices[j].second;
      if (orig_index != -1) old_to_new_indices[orig_index] = numTermsNew;
      columnsNew[numTermsNew++] = cols_and_indices[j].first;
    }
    rowBeginNew[i+1] = numTermsNew;
  }
  rowBegin = rowBeginNew;
  columns = columnsNew;
  valuesSub.resize(numTermsNew, 0.0);
}

void D3Solver::getProcName()
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int length;
  MPI_Get_processor_name(name, &length);
  node_name = name;
  // for testing purposes only, remove later
  /*
  if ((myPID % 2) == 0) {
    node_name = node_name + "b";
  }
  else {
    node_name = node_name + "aa";
  }
  std::cout << "node name = " << node_name << std::endl;
  */
}

int D3Solver::num_nodes_use(const int num_nodes) const
{
  // nodes to use is a power of 2 for simplicity
  const int log_num_nodes = static_cast<int>(std::log2(num_nodes));
  return std::pow(2, log_num_nodes);
}

void D3Solver::process_names(std::string & all_names,
                             const int numProc,
                             const int max_length,
                             int & num_nodes)
{
  node_names.resize(numProc);
  for (int i=0; i<numProc; i++) {
    node_names[i] = all_names.substr(max_length*i, max_length);
    //    std::cout << "node_names[" << i << "] = " << names[i] << std::endl;
  }
  auto names = node_names;
  std::sort(names.begin(), names.end());
  auto iter = std::unique(names.begin(), names.end());
  names.erase(iter, names.end());
  num_nodes = names.size();
  for (int i=0; i<num_nodes; i++) {
    all_names.insert(max_length*i, names[i]);
  }
}

std::vector<std::vector<int>> D3Solver::gather_node_pids(int & num_nodes)
{
  const int length = node_name.size();
  int max_length;
  MPI_Allreduce(&length, &max_length, 1, MPI_INT, MPI_MAX, comm);
  node_name.resize(max_length, ' ');
  std::string all_names;
  const int root = 0;
  int numProc;
  MPI_Comm_size(comm, &numProc);
  if (myPID == root) {
    all_names.resize(max_length*numProc);
  }
  MPI_Gather(node_name.data(), max_length, MPI_CHAR, all_names.data(), max_length,
             MPI_CHAR, root, comm);
  if (myPID == root) {
    process_names(all_names, numProc, max_length, num_nodes);
  }
  MPI_Bcast(&num_nodes, 1, MPI_INT, root, comm);
  all_names.resize(max_length*num_nodes);
  MPI_Bcast(all_names.data(), all_names.size(), MPI_CHAR, root, comm);
  int my_node = -1;
  for (int i=0; i<num_nodes; i++) {
    std::string name(all_names.begin() + i*max_length,
                     all_names.begin() + (i+1)*max_length);
    if (node_name == name) {
      my_node = i;
    }
  }
  ThrowAssert(my_node != -1, "node not found");
  std::vector<int> node_numbers;
  if (myPID == root) {
    node_numbers.resize(numProc);
  }
  MPI_Gather(&my_node, 1, MPI_INT, node_numbers.data(), 1, MPI_INT, root, comm);
  std::vector<std::vector<int>> node_pids;
  if (myPID == root) {
    node_pids.resize(num_nodes);
    for (int i=0; i<numProc; i++) {
      const int node = node_numbers[i];
      node_pids[node].push_back(i);
    }
  }
  return node_pids;
}

std::vector<int> D3Solver::gather_nnz_proc(const std::vector<int> & rowBegin) const
{
  std::vector<int> nnz_sub(numProcSolver, 0);
  for (int i=0; i<numRows_proc; i++) {
    const int gID = startGID + i;
    const int sub = rowSubIDs[gID];
    if (sub >= 0) {
      const int nnz_row = rowBegin[i+1] - rowBegin[i];
      nnz_sub[sub] += nnz_row;
    }
  }
  int numProc, root(0);
  MPI_Comm_size(comm, &numProc);
  std::vector<int> nnz_sub_all;
  if (myPID == root) {
    nnz_sub_all.resize(numProc*numProcSolver);
  }
  MPI_Gather(nnz_sub.data(), numProcSolver, MPI_INT, nnz_sub_all.data(), numProcSolver,
             MPI_INT, root, comm);
  return nnz_sub_all;
}

int D3Solver::get_best_node(const int sub_start,
                            const int num_subs_per_node,
                            const std::vector<std::vector<int>> & node_pids,
                            const std::vector<int> & nnz_proc,
                            std::vector<bool> & node_flag) const
{
  const int num_nodes = node_flag.size();
  int max_nnz(0), best_node(-1);
  for (int i=0; i<num_nodes; i++) {
    if (node_flag[i] == false) {
      int nnz = 0;
      for (size_t j=0; j<node_pids[i].size(); j++) {
        const int pid = node_pids[i][j];
        for (int k=0; k<num_subs_per_node; k++) {
          const int sub = sub_start + k;
          nnz += nnz_proc[pid*numProcSolver+sub];
        }
      }
      if (nnz > max_nnz) {
        max_nnz = nnz;
        best_node = i;
      }
    }
  }
  ThrowAssert(best_node != -1, "logic error");
  node_flag[best_node] = true;
  return best_node;
}

void D3Solver::get_best_ranks(const int node,
                              const int sub_start,
                              const int num_subs_per_node,
                              const std::vector<std::vector<int>> & node_pids,
                              const std::vector<int> & nnz_proc,
                              std::vector<int> & best_ranks) const
{
  const int num_pids = node_pids[node].size();
  std::vector<bool> pid_flag(num_pids, false);
  for (int i=0; i<num_subs_per_node; i++) {
    const int sub = sub_start + i;
    int max_nnz(0), best_pid(-1);
    for (int j=0; j<num_pids; j++) {
      if (pid_flag[j] == false) {
        const int proc_id = node_pids[node][j];
        const int nnz = nnz_proc[numProcSolver*proc_id+sub];
        if (nnz > max_nnz) {
          max_nnz = nnz;
          best_pid = j;
        }
      }
    }
    if (best_pid == -1) {
      for (int j=0; j<num_pids; j++) {
        if (pid_flag[j] == false) {
          best_pid = j;
          break;
        }
      }
    }
    best_ranks[i] = node_pids[node][best_pid];
    pid_flag[best_pid] = true;
  }
}

void D3Solver::assignTargetMPIs(const std::vector<int> & rowBegin)
{
  int num_nodes;
  std::vector<std::vector<int>> node_pids = gather_node_pids(num_nodes);
  std::vector<int> nnz_proc = gather_nnz_proc(rowBegin);
  targetMPIs.resize(numProcSolver);
  if (myPID == 0) {
    const int num_subs_per_node = numProcSolver/num_nodes_use(num_nodes);
    std::vector<bool> node_flag(num_nodes, false);
    int sub_start = 0;
    std::vector<int> best_ranks(num_subs_per_node);
    while (sub_start < numProcSolver) {
      const int node = get_best_node(sub_start, num_subs_per_node, node_pids,
                                     nnz_proc, node_flag);
      get_best_ranks(node, sub_start, num_subs_per_node, node_pids, nnz_proc, best_ranks);
      for (int i=0; i<num_subs_per_node; i++) {
        targetMPIs[sub_start+i] = best_ranks[i];
      }
      sub_start += num_subs_per_node;
    }
    // check
    auto target_copy = targetMPIs;
    std::sort(target_copy.begin(), target_copy.end());
    auto iter = std::unique(target_copy.begin(), target_copy.end());
    target_copy.erase(iter, target_copy.end());
    ThrowAssert(target_copy.size() == targetMPIs.size(), "duplicate MPIs in targetMPIs");
    if (debug_level) {
      for (int i=0; i<numProcSolver; i++) {
        std::cout << "subdomain " << i << " is on node " << node_names[targetMPIs[i]]
                  << " and has MPI rank " << targetMPIs[i] << std::endl;
      }
    }
  }
  MPI_Bcast(targetMPIs.data(), numProcSolver, MPI_INT, 0, comm);
}

int D3Solver::setNumProcSolver(const int numProcSolver_in)
{
  int logInput = static_cast<int>(std::log2(numProcSolver_in));
  numProcSolver = std::pow(2, logInput);
  if (numProcSolver != numProcSolver_in) {
    if (myPID == 0) {
      std::cout << "numProcSolver should be a power of 2" << std::endl;
      std::cout << "resetting to next lower power of 2" << std::endl;
    }
  }
  return numProcSolver;
}

void D3Solver::initialize(const std::vector<int> & rowBegin_in,
                          const std::vector<int> & columns_in,
                          const int startGID_in,
                          const int numProcSolver_in)
{
  startGID = startGID_in;
  numRows_proc = rowBegin_in.size() - 1;
  numProcSolver = numProcSolver_in;
  num_level = std::log2(numProcSolver);
  getRowSubIDs(rowBegin_in, columns_in);
  getProcName();
  numProcSolver = setNumProcSolver(numProcSolver_in);
  const std::vector<int> & rowBegin = *rowBeginPtr;
  const std::vector<int> & columns = *columnsPtr;
  assignTargetMPIs(rowBegin);
  std::vector<int> rowGIDsSub;
  getSubMatrices(rowBegin, columns, rowBeginSub, columnsSub, valuesSub,
                 rowGIDsSub);
  const int num_rows_sub = rowBeginSub.size() - 1;
  resize_vectors();
  double startTime = clockIt();
  if (num_rows_sub > 0) {
    // at this point the matrix is structurally symmetric by construction
    structurally_symmetric = 1;
    sort_and_add_zero_diags(rowBeginSub, columnsSub);
#ifdef USE_INTEL_PARDISO
    const int num_rows_subB = rowsBSub.size();
    pardiso_solver.initialize(num_rows_sub, rowBeginSub.data(), columnsSub.data(),
                              num_rows_subB, rowsBSub.data(), msg_level, num_threads,
                              reorder_option, structurally_symmetric, debug_level);
#endif
  }
  timer_pardiso_symbolic = clockIt() - startTime;
  int level = 0;
  sc_GIDs[0] = getRowGIDsSubB(rowGIDsSub);
  while (level < num_level) {
    calculate_schur_complement(level, rowBegin, columns, sc_GIDs[level], sc_GIDs[level+1]);
    level++;
  }
  columnsUse.resize(0);
  columnsUse.shrink_to_fit();
}

void D3Solver::getSubMatrices(const std::vector<double> & values,
                              std::vector<double> & valuesSub)
{
  communicateMatrixValues(values);
  const int num_recvs = values_recv.size();
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<values_recv[i].size(); j++) {
      const int index1 = index_map_sub[i][j];
      const int index2 = old_to_new_indices[index1];
      valuesSub[index2] = values_recv[i][j];
    }
  }
}

void D3Solver::assign_values(const std::vector<double> & values_in)
{
  if (num_extra_edges == 0) {
    valuesPtr = &values_in;
  }
  else {
    // Note: extra columns for structural symmetry are at the end of each row
    for (int i=0; i<numRows_proc; i++) {
      int index = rowBeginUse[i];
      for (int j=rowBeginOrig[i]; j<rowBeginOrig[i+1]; j++) {
        valuesUse[index++] = values_in[j];
      }
    }
    valuesPtr = &valuesUse;
  }
}

void D3Solver::factorize(const std::vector<double> & values_in)
{
  assign_values(values_in);
  const std::vector<double> & values = *valuesPtr;
  double startTime = clockIt();
  getSubMatrices(values, valuesSub);
  timer_gather_matrices = clockIt() - startTime;
  
  output_sub_matrices(rowBeginSub, columnsSub, valuesSub);

  startTime = clockIt();
#ifdef USE_INTEL_PARDISO
  if (pardiso_solver.getNumRows() != 0) {
    pardiso_solver.factorize(valuesSub.data());
    sc[0] = pardiso_solver.getSchurComplement();
  }
#endif
  timer_pardiso_numeric = clockIt() - startTime;
  
  int level = 0;
  while (level < num_level) {
    startTime = clockIt();
    calculate_schur_complement(level, values);
    timer_factor[level] = clockIt() - startTime;
    level++;
  }
}

void D3Solver::communicateMatrixValuesB(const int level,
                                        const std::vector<double> & values)
{
  std::vector<std::vector<double>> & values_send = values_send_B[level];
  std::vector<std::vector<int>> & indices = values_send_B_index[level];
  const int num_send = values_send.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<values_send[i].size(); j++) {
      values_send[i][j] = values[indices[i][j]];
    }
  }
  communicateData(values_send_B[level], my_recv_PIDs_B[level], my_send_PIDs_B[level],
                  values_recv_B[level]);
}

void D3Solver::communicateRhsValuesB(const int level,
                                     const std::vector<double> & rhs)
{
  std::vector<std::vector<double>> & rhs_send = rhs_send_sep[level];
  std::vector<std::vector<int>> & indices = rhs_send_sep_index[level];
  const int num_send = rhs_send.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<rhs_send[i].size(); j++) {
      rhs_send[i][j] = rhs[indices[i][j]];
    }
  }
  communicateData(rhs_send_sep[level], my_recv_PIDs_sep[level],
                  my_send_PIDs_sep[level], rhs_recv_sep[level]);
}

void D3Solver::get_comm_data(const int level,
                             int & send_to_pid,
                             int & recv_from_pid,
                             int & recv_index) const
{
  int numSub, mult, sep_start;
  get_level_ints(level, numSub, mult, sep_start);
  get_comm_data(level, numSub, mult, send_to_pid, recv_from_pid, recv_index);
}

void D3Solver::calculate_schur_complement(const int level,
                                          const std::vector<double> & values)
{
  int numSub, mult, sep_start;
  get_level_ints(level, numSub, mult, sep_start);
  communicateMatrixValuesB(level, values);
  extractMatrix(level);
  bool output_mats = false;
  if (debug_level >= 2) output_mats = true;
  if (output_mats) {
    output_matrices("A_BB", rowBegin_B[level], columns_B[level], values_B[level], level);
  }
  int send_to_pid, recv_from_pid, recv_index;
  get_comm_data(level, numSub, mult, send_to_pid, recv_from_pid, recv_index);
  point_to_point_single(send_to_pid, recv_from_pid, sc[level], sc_recv[level], comm_level[level]);
  if (recv_index != -1) {
    const int num_rows = n1a[level] + n2a[level];
    assemble_dense(level, num_rows);
    add_sparse_contrib(level, num_rows);
    eliminate_separator(level);
    rhs_sep[level].resize(n1a[level]);
    if (output_mats) {
      output_dense_matrix("Sc", num_rows, level, AS[level]);
      if (n2a[level] > 0) {
        output_dense_matrix("Sc_red", n2a[level], level, sc[level+1]);
      }
    }
  }
}

void D3Solver::get_comm_data(const int level,
                             const int numSub,
                             const int mult,
                             int & send_to_pid,
                             int & recv_from_pid,
                             int & recv_index) const
{
  int color, key;
  get_color_and_key(numSub, mult, color, key);
  int myPID_level(-1), num_proc_level(-1);
  if (color == 1) {
    MPI_Comm_rank(comm_level[level], &myPID_level);
    MPI_Comm_size(comm_level[level], &num_proc_level);
  }
  get_comm_pairs(num_proc_level, myPID_level, send_to_pid, recv_from_pid, recv_index);
}

void D3Solver::calculate_schur_complement_rhs(const int level,
                                              const std::vector<double> & rhs)
{
  int numSub, mult, sep_start;
  get_level_ints(level, numSub, mult, sep_start);
  communicateRhsValuesB(level, rhs);
  extractRhs(level);
  int send_to_pid, recv_from_pid, recv_index;
  get_comm_data(level, numSub, mult, send_to_pid, recv_from_pid, recv_index);
  point_to_point_single(send_to_pid, recv_from_pid, rhs_sc[level],
                        rhs_sc_recv[level], comm_level[level]);
  if (recv_index != -1) {
    assemble_rhs(level);
    add_sep_contrib(level);
    eliminate_separator_rhs(level);
  }
}

void D3Solver::scatter_sol(const int level)
{
  const std::vector<double> & sol = AS_rhs[level];
  int length = sep_map[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map[level][i];
    rhs_sc[level][i] = sol[row];
  }
  length = sep_map_recv[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map_recv[level][i];
    rhs_sc_recv[level][i] = sol[row];
  }
}

void D3Solver::getSubRhs(const std::vector<double> & rhs)
{
  for (size_t i=0; i<rhs_index_send.size(); i++) {
    for (size_t j=0; j<rhs_index_send[i].size(); j++) {
      rhs_send[i][j] = rhs[rhs_index_send[i][j]];
    }
  }
  communicateData(rhs_send, my_recv_PIDs_rhs, my_send_PIDs_rhs, rhs_recv);
  gatherSubRhsI();
}

void D3Solver::putSubSol(std::vector<double> & sol)
{
  scatterSubSolI();
  const bool reverse_comm = true;
  communicateData(rhs_recv, my_send_PIDs_rhs, my_recv_PIDs_rhs, rhs_send, reverse_comm);
  for (size_t i=0; i<rhs_index_send.size(); i++) {
    for (size_t j=0; j<rhs_index_send[i].size(); j++) {
      sol[rhs_index_send[i][j]] = rhs_send[i][j];
    }
  }
}

void D3Solver::gatherSubRhsI()
{
  const int num_recvs = rhs_recv.size();
  int num_rows = 0;
  for (int i=0; i<num_recvs; i++) num_rows += rhs_recv[i].size();
  rhsI.resize(num_rows);
  num_rows = 0;
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<rhs_recv[i].size(); j++) {
      rhsI[num_rows++] = rhs_recv[i][j];
    }
  }
}

void D3Solver::scatterSubSolI()
{
  const int num_recvs = rhs_recv.size();
  int index = 0;
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<rhs_recv[i].size(); j++) {
      rhs_recv[i][j] = rhsI[index++];
    }
  }
}  

void D3Solver::solve(const std::vector<double> & rhs,
                     std::vector<double> & sol,
                     const int numRhs)
{
  ThrowAssert(numRhs == 1, "solver currently setup for only a single rhs");
  getSubRhs(rhs);
  int num_rows = 0;
#ifdef USE_INTEL_PARDISO
  num_rows = pardiso_solver.getNumRows();
#endif
  sol_pardiso.resize(num_rows);
  rhs_pardiso.assign(num_rows, 0);
  for (size_t i=0; i<rowsISub.size(); i++) {
    rhs_pardiso[rowsISub[i]] = rhsI[i];
  }
#ifdef USE_INTEL_PARDISO
  int phase = 331;
  pardiso_solver.solve(rhs_pardiso.data(), sol_pardiso.data(), phase);
#endif
  for (size_t i=0; i<rowsISub.size(); i++) {
    const int row = rowsISub[i];
    rhs_pardiso[row] = sol_pardiso[row];
  }
  rhs_sc[0].resize(rowsBSub.size());
  for (size_t i=0; i<rowsBSub.size(); i++) {
    rhs_sc[0][i] = sol_pardiso[rowsBSub[i]];
  }
  int level = 0;
  while (level < num_level) {
    const double startTime = clockIt();
    calculate_schur_complement_rhs(level, rhs);
    timer_solve[level] += clockIt() - startTime;
    level++;
  }
  // At this point, we have the root separator solution.
  level = num_level - 1;
  communicate_solution(level, sol);
  while (level >= 0) {
    const double startTime = clockIt();
    backsolve(level);
    communicate_solution(level, sol);
    timer_solve[level] += clockIt() - startTime;
    level--;
  }
  // Finally, calculate solution in subdomain interior
  ThrowAssert(rowsBSub.size() == rhs_sc[0].size(), "inconsistent sizes");
  for (size_t i=0; i<rowsBSub.size(); i++) {
    const int row = rowsBSub[i];
    rhs_pardiso[row] = rhs_sc[0][i];
  }
#ifdef USE_INTEL_PARDISO
  phase = 333;
  pardiso_solver.solve(rhs_pardiso.data(), sol_pardiso.data(), phase);
#endif
  for (size_t i=0; i<rowsISub.size(); i++) {
    rhsI[i] = sol_pardiso[rowsISub[i]];
  }
  putSubSol(sol);
}

void D3Solver::backsolve(const int level)
{
  int numSub, mult, sep_start;
  get_level_ints(level, numSub, mult, sep_start);
  int send_to_pid, recv_from_pid, recv_index;
  get_comm_data(level, numSub, mult, send_to_pid, recv_from_pid, recv_index);
  if (recv_index != -1) {
    const int n1 = n1a[level];
    const int n2 = n2a[level];
    if (n2 > 0) {
      double* sol1 = AS_rhs[level].data();
      double* sol2 = sol1 + n1;
      for (int i=0; i<n2; i++) sol2[i] = rhs_sc[level+1][i];
      if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
        const int num_rhs = 1;
        const int n = n1 + n2; // leading dimension of AS_rhs[level]
        CBLAS_LAYOUT layout = CblasColMajor;
        const double alpha(-1), beta(1);
        cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n1, num_rhs, n2, alpha,
                    A12[level].data(), n1, sol2, n, beta, sol1, n);
#endif
      }
    }
    scatter_sol(level);    
  }
  point_to_point_single(recv_from_pid, send_to_pid, rhs_sc_recv[level],
                        rhs_sc[level], comm_level[level]);
}

void D3Solver::get_level_ints(const int level,
                              int & numSub,
                              int & mult,
                              int & sep_start) const
{
  numSub = numProcSolver / std::pow(2, level);
  mult = std::pow(2, level);
  sep_start = 2*numProcSolver - numSub;
}

void D3Solver::communicate_solution(const int level,
                                    std::vector<double> & sol)
{
  // load separator solution back into rhs_recv_sep
  const int num_recv = rhs_recv_sep[level].size();
  for (int i=0; i<num_recv; i++) {
    for (size_t j=0; j<rhs_recv_sep_index[level][i].size(); j++) {
      const int index = rhs_recv_sep_index[level][i][j];
      rhs_recv_sep[level][i][j] = AS_rhs[level][index];
    }
  }
  const bool reverse_comm = true;
  communicateData(rhs_recv_sep[level], my_send_PIDs_sep[level],
                  my_recv_PIDs_sep[level], rhs_send_sep[level], reverse_comm);
  // unload separator solutions back into sol
  const int num_send = rhs_send_sep[level].size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<rhs_send_sep_index[level][i].size(); j++) {
      const int index = rhs_send_sep_index[level][i][j];
      sol[index] = rhs_send_sep[level][i][j];
    }
  }
}

std::vector<int> D3Solver::getRowGIDsSubB(const std::vector<int> & rowGIDsSub)
{
  const int numRowsB = rowsBSub.size();
  std::vector<int> rowGIDsSubB(numRowsB);
  for (int i=0; i<numRowsB; i++) {
    rowGIDsSubB[i] = rowGIDsSub[rowsBSub[i]];
  }
  return rowGIDsSubB;
}

void D3Solver::get_comm_pairs(const int num_proc_here,
                              const int myPID_here,
                              int & send_to_pid,
                              int & recv_from_pid,
                              int & recv_index) const
{
  send_to_pid = -1; recv_from_pid = -1; recv_index = -1;
  if (myPID_here == -1) return;
  for (int i=0; i<num_proc_here/2; i++) {
    const int index = 2*i;
    const int receiver = index;
    const int sender = index+1;
    if (myPID_here == receiver) {
      recv_from_pid = sender;
      recv_index = i;
    }
    if (myPID_here == sender) send_to_pid = receiver;
  }
}

template <typename T>
void D3Solver::point_to_point_single(const int send_to_pid,
                                     const int recv_from_pid,
                                     const std::vector<T> & send_data,
                                     std::vector<T> & recv_data,
                                     MPI_Comm comm_here)
{
  MPI_Datatype MPI_type = MPI_INT;
  if constexpr (std::is_same_v<T, double>) MPI_type = MPI_DOUBLE;
  const int tag = 0;
  const int num_send = send_data.size();
  const int num_recv = recv_data.size();
  MPI_Request send_request, recv_request;
  MPI_Status status;
  if (recv_from_pid != -1) {
    MPI_Irecv(recv_data.data(), num_recv, MPI_type, recv_from_pid, tag, comm_here,
              &recv_request);
  }
  if (send_to_pid != -1) {
    MPI_Isend(send_data.data(), num_send, MPI_type, send_to_pid, tag, comm_here,
              &send_request);
  }
  if (send_to_pid != -1) {
    MPI_Wait(&send_request, &status);
  }
  if (recv_from_pid != -1) {
    MPI_Wait(&recv_request, &status);
  }
}

void D3Solver::get_schur_gids(const int send_to_pid,
                              const int recv_from_pid,
                              const std::vector<int> & sourceGIDs,
                              std::vector<int> & targetGIDs,
                              MPI_Comm comm_here)
{
  std::vector<int> num_source(1, sourceGIDs.size()), num_target(1);
  point_to_point_single(send_to_pid, recv_from_pid, num_source, num_target,
                        comm_here);
  targetGIDs.resize(num_target[0]);
  point_to_point_single(send_to_pid, recv_from_pid, sourceGIDs, targetGIDs,
                        comm_here);
}

void D3Solver::assemble_dense(const int level,
                              const int n)
{
  // recall that sc and sc_recv use row-major ordering
  // and we choose A to also be col-major
  std::vector<double> & A = AS[level];
  A.assign(n*n, 0);
  int index = 0;
  int length = sep_map[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map[level][i];
    for (int j=0; j<length; j++) {
      const int col = sep_map[level][j];
      A[row+col*n] = sc[level][index++];
    }
  }
  index = 0;
  length = sep_map_recv[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map_recv[level][i];
    for (int j=0; j<length; j++) {
      const int col = sep_map_recv[level][j];
      A[row+col*n] += sc_recv[level][index++];
    }
  }
}

void D3Solver::assemble_rhs(const int level)
{
  const int n = n1a[level] + n2a[level];
  std::vector<double> & rhs = AS_rhs[level];
  rhs.assign(n, 0);
  ThrowAssert(sep_map[level].size() == rhs_sc[level].size(), "unequal sizes");
  int length = sep_map[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map[level][i];
    rhs[row] = rhs_sc[level][i];
  }
  ThrowAssert(sep_map_recv[level].size() == rhs_sc_recv[level].size(), "unequal sizes");
  length = sep_map_recv[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map_recv[level][i];
    rhs[row] += rhs_sc_recv[level][i];
  }
}

void D3Solver::add_sparse_contrib(const int level,
                                  const int sep_number,
                                  const std::vector<int> & not_in_sep,
                                  const std::vector<int> & rowGIDsB)
{
  const int sep = getLocalID(sep_number, sepIDs);
  const int* sep_gIDs = &sepRows[sepBegin[sep]];
  const int num_rows_sep = sepBegin[sep+1] - sepBegin[sep];
  sep_map_B[level].resize(rowGIDsB.size());
  const bool do_not_throw = true;
  for (size_t i=0; i<rowGIDsB.size(); i++) {
    const int row = rowGIDsB[i];
    const int index = getLocalID(row, sep_gIDs, num_rows_sep, do_not_throw);
    if (index != -1) sep_map_B[level][i] = index;
    else sep_map_B[level][i] = num_rows_sep + getLocalID(row, not_in_sep);
  }
}

void D3Solver::add_sparse_contrib(const int level,
                                  const int num_rows)
{
  const std::vector<int> & rowBeginB = rowBegin_B[level];
  const std::vector<int> & columnsB = columns_B[level];
  const std::vector<double> & valuesB = values_B[level];
  std::vector<double> & sc_here = AS[level];
  const int num_rowsB = rowBeginB.size() - 1; 
  ThrowAssert(num_rowsB == int(sep_map_B[level].size()), "unequal lengths");
  for (int i=0; i<num_rowsB; i++) {
    const int row = sep_map_B[level][i];
    for (int j=rowBeginB[i]; j<rowBeginB[i+1]; j++) {
      const int col = sep_map_B[level][columnsB[j]];
      sc_here[row+col*num_rows] += valuesB[j]; // col-major ordering for sc
    }
  }
}

void D3Solver::add_sep_contrib(const int level)
{
  std::vector<double> & rhs_here = AS_rhs[level];
  for (size_t i=0; i<rhs_sep[level].size(); i++) {
    rhs_here[i] += rhs_sep[level][i];
  }
}

int D3Solver::get_num_rows_sep(const int sep_number) const
{
  const int sep = getLocalID(sep_number, sepIDs);
  return sepBegin[sep+1] - sepBegin[sep];
}
                               
void D3Solver::assemble_dense(const int level,
                              const std::vector<int> & gIDs,
                              const std::vector<int> & gIDs_recv,
                              const int sep_number,
                              const std::vector<int> & rowGIDsB,
                              std::vector<int> & not_in_sep)
{
  const int sep = getLocalID(sep_number, sepIDs);
  const int* sep_gIDs = &sepRows[sepBegin[sep]];
  const int num_rows_sep = sepBegin[sep+1] - sepBegin[sep];
  const bool do_not_throw = true;
  not_in_sep.resize(0);
  for (size_t i=0; i<gIDs.size(); i++) {
    const int index = getLocalID(gIDs[i], sep_gIDs, num_rows_sep, do_not_throw);
    if (index == -1) not_in_sep.push_back(gIDs[i]);
  }
  for (size_t i=0; i<gIDs_recv.size(); i++) {
    const int index = getLocalID(gIDs_recv[i], sep_gIDs, num_rows_sep, do_not_throw);
    if (index == -1) not_in_sep.push_back(gIDs_recv[i]);
  }
  std::sort(not_in_sep.begin(), not_in_sep.end());
  auto iter = std::unique(not_in_sep.begin(), not_in_sep.end());
  not_in_sep.erase(iter, not_in_sep.end());
  // add potentially missing rows to not_in_sep (this is rare but verified it is possible)
  int num_additional = 0;
  for (size_t i=0; i<rowGIDsB.size(); i++) {
    const int row = rowGIDsB[i];
    const int index = getLocalID(row, sep_gIDs, num_rows_sep, do_not_throw);
    if (index == -1) {
      const int index2 = getLocalID(row, not_in_sep, do_not_throw);
      if (index2 == -1) {
        not_in_sep.push_back(row);
        num_additional++;
      }
    }
  }
  if (num_additional > 0) {
    std::sort(not_in_sep.begin(), not_in_sep.end());
    auto iter = std::unique(not_in_sep.begin(), not_in_sep.end());
    not_in_sep.erase(iter, not_in_sep.end());
  }
  const int n1 = num_rows_sep;
  n1a[level] = n1;
  n2a[level] = not_in_sep.size();
  sep_map[level].resize(gIDs.size());
  for (int i=0; i<gIDs.size(); i++) {
    int index = getLocalID(gIDs[i], sep_gIDs, n1, do_not_throw);
    if (index != -1) sep_map[level][i] = index;
    else sep_map[level][i] = n1 + getLocalID(gIDs[i], not_in_sep);
  }
  sep_map_recv[level].resize(gIDs_recv.size());
  for (int i=0; i<gIDs_recv.size(); i++) {
    int index = getLocalID(gIDs_recv[i], sep_gIDs, n1, do_not_throw);
    if (index != -1) sep_map_recv[level][i] = index;
    else sep_map_recv[level][i] = n1 + getLocalID(gIDs_recv[i], not_in_sep);
  }
}

void D3Solver::output_dense_matrix(const std::string prefix,
                                   const int numRows,
                                   const int level,
                                   const std::vector<double> & A) const
{
  if (numRows == 0) return;
  if (debug_level < 2) return;
  std::string fname = prefix + "_proc_" + std::to_string(myPID) + "_level"
    + std::to_string(level) + ".dat";
  std::ofstream fout;
  fout.open(fname);
  for (int i=0; i<numRows; i++) {
    for (int j=0; j<numRows; j++) {
      fout << i+1 << " " << j+1 << " ";
      const double value = A[i+j*numRows]; // matrix uses col-major ordering
      fout << std::setw(23) << std::setprecision(16) << value << std::endl;
    }
  }
  fout.close();
}

void D3Solver::output_matrices(const std::string prefix,
                               const std::vector<int> & rowBegin,
                               const std::vector<int> & columns,
                               const std::vector<double> & values,
                               const int level) const
{
  if (debug_level < 2) return;
  const int numRows = rowBegin.size() - 1;
  if (numRows > 0) {
    std::string fname = prefix + "_proc_" + std::to_string(myPID) + "_level"
      + std::to_string(level) + ".dat";
    std::ofstream fout;
    fout.open(fname);
    for (int i=0; i<numRows; i++) {
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int col = columns[j];
        fout << i+1 << " " << col+1 << " ";
        fout << std::setw(23) << std::setprecision(16) << values[j] << std::endl;
      }
    }
    fout.close();
  }
}

void D3Solver::get_color_and_key(const int numSub,
                                 const int mult,
                                 int & color,
                                 int & key) const
{
  color = MPI_UNDEFINED; key = -1;
  for (int i=0; i<numSub; i++) {
    const int rank = mult*i;
    if (myPID == targetMPIs[rank]) {
      color = 1;
      key = i;
    }
  }
}

void D3Solver::calculate_schur_complement(const int level,
                                          const std::vector<int> & rowBegin,
                                          const std::vector<int> & columns,
                                          const std::vector<int> & rowGIDsSubB,
                                          std::vector<int> & not_in_sep)
{
  int numSub, mult, sep_start;
  get_level_ints(level, numSub, mult, sep_start);
  // first step is to gather original matrix contributions to Schur complements
  const int num_sep = numSub/2;
  std::vector<int> separators(num_sep);
  for (int i=0; i<num_sep; i++) separators[i] = sep_start + i;
  const int delta_pid = 2*mult;
  std::vector<std::vector<int>> num_rows_recv, row_GIDs_recv, column_counts_recv,
    column_GIDs_recv;
  phase2(level, rowBegin, columns, separators, delta_pid, num_rows_recv,
         row_GIDs_recv, column_counts_recv, column_GIDs_recv);

  std::vector<int> rowGIDsB;
  
  extractMatrix(level, row_GIDs_recv, column_counts_recv, column_GIDs_recv,
                rowBegin_B[level], columns_B[level], values_B[level], rowGIDsB);

  // split communicator
  int color, key;
  get_color_and_key(numSub, mult, color, key);
  MPI_Comm_split(comm, color, key, &comm_level[level]);
  int myPID_level(-1), num_proc_level(-1);
  if (color == 1) {
    MPI_Comm_rank(comm_level[level], &myPID_level);
    MPI_Comm_size(comm_level[level], &num_proc_level);
  }
  // next step is to gather and sum dense matrix contributions to Schur complements
  int send_to_pid, recv_from_pid, recv_index;
  get_comm_pairs(num_proc_level, myPID_level, send_to_pid, recv_from_pid, recv_index);
  std::vector<int> rowGIDsSubB_recv;
  get_schur_gids(send_to_pid, recv_from_pid, rowGIDsSubB, rowGIDsSubB_recv, comm_level[level]);
  const int length = rowGIDsSubB_recv.size();
  sc_recv[level].resize(length*length);
  rhs_sc_recv[level].resize(length);
  const int sep_number = get_sep_number(sep_start, recv_index);
  phase2_rhs(level, separators, delta_pid, sep_number);
  if (recv_index != -1) {
    assemble_dense(level, rowGIDsSubB, rowGIDsSubB_recv, sep_number, rowGIDsB, not_in_sep);
    add_sparse_contrib(level, sep_number, not_in_sep, rowGIDsB);
  }
  MPI_Barrier(comm);
}

int D3Solver::get_sep_number(const int sep_start,
                             const int recv_index) const
{
  int sep_number = -1;
  if (recv_index != -1) {
    sep_number = sep_start + recv_index;
  }
  return sep_number;
}

void D3Solver::assign_matrix_blocks(const int level)
{
  const int n1 = n1a[level];
  const int n2 = n2a[level];
  const std::vector<double> & A = AS[level]; // A is in col_major format, as are
  // A11, A12, A21, and A22
  A11[level].resize(n1*n1);
  A12[level].resize(n1*n2);
  A21[level].resize(n2*n1);
  A22[level].resize(n2*n2);
  int index11(0), index12(0), index21(0), index22(0);
  int index = 0;
  for (int j=0; j<n1; j++) {
    for (int i=0; i<n1; i++) {
      A11[level][index11++] = A[index++];
    }
    for (int i=0; i<n2; i++) {
      A21[level][index21++] = A[index++];
    }
  }
  for (int j=n1; j<n1+n2; j++) {
    for (int i=0; i<n1; i++) {
      A12[level][index12++] = A[index++];
    }
    for (int i=0; i<n2; i++) {
      A22[level][index22++] = A[index++];
    }
  }
}

void D3Solver::convert_to_row_major(const std::vector<double> & A_col_major,
                                    const int num_rows,
                                    const int num_cols,
                                    std::vector<double> & A_row_major) const
{
  int index = 0;
  for (int j=0; j<num_cols; j++) {
    for (int i=0; i<num_rows; i++) {
      A_row_major[j+i*num_cols] = A_col_major[index++];
    }
  }
}

void D3Solver::eliminate_separator(const int level)
{
  const int n1 = n1a[level];
  const int n2 = n2a[level];
  assign_matrix_blocks(level);
  const double startTime = clockIt();
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    ipiv[level].resize(n1);
    int matrix_layout = LAPACK_COL_MAJOR;
    MKL_INT info = LAPACKE_dgetrf(matrix_layout, n1, n1, A11[level].data(), n1,
                                  ipiv[level].data());
    ThrowAssert(info == 0, "error in call to LAPACKE_dgetrf");
#endif
  }
  if (n2 == 0) {
    timer_factor_dla[level] = clockIt() - startTime;
    return;
  }
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    int matrix_layout = LAPACK_COL_MAJOR;
    MKL_INT info = LAPACKE_dgetrs(matrix_layout, 'N', n1, n2, A11[level].data(), n1,
                                  ipiv[level].data(), A12[level].data(), n1);
    ThrowAssert(info == 0, "error in call to LAPACKE_dgetrs");
    double alpha(-1), beta(1);
    CBLAS_LAYOUT layout = CblasColMajor;
    cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n2, n2, n1, alpha,
                A21[level].data(), n2, A12[level].data(), n1, beta,
                A22[level].data(), n2);
#endif
    timer_factor_dla[level] = clockIt() - startTime;
  }
  // recall that we use row-major ordering for sc --> convert A22 accordingly
  sc[level+1].resize(n2*n2);
  convert_to_row_major(A22[level], n2, n2, sc[level+1]);
}

void D3Solver::eliminate_separator_rhs(const int level)
{
  const double startTime = clockIt();
  const int n1 = n1a[level];
  const int n2 = n2a[level];
  double* rhs = AS_rhs[level].data();
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    const int n = n1 + n2; // leading dimension of AS_rhs[level]
    int matrix_layout = LAPACK_COL_MAJOR;
    const int num_rhs = 1;
    int info = LAPACKE_dgetrs(matrix_layout, 'N', n1, num_rhs, A11[level].data(), n1,
                              ipiv[level].data(), rhs, n);
    ThrowAssert(info == 0, "error in call to LAPACKE_dgetrs");
#endif
  }
  if (n2 == 0) {
    timer_solve_dla[level] += clockIt() - startTime;
    return;
  }
  rhs_sc[level+1].resize(n2);
  double* C = rhs_sc[level+1].data();
  for (int i=0; i<n2; i++) C[i] = rhs[n1+i];
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    double alpha(-1), beta(1);
    const int num_rhs = 1;
    const int n = n1 + n2; // leading dimension of AS_rhs[level]
    CBLAS_LAYOUT layout = CblasColMajor;
    cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n2, num_rhs, n1, alpha,
                A21[level].data(), n2, rhs, n, beta, C, n2);
#endif
  }
  timer_solve_dla[level] += clockIt() - startTime;
}

void D3Solver::output_time(const std::string & message,
                           const double time) const
{
  double time_max;
  MPI_Allreduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, comm);
  if (myPID == 0) {
    std::cout << message << time_max << std::endl;
  }
}

void D3Solver::output_timers() const
{
  std::string message = "max pardiso symbolic time = ";
  output_time(message, timer_pardiso_symbolic);
  message = "max pardiso numeric time  = ";
  output_time(message, timer_pardiso_numeric);
  message = "time to gather sub matrices = ";
  output_time(message, timer_gather_matrices);
  /*
  for (int i=0; i<num_level; i++) {
    message = "max factor time (level " + std::to_string(i) + ") = ";
    output_time(message, timer_factor[i]);
    message = "  dense linear algebra only = ";
    output_time(message, timer_factor_dla[i]);
  }
  for (int i=0; i<num_level; i++) {
    message = "max solve time (level " + std::to_string(i) + ") = ";
    output_time(message, timer_solve[i]);
    message = "  dense linear algebra only = ";
    output_time(message, timer_solve_dla[i]);
  }
  */
}

inline double D3Solver::clockIt() const
{
  struct timeval start;
  gettimeofday(&start, NULL);
  double duration = 
    (double)(start.tv_sec + start.tv_usec/1000000.0);
  return duration;
}

