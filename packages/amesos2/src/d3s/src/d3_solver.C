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
#include "KokkosBlas3_gemm.hpp"
#include "KokkosSparse_spmv.hpp"
#include "Amesos2.hpp"

// Direct Domain Decomposition Solver, a sparse distributed memory direct solver based on
// domain decomposition concepts.
// Author: Clark R. Dohrmann

D3Solver::D3Solver(MPI_Comm commIn) :
  comm(commIn)
{
#ifndef USE_INTEL_PARDISO
  ThrowAssert(true, 0, "d3_solver currently requires an Intel build with MKL/Pardiso");
#endif
  MPI_Comm_rank(comm, &myPID);
  MPI_Comm_size(comm, &numProcs);
  ThrowAssert(true, numProcs > 1, "d3_solver currently must be run on at least 2 MPI processes");

  num_threads = 1;

  // option for pardiso to enhance stability
  robust_option = true;

  // ordering option
  matching_option = 0;
  reorder_option = 2;

  // message level
  msg_level = 0;
  debug_level_interior = 0;

  // interior solver
  solvername = "";
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

void D3Solver::setNumThreads(const int num_threadsIn) {
  num_threads = num_threadsIn;
}

void D3Solver::setOrderingOption(const int matching_optionIn, const int reorder_optionIn) {
  matching_option = matching_optionIn;
  reorder_option = reorder_optionIn;
}

void D3Solver::setVerbose(const int msg_levelIn, const int debug_levelIn) {
  msg_level = msg_levelIn;
  debug_level_interior = debug_levelIn;
}

void D3Solver::setInteriorSolverName(const std::string solvername_in) {
    solvername = solvername_in;
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
  ThrowAssert(true, index != -1, "index not found");
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
    std::cout << " Invalid getLocalID: myPID, gID = " << myPID << " " << gID << std::endl;
  }
  ThrowAssert(true, valid, "index not found");
  return std::distance(array, it);
}

void D3Solver::gatherScatterSol(std::vector<double> & sol,
                                std::vector<double> & solAll) const
{
  const int numRows = sol.size();
  ThrowAssert(true, numRows == numRows_proc, "incompatible number of rows");
  std::vector<int> numRowsProc;
  const int root = 0;
  if (myPID == root) {
    numRowsProc.resize(numProcs);
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
                                std::vector<int> &rowperm,
                                std::vector<int> &irowperm,
                                std::vector<idx_t> & rowBeginMetis,
                                std::vector<idx_t> & columnsMetis,
                                std::vector<std::pair<int,int>> & additional_edges)
{
  const int numRows = rowBegin.size() - 1;
  std::vector<int> sort_perm;
  std::vector<int> sortedCols(numRows);
  if (matching_option == 0) {
    // No matching
    for (int i=0; i<numRows; i++) rowperm[i] = i;
  } else {
    // Max-cardinarity matching
    int nMatch = 0;
    double work;
    double maxwork = 0.0;
    std::vector<int> workspace(5*numRows, 0);
    int *perm  = const_cast <int*> (rowperm.data());
    int *iwork = const_cast <int*> (workspace.data());
    if (matching_option == 1) {
      // Compute max-cardinarity matching for col-permutation
      int *col_ptr = const_cast <int*> (rowBegin.data());
      int *row_idx = const_cast <int*> (columns.data());
      /*if (sort_columns_before_matching) {
        ns_sorted.resize(nnz, 0);
        for (int i=0; i<numRows; i++) {
          const int num_cols = rowBegin[i+1] - rowBegin[i];
          int index = rowBegin[i];
          for (int j=0; j<num_cols; j++) sortedCols[j] = columnsMetis[index++];
          std::sort(sortedCols.begin(), sortedCols.begin() + num_cols);
          index = rowBegin[i];
          for (int j=0; j<num_cols; j++) columns_sorted[index++] = sortedCols[j];
        }
        row_idx = const_cast <int*> (columns_sorted.data());
      }*/
      nMatch = trilinos_btf_maxtrans(numRows, numRows, col_ptr, row_idx, maxwork, &work, perm, iwork);
    } else {
      // Compute max-cardinarity matching for row-permutation
      std::vector<int> rowBeginT, columnsT;
      getGraphTranspose(rowBegin, columns, rowBeginT, columnsT);

      int *col_ptr = const_cast <int*> (rowBeginT.data());
      int *row_idx = const_cast <int*> (columnsT.data());
      nMatch = trilinos_btf_maxtrans(numRows, numRows, col_ptr, row_idx, maxwork, &work, perm, iwork);
    }
    printf( " nMatch = %d / %d\n",nMatch,numRows );
  }
  // inverse-matching
  for (int i=0; i<numRows; i++) irowperm[rowperm[i]] = i;
//#define MATRIX_OUT
#ifdef MATRIX_OUT
  {
    FILE *fp = fopen("G.dat", "w");
    for (int i=0; i<numRows; i++) {
      for (int k=rowBegin[i]; k<rowBegin[i+1]; k++)
        fprintf(fp,"%d %d\n",i,columns[k]);
    }
    fclose(fp);
  }
#endif
  // remove edges to self (row or col matching is incorporated)
  int numTerms = rowBegin[numRows];
  std::vector<int> rowBeginG(numRows+1, 0);
  std::vector<int> columnsG(numTerms);
  numTerms = 0;
  for (int i=0; i<numRows; i++) {
    int row = (matching_option > 1 ? rowperm[i] : i); //row-matching
    for (int j=rowBegin[row]; j<rowBegin[row+1]; j++) {
      const int col = (matching_option == 1 ? irowperm[columns[j]] : columns[j]); //col-matching
      if (col != i) {
        columnsG[numTerms++] = col;
      }
    }
    rowBeginG[i+1] = numTerms;
  }
  columnsG.resize(numTerms);
#ifdef MATRIX_OUT
  {
    FILE *fp = fopen("G_2.dat", "w");
    for (int i=0; i<numRows; i++) {
      for (int k=rowBeginG[i]; k<rowBeginG[i+1]; k++)
        fprintf(fp,"%d %d\n",i,columnsG[k]);
    }
    fclose(fp);
  }
#endif
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
  for (int i=0; i<numRows; i++) {
    const int num_cols = rowBeginMetis[i+1] - rowBeginMetis[i];
    int index = rowBeginMetis[i];
    for (int j=0; j<num_cols; j++) sortedCols[j] = columnsMetis[index++];
    std::sort(sortedCols.begin(), sortedCols.begin() + num_cols);
    index = rowBeginMetis[i];
    for (int j=0; j<num_cols; j++) columnsMetis[index++] = sortedCols[j];
  }
  if (num_additional_edges > 0) {
    if (msg_level > 0) {
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
                                const std::vector<idx_t> & iperm,
                                std::vector<int> & out_rowSubIDs) const
{
  const int numRows = iperm.size();
  out_rowSubIDs.resize(numRows);
  const int numSubs = node_begin.size() - 1;
  for (int i=0; i<numRows; i++) {
    int id = -1;
    // i (input for METIS, after matching) -> row (post-order, METIS)
    //  iperm: ith row of original is iperm[i] row after ND
    int row = iperm[i];
    for (int j=0; j<numSubs; j++) {
      if ((row >= node_begin[j]) && (row < node_begin[j+1])) {
        id = j;
        break;
      }
    }
    ThrowAssert(true, id != -1, "row not found in bounds of node_begin");
    // node_sub_id maps from post-order (METIS) to bottom-up
    // subdomain ID for i in oridinal ordering
    out_rowSubIDs[i] = node_sub_id[id];
  }
}

void D3Solver::checkRowSubIDs(const std::vector<int> & in_rowSubIDs,
                              const std::vector<idx_t> & rowperm,
                              const std::vector<int> & rowBegin,
                              const std::vector<int> & columns) const
{
  int maxSubID = 0;
  std::vector<std::vector<int>> subI(numProcSolver);
  for (size_t i=0; i<in_rowSubIDs.size(); i++) {
    int sub = in_rowSubIDs[i];
    if (sub >= 0) {
      // push ith row into subdomain, if not separator
      subI[sub].push_back(i);
    }
    sub = std::abs(in_rowSubIDs[i]);
    if (sub > maxSubID) maxSubID = sub;
  }
  char msg[100];
  const int num_group = maxSubID + 1;
  // numProcSolver == # of interior subdomains
  for (int i=0; i<numProcSolver; i++) {
    std::vector<int> adj_sub(num_group, 0);
    for (size_t j=0; j<subI[i].size(); j++) {
      // subI stores row ids after matching
      // rowperm: ith row after matching is rowperm[i] of original
      const int row = rowperm[subI[i][j]];
      for (int k=rowBegin[row]; k<rowBegin[row+1]; k++) {
        const int col = columns[k];
        const int sub2 = std::abs(in_rowSubIDs[col]);
        adj_sub[sub2] = 1;
      }
    }
    // checking if this interior is not connected to another interior
    //  (numProcSolver == # of interior subdomains)
    for (int j=0; j<numProcSolver; j++) {
      if (j == i) {
        sprintf(msg, "%d: subdomain %d has no interior unknowns", myPID,i);
        ThrowAssert(true, adj_sub[j] == 1, msg);
      }
      else {
        sprintf(msg, "%d:%d: partitioning error for subdomain %d", myPID,i,j);
        ThrowAssert(true, adj_sub[j] == 0, msg);
      }
    }
  }
}

void D3Solver::get_separators(const std::vector<int> & in_rowSubIDs,
                              std::vector<int> & in_sepIDs,
                              std::vector<int> & in_sepBegin,
                              std::vector<int> & in_sepRows) const
{
  const int numRows = in_rowSubIDs.size();
  const int num_sep = numProcSolver - 1;
  in_sepIDs.resize(num_sep);
  for (int i=0; i<num_sep; i++) {
    in_sepIDs[i] = numProcSolver + i;
  }
  std::vector<int> count(2*numProcSolver, 0);
  for (int i=0; i<numRows; i++) {
    if (in_rowSubIDs[i] < 0) {
      const int sepID = -in_rowSubIDs[i];
      count[sepID]++;
    }
  }
  if (msg_level > 0) {
    std::cout << "number of separators = " << num_sep << std::endl;
  }
  in_sepBegin.resize(num_sep+1, 0);
  for (int i=0; i<num_sep; i++) {
    const int sep = in_sepIDs[i];
    in_sepBegin[i+1] = in_sepBegin[i] + count[sep];
    count[i] = 0;
  }
  const int num_terms = in_sepBegin[num_sep];
  in_sepRows.resize(num_terms);
  for (int i=0; i<numRows; i++) {
    if (in_rowSubIDs[i] < 0) {
      const int sepID = -in_rowSubIDs[i];
      const int index = getLocalID(sepID, in_sepIDs);
      const int index2 = in_sepBegin[index] + count[index];
      in_sepRows[index2] = i;
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
      ThrowAssert(true, col < numRows, "graph not square");
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
  ThrowAssert(true, proc != -1, "processor not found");
  return proc;
}

void D3Solver::assign_graph(const std::vector<int> & rowBegin,
                            const std::vector<int> & columns,
                            const std::vector<int> & extraEdges) {
  assign_graph(rowBegin, columns, rowBegin, columns, extraEdges);
}

void D3Solver::assign_graph(const std::vector<int> & rowBegin_in, // original input distributed matrix
                            const std::vector<int> & columns_in,
                            const std::vector<int> & rowBegin,    // after matching
                            const std::vector<int> & columns,
                            const std::vector<int> & extraEdges)
{
  num_extra_edges = extraEdges.size() / 2;
  if (num_extra_edges == 0 && matching_option == 0) {
    rowBeginPtr = &rowBegin;
    columnsPtr = &columns;
  }
  else {
    // update graph to be the symmetrized version by adding extra edges
    update_graph(rowBegin, columns, extraEdges);
    rowBeginPtr = &rowBeginUse; // output (after matching then symmetrized)
    columnsPtr = &columnsUse;
    rowBeginOrig = rowBegin_in; // input (original)
  }
}

void D3Solver::scatter_additional_edges(const std::vector<std::pair<int,int>> & additional_edges,
                                        std::vector<int> & extraEdges)
{
  int root(0);
  std::vector<int> numRowsAll, count, displs;
  if (myPID == root) {
    numRowsAll.resize(numProcs);
    count.resize(numProcs, 0);
    displs.resize(numProcs, 0);
  }
  MPI_Gather(&numRows_proc, 1, MPI_INT, numRowsAll.data(), 1, MPI_INT, root, comm);
  std::vector<int> row_col_pair_send;
  if (myPID == root) {
    for (int i=1; i<numProcs; i++) {
      displs[i] = displs[i-1] + numRowsAll[i-1];
    }
    // additional edges are for input to METIS (after Matching, but before ND)
    const int num_additional_edges = additional_edges.size();
    row_col_pair_send.resize(2*num_additional_edges);
    int first_proc(0), first_row(0), index(0);
    for (int i=0; i<num_additional_edges; i++) {
      int row = additional_edges[i].first;   // after row matching
      int col = additional_edges[i].second;  // additional elemnts to symmetrize matrix after matching, after col matching
      row_col_pair_send[index++] = row;
      row_col_pair_send[index++] = col;
      const int proc = get_proc_for_row(row, numRowsAll, numProcs, first_proc, first_row);
      count[proc] += 2;
    }
  }
  int num_extra;
  MPI_Scatter(count.data(), 1, MPI_INT, &num_extra, 1, MPI_INT, root, comm);
  extraEdges.resize(num_extra);
  if (myPID == root) {
    for (int i=1; i<numProcs; i++) {
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
  const int num_extra = extraEdges.size() / 2;
  for (int i=0; i<num_extra; i++) {
    const int row = extraEdges[2*i];
    const int local_row = row - startGID;
    ThrowAssert(true, (local_row >= 0) && (local_row < numRows_proc), "row is out of range");
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
  // original distributed matrix (in 1D block row)
  for (int i=0; i<numRows_proc; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int index = rowBeginUse[i] + count[i];
      columnsUse[index] = columns[j];;
      count[i]++;
    }
  }
  for (int i=0; i<num_extra; i++) {
    const int row = extraEdges[2*i];
    const int col = extraEdges[2*i+1];
    const int local_row = row - startGID;
    const int index = rowBeginUse[local_row] + count[local_row];
    columnsUse[index] = col;
    count[local_row]++;
  }
}

void D3Solver::getRowSubIDs(const std::vector<int> & rowBegin, // original
                            const std::vector<int> & columns)  // original
{
  // Gather graph to root: (rowBegin, columns) in 1D block row -> (rowBeginRoot, columnsRoot) gathered
  GatherToRootSimple gatherer(rowBegin, columns, comm);
  gatherer.initialize();
  const std::vector<int> & rowBeginRoot = gatherer.getRowBeginRoot(); // original
  const std::vector<int> & columnsRoot = gatherer.getColumnsRoot();   // original
  std::vector<std::pair<int,int>> additional_edges;
  int numRows, numSep, numTerms, numRowsB;

  if (myPID == 0) {
    numRows = rowBeginRoot.size() - 1;
    std::vector<idx_t> rowBeginMetis, columnsMetis;
    permMatching.resize(numRows, 0);
    ipermMatching.resize(numRows, 0);
    // Prepar graph for calling Metis
    getGraphForMetis(rowBeginRoot, columnsRoot, permMatching, ipermMatching,
                     rowBeginMetis, columnsMetis, additional_edges);
    // calling Metis
    std::vector<idx_t> options(METIS_NOPTIONS), perm(numRows), iperm(numRows), sizes(2*numProcSolver);
    int info = METIS_SetDefaultOptions(options.data());
    ThrowAssert(true, info == METIS_OK, "METIS_SetDefaultOptions failed");
    idx_t* vwgt = nullptr;
    if (true) {
      // sort for debugging
      std::vector<int> sortedCols(numRows);
      std::vector<idx_t> columnsSorted(columnsMetis.size());
      for (int i=0; i<numRows; i++) {
        const int num_cols = rowBeginMetis[i+1] - rowBeginMetis[i];
        int index = rowBeginMetis[i];
        for (int j=0; j<num_cols; j++) sortedCols[j] = columnsMetis[index++];
        std::sort(sortedCols.begin(), sortedCols.begin() + num_cols);
        index = rowBeginMetis[i];
        for (int j=0; j<num_cols; j++) columnsSorted[index++] = sortedCols[j];
      }
      METIS_NodeNDP(numRows, rowBeginMetis.data(), columnsSorted.data(), vwgt,
                    numProcSolver, options.data(), perm.data(), iperm.data(), 
                    sizes.data());
#ifdef MATRIX_OUT
      {
        FILE *fp = fopen("G_Metis.dat", "w");
        for (int i=0; i<numRows; i++) {
          for (int k=rowBeginMetis[i]; k<rowBeginMetis[i+1]; k++)
            fprintf(fp,"%d %d\n",columnsSorted[k],i);
        }
        fclose(fp);
      }
#endif
    } else {
      METIS_NodeNDP(numRows, rowBeginMetis.data(), columnsMetis.data(), vwgt,
                    numProcSolver, options.data(), perm.data(), iperm.data(), 
                    sizes.data());
#ifdef MATRIX_OUT
      {
        FILE *fp = fopen("G_Metis.dat", "w");
        for (int i=0; i<numRows; i++) {
          for (int k=rowBeginMetis[i]; k<rowBeginMetis[i+1]; k++)
            fprintf(fp,"%d %d\n",columnsMetis[k],i);
        }
        fclose(fp);
      }
#endif
    }
#ifdef MATRIX_OUT
    if (matching_option) {
      printf("\n");
      for (int i=0; i<2*numProcSolver; i++) printf("%d\n",sizes[i]);
      FILE *fp = fopen("p_metis.dat", "w");
      for (int i=0; i<numRows; i++) fprintf(fp, "%d %d %d\n",i,perm[i],iperm[i] );
      fclose(fp);

      fp = fopen("G_orig.dat", "w");
      for (int i=0; i<numRows; i++) {
        for (int k=rowBeginRoot[i]; k<rowBeginRoot[i+1]; k++)
          fprintf(fp,"%d %d\n",i,columnsRoot[k]);
      }
      fclose(fp);
    }
#endif
    int lowerIndex = 2*numProcSolver - 2;
    int numSepLevel(1), interfaceSize(0);
    const int numLevel = std::log2(numProcSolver) + 1;
    for (int level=0; level<numLevel; level++) {
      if (msg_level > 0) {
        if (level < numLevel-1) {
          std::cout << "separator sizes for level " << level << ": \n";
        }
        else {
          std::cout << "subdomain interior sizes:\n";
        }
      }
      for (int i=0; i<numSepLevel; i++) {
        const int sepSize = sizes[lowerIndex--];
        if (msg_level > 0) std::cout << sepSize << std::endl;
        if (level < numLevel-1) interfaceSize += sepSize;
      }
      numSepLevel *= 2;
    }
    if (msg_level > 0) {
      std::cout << "total interface size = " << interfaceSize << std::endl;
    }
    std::vector<int> level, location;
    getLevelsAndLocations(numProcSolver, level, location);
    const int num_level_nd = std::log2(numProcSolver) + 1;
    std::vector<int> start_level(num_level_nd);
    int delta = numProcSolver;
    for (int i=1; i<num_level_nd; i++) {
      start_level[i] = start_level[i-1] + delta;
      delta /= 2;
    }

    // Map from post-order (METIS) to bottom-up ordering of nested-dissection tree of subdomains, 
    //  e.g., [0, 1 | -4 | 2, 3 | -5 | -6] with 4 MPIs
    const int num_node = location.size();
    std::vector<int> node_size(num_node), node_sub_id(num_node), node_begin(num_node+1, 0);
    for (int i=0; i<num_node; i++) {
      node_size[i] = sizes[start_level[level[i]] + location[i]];
      node_sub_id[i] = start_level[level[i]] + location[i];
      if (level[i] > 0) node_sub_id[i] *= -1;
      node_begin[i+1] = node_begin[i] + node_size[i];
    }

    // Mapping of rows to subdomains (interior or separators)
    //   perm: ith row after ND is perm[i] of original
    //  iperm: ith row of original is iperm[i] row after ND
    extractRowSubIDs(node_begin, node_sub_id, iperm, rowSubIDs);
#ifdef MATRIX_OUT
    {
      FILE *fp = fopen("rowperm.dat", "w");
      for (int i=0; i<numRows; i++) {
        fprintf(fp,"%d %d %d\n",i,permMatching[i],ipermMatching[i]);
      }
      fclose(fp);
    }
#endif

    // Checking (to make sure each iterior is not connected to another interior)
    checkRowSubIDs(rowSubIDs, permMatching, rowBeginRoot, columnsRoot);

    //
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
        int row = sepRows[j]; // after row-matching
        if (matching_option == 2) {
          row = permMatching[sepRows[j]]; // to original
        }
        for (int k=rowBeginRoot[row]; k<rowBeginRoot[row+1]; k++) {
          int col = columnsRoot[k]; // original
          if (matching_option == 1) {
            col = ipermMatching[col]; // after col-matching
          }
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

  // symmetrize the original local matrix
  std::vector<int> extraEdges; // extra edges on each proc (to make it structurally-symmetric)
  scatter_additional_edges(additional_edges, extraEdges);

  // global DoFs
  MPI_Bcast(&numRows, 1, MPI_INT, 0, comm);
  numRows_global = numRows;

  // symmetrize local graph
  if (matching_option == 0) {
    // add additional edges to the original distributed matrix (1D block row) to make it symmetric
    assign_graph(rowBegin, columns, extraEdges);
  } else {
    if (myPID != 0) {
      permMatching.resize(numRows, 0);
      ipermMatching.resize(numRows, 0);
    }
    MPI_Bcast( permMatching.data(), numRows, MPI_INT, 0, comm);
    MPI_Bcast(ipermMatching.data(), numRows, MPI_INT, 0, comm);

    // row-distribution
    fstRows.resize(numProcs+1, 0);
    MPI_Allgather(&startGID, 1, MPI_INT, fstRows.data(), 1, MPI_INT, comm);
    fstRows[numProcs] = numRows_global;

    int nnz = rowBegin[numRows_proc];
    if (matching_option == 1) {
      // Apply column matching
      rowBeginRe.resize(numRows_proc+1, 0);
      columnsRe.resize(nnz, 0);
      valuesRe.resize(nnz, 0);
      for (int i=0; i<numRows_proc; i++) {
        for (int k=rowBegin[i]; k<rowBegin[i+1]; k++) {
          int col = ipermMatching[columns[k]]; // after col-matching
          columnsRe[k] = col;
        }
        rowBeginRe[i+1] = rowBegin[i+1];
      }
    } else {
      // Redistribute to apply row matching
      // count nnz to send
      sendcounts.resize(numProcs, 0);
      for (int i=0; i<numRows_proc; i++) {
        int row = ipermMatching[startGID+i]; // perm original i to row
        for (int p=0; p<numProcs; p++) {
          if (row >= fstRows[p] && row < fstRows[p+1]) {
            sendcounts[p] += rowBegin[i+1]-rowBegin[i];
            break;
          }
        }
      }
      senddispls.resize(numProcs+1, 0);
      for (int p=0; p<numProcs; p++) {
        sendcounts[p] *= 2;
        senddispls[p+1] = senddispls[p] + sendcounts[p];
      }

      // fill send-buffer
      std::vector<int> sendbuf;
      sendbuf.resize(2*nnz, 0);
      for (int i=0; i<numRows_proc; i++) {
        int row = ipermMatching[startGID+i]; // perm origina i to row
        for (int p=0; p<numProcs; p++) {
          if (row >= fstRows[p] && row < fstRows[p+1]) {
            nnz = senddispls[p];
            for (int k=0; k<rowBegin[i+1]-rowBegin[i]; k++) {
              int col = columns[rowBegin[i]+k];
              sendbuf[nnz + 2*k+0] = row;
              sendbuf[nnz + 2*k+1] = col;
            }
            senddispls[p] += 2*(rowBegin[i+1]-rowBegin[i]);
            break;
          }
        }
      }
      // shift back
      for (int p=numProcs; p>0; p--) {
        senddispls[p] = senddispls[p-1];
      }
      senddispls[0] = 0;

      // setup counts/displs to receive
      recvcounts.resize(numProcs,   0);
      recvdispls.resize(numProcs+1, 0);
      MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
      for (int p=0; p<numProcs; p++) {
        recvdispls[p+1] = recvdispls[p] + recvcounts[p];
      }

      // communicate !!
      nnz = recvdispls[numProcs];
      std::vector<int> recvbuf;
      recvbuf.resize(nnz, 0);
      MPI_Alltoallv(sendbuf.data(), sendcounts.data(), senddispls.data(), MPI_INT,
                    recvbuf.data(), recvcounts.data(), recvdispls.data(), MPI_INT,
                    comm);

      // put it into CSR
      rowBeginRe.resize(numRows_proc+1, 0);
      columnsRe.resize(nnz, 0);
      valuesRe.resize(nnz, 0);
      for (int i=0; i<nnz; i+=2) {
        int row = recvbuf[i]-startGID;
        rowBeginRe[1+row]++;
      }
      for (int i=0; i<numRows_proc; i++) rowBeginRe[i+1] += rowBeginRe[i];
      for (int i=0; i<nnz; i+=2) {
        int row = recvbuf[i]-startGID;
        int col = recvbuf[i+1];
        columnsRe[rowBeginRe[row]] = col;
        rowBeginRe[row] ++;
      }
      for (int i=numRows_proc; i>0; i--) rowBeginRe[i] = rowBeginRe[i-1];
      rowBeginRe[0] = 0;
    }

    // now add additional edges to make it symmetric
    assign_graph(rowBegin, columns, rowBeginRe, columnsRe, extraEdges);
  }

  // communicate rest of parameters
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
  ThrowAssert(true, my_recv_PIDs.size() == data_recv.size(), "incompatible sizes");
  ThrowAssert(true, my_send_PIDs.size() == data_send.size(), "incompatible sizes");
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
  // > (rowBegin, columns) stored original local distribued matrix in 1D block row
  //    after symmetrization, no matching nor ND applied
#ifdef MATRIX_OUT
  {
    char filename[100];
    sprintf(filename, "lG_%d.dat", myPID);
    FILE *fp = fopen(filename, "w");
    for (int i=0; i<numRows; i++) {
      for (int k=rowBegin[i]; k<rowBegin[i+1]; k++)
        fprintf(fp,"%d %d\n",i,columns[k]);
    }
    fclose(fp);
  }
#endif
  for (int i=0; i<numRows; i++) {
    const int gID = startGID + i;   // original
    const int sub = rowSubIDs[gID]; //
    if (sub >= 0) { // row is in interior of a subdomain
      int nnzRow = rowBegin[i+1] - rowBegin[i];
      row_GIDs[sub].push_back(gID);
      num_terms_sub[sub] += nnzRow;
    }
    else { // row is on the interface
      std::unordered_set<int> sub_set2;
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int gID2 = columns[j];         // called with rowBeginPtr = after matching
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
    const int gID = startGID + i;      // original
    const int sub = rowSubIDs[gID];
    if (sub >= 0) { // row is in interior of a subdomain
      const int local_sub = getLocalID(sub, activeSubs);
      const int row_sub = num_rows_sub[local_sub];
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int index = num_terms_sub[local_sub];
        const int gID2 = columns[j];         // original
        column_GIDs_send[local_sub][index] = gID2;
        values_send_index[local_sub][index] = j;
        num_terms_sub[local_sub]++;
        column_counts_send[local_sub][row_sub]++;
      }
      num_rows_sub[local_sub]++;
    }
    else { // row is on the interface
      std::unordered_set<int> sub_set_local;
      for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
        const int gID2 = columns[j];         // original
        const int sub2 = rowSubIDs[gID2];
        if (sub2 >= 0) {
          const int local_sub = getLocalID(sub2, activeSubs);
          sub_set_local.insert(local_sub);
          const int row_sub = num_rows_sub[local_sub];
          column_counts_send[local_sub][row_sub]++;
          const int index = num_terms_sub[local_sub];
          column_GIDs_send[local_sub][index] = gID2;
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
        const int row_sep = num_rows_sep[local_sep_active];
        for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
          const int gID2 = columns[j];
          const int validRow2 = determine_valid_row(gID2, separators);
          if (validRow2) {
            const int index = num_terms_sep[local_sep_active];
            column_GIDs_send[local_sep_active][index] = columns[j];
            //            values_send_B[local_sep_active][index] = values[j];
            values_send_B_index[level][local_sep_active][index] = j;
            num_terms_sep[local_sep_active]++;
            column_counts_send[local_sep_active][row_sep]++;
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
            const int row_sep = num_rows_sep[local_sep2_active];
            column_counts_send[local_sep2_active][row_sep]++;
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
    ThrowAssert(true, rowGIDs[i] > rowGIDs[i-1], "rowGIDs not in ascending order");
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
        ThrowAssert(true, check, "should not be off-diagonal A_BB entries here");
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
    ThrowAssert(true, rowSubIDs[rowGID] < 0, "logic error a");
    for (int j=rowBegin[row]; j<rowBegin[row+1]; j++) {
      const int col = columns[j];
      const int colGID = rowGIDs[col];
      if (colGID != rowGID) {
        ThrowAssert(true, rowSubIDs[colGID] >= 0, "logic error b");
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

void D3Solver::extractMatrixStructures(const int level,
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
    ThrowAssert(true, rowGIDs[i] > rowGIDs[i-1], "rowGIDs not in ascending order");
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
        count[local_row]++;
      }
    }
  }
}

void D3Solver::extractRhs(const int level)
{
  std::vector<double> & rhs = rhs_sep[level];
  const std::vector<std::vector<double>> & rhs_recv_l = rhs_recv_sep[level];
  const std::vector<std::vector<int>> & index_map = rhs_recv_sep_index[level];
  const int num_recvs = index_map.size();
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<index_map[i].size(); j++) {
      const int index = index_map[i][j];
      rhs[index] = rhs_recv_l[i][j];
    }
  }
}

void D3Solver::extractMatrixValues(const int level)
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
                              std::vector<int> & in_rowBeginSub,
                              std::vector<int> & in_columnsSub,
                              std::vector<double> & in_valuesSub,
                              std::vector<int> & rowGIDsSub)
{
  // subdomain matrices [A_{II} A_{IB}; A_{BI} 0]
  // redistribution from 2D block row to nested-dissection
  std::vector<std::vector<int>> num_rows_recv, row_GIDs_recv,
    column_counts_recv, column_GIDs_recv;

  // communicate required data (into buffer)
  phase1(rowBegin, columns, num_rows_recv, row_GIDs_recv,
         column_counts_recv, column_GIDs_recv);

  // split the received data into sub-matrices
  generateSubMatrices(row_GIDs_recv, column_counts_recv, column_GIDs_recv,
                      in_rowBeginSub, in_columnsSub, in_valuesSub, rowGIDsSub);
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
  if (msg_level < 2) return;
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
        ThrowAssert(true, col != i, "diagonal on interface should not be there");
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
  ThrowAssert(true, my_node != -1, "node not found");
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
  ThrowAssert(true, best_node != -1, "logic error");
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
    ThrowAssert(true, target_copy.size() == targetMPIs.size(), "duplicate MPIs in targetMPIs");
    if (msg_level > 0) {
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
  int numProcSolver_out = numProcSolver_in;
  if (numProcSolver_out > numProcs) {
    if (msg_level > 0 && myPID == 0) {
      std::cout << std::endl 
                << std::endl << " ** User-specified number of subdomains is greater "
                << std::endl << " ** than the number of MPI processes"
                << std::endl << " ** resetting to be " << numProcs << std::endl
                << std::endl;
    }
    numProcSolver_out = numProcs;
  }
  int logInput = static_cast<int>(std::log2(numProcSolver_out));
  if (std::pow(2, logInput) != numProcSolver_out) {
    if (msg_level > 0 && myPID == 0) {
      std::cout << " ** numProcSolver should be a power of 2" << std::endl;
      std::cout << " ** resetting to next lower power of 2" << std::endl
                << " ** resetting to be " << std::pow(2, logInput) << std::endl
                << std::endl;
    }
    numProcSolver_out = std::pow(2, logInput);
  }
  return numProcSolver_out;
}

int D3Solver::initialize(const std::vector<int> & rowBegin_in,
                         const std::vector<int> & columns_in,
                         const int startGID_in,
                         const int numProcSolver_in)
{
  if (msg_level > 0) {
    MPI_Barrier(comm);
    if (myPID == 0) {
      printf( "\n -- D3Solver::initialize -- \n" ); fflush(stdout);
    }
  }
  startGID = startGID_in;
  numProcSolver = setNumProcSolver(numProcSolver_in);
  num_level = std::log2(numProcSolver);
  numRows_proc = rowBegin_in.size() - 1;
  // call METIS to get nested-dissection
  getRowSubIDs(rowBegin_in, columns_in);
  getProcName();

  // rowBeginPtr (after symmetrization of the original matrix)
  const std::vector<int> & rowBegin = *rowBeginPtr;
  const std::vector<int> & columns = *columnsPtr;
  assignTargetMPIs(rowBegin);

  // redistribute to generate DD
  std::vector<int> rowGIDsSub;
  getSubMatrices(rowBegin, columns, rowBeginSub, columnsSub, valuesSub,
                 rowGIDsSub);
  const int num_rows_sub = rowBeginSub.size() - 1;
  resize_vectors();

  // calling Pardiso Initialize
  int r_val = 0;
  double startTime = clockIt();
  if (num_rows_sub > 0) {
    // at this point the matrix is structurally symmetric by construction
    structurally_symmetric = 1;
    sort_and_add_zero_diags(rowBeginSub, columnsSub);
#ifdef MATRIX_OUT
    {
      char filename[100];
      sprintf(filename, "lT_%d.dat", myPID);
      FILE *fp = fopen(filename, "w");
      for (int i=0; i<num_rows_sub; i++) {
        for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++)
          fprintf(fp,"%d %d\n",i,columnsSub[k]);
      }
      fclose(fp);
    }
#endif
    if (solvername == "") {
      // PardisoMKL initialize
#ifdef USE_INTEL_PARDISO
      bool verbose = (myPID == 0 && msg_level > 0);
      const int num_rows_subB = rowsBSub.size();
      pardiso_solver.setMessageLevel(debug_level_interior);
      r_val = pardiso_solver.initialize(num_rows_sub, rowBeginSub.data(), columnsSub.data(),
                                        num_rows_subB, rowsBSub.data(), num_threads,
                                        reorder_option, structurally_symmetric, robust_option,
                                        debug_level_interior, verbose);
#endif
    } else {
      // Amesos2 Symbolic factorization
      using comm_type = Teuchos::MpiComm<int>;

      size_t n = rowBeginSub.size()-1;
      int n2 = rowsBSub.size();
      // mark separtor nodes
      Kokkos::resize(m_parts, n);
      Kokkos::deep_copy(m_parts, 0);
      for (int i=0; i<n2; i++) m_parts(rowsBSub[i]) = -1;

      n2 = 1;
      int n1 = 0;
      int nnzD = 0;
      int nnzF = 0;
      for (int i=0; i<n; i++) {
        if (m_parts(i) == 0) { // interior
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) {
            if (m_parts(columnsSub[k]) >= 0) nnzD ++;
          }
          m_parts(i) = n1;
          n1++;
        } else { // separator
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) {
            if (m_parts(columnsSub[k]) >= 0) nnzF++;
          }
          m_parts(i) = -n2;
          n2++;
        }
      }
      n2--;
      if (n2 != rowsBSub.size()) printf( " D3S: ERROR n2 mismatch(%d vs %d)\n",n2,int(rowsBSub.size()) );
      if (n1+n2 != n) printf( " D3S: ERROR n1 mismatch(%d vs %d)\n",n1,int(n-n2) );
      n2 = rowsBSub.size();

      // extract interior part of local subdomain
      // [D, G; H, S]
      Kokkos::resize(rowmap_view_D, n1+1);
      Kokkos::resize(colind_view_D, nnzD);
      Kokkos::resize(values_view_D, nnzD);

      Kokkos::resize(E_view, n1,n2);
      #ifdef D3S_DENSE_F
        Kokkos::resize(F_view, n2,n1);
      #else
        Kokkos::resize(rowmap_view_F, n2+1);
        Kokkos::resize(colind_view_F, nnzF);
        Kokkos::resize(values_view_F, nnzF);
      #endif
      Kokkos::resize(G_view, n1,n2);
      Kokkos::resize(S_view, n2,n2);
      nnzD = 0;
      nnzF = 0;
      rowmap_view_D(0) = 0;
      rowmap_view_F(0) = 0;
      for (int i=0; i<n; i++) {
        int row = m_parts(i);
        if (m_parts(i) >= 0) { // interior rows
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) {
            int col = m_parts(columnsSub[k]);
            if (col >= 0) {
              // interior, D
              colind_view_D(nnzD) = col;
              nnzD++;
            }
          }
          rowmap_view_D(row+1)=nnzD;
        } else {
          #ifndef D3S_DENSE_F
          row = -row-1;
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) {
            int col = m_parts(columnsSub[k]);
            if (col >= 0) {
              // interface, F
              colind_view_F(nnzF) = col;
              nnzF++;
            }
          }
          rowmap_view_F(row+1)=nnzF;
          #endif
        }
      }

      // wrap them into kokkos crsmatrix
      graph_t static_graph(colind_view_D, rowmap_view_D);
      crsmat_t crsmat("CrsMatrix", n1, values_view_D, static_graph);
#ifdef D3S_USE_KOKKOS_BACKEND
      // kokkos-backend for symbolic
      amesos2_solver = Amesos2::create<crsmat_t, mv_view_t>(solvername, Teuchos::rcpFromRef(crsmat));
#else
      // tpetra-backend
      // wrap it into tpetra crsmatrix
      int indexBase = 0;
      Teuchos::RCP<const comm_type> localComm = Teuchos::rcp(new comm_type(MPI_COMM_SELF));
      localMap = Teuchos::rcp(new map_type(n1, indexBase, localComm, Tpetra::GloballyDistributed));
      A = Teuchos::rcp(new MAT(crsmat, localMap,localMap));
      amesos2_solver = Amesos2::create<MAT,MV>(solvername, A);
#endif
      {
        Teuchos::ParameterList amesos2Params;
        amesos2Params.setName("Amesos2");
        Teuchos::ParameterList &solverParams = amesos2Params.sublist(solvername);
        //if (msg_level > 0) {
        //  solverParams.set("verbose",(myPID == 1));
        //}
        //solverParams.set("replace_tiny_pivot",true);
        amesos2_solver->setParameters(Teuchos::rcpFromRef(amesos2Params));
      }
      amesos2_solver->symbolicFactorization();
    }
  }
  timer_pardiso_symbolic = clockIt() - startTime;
  if (msg_level > 0) {
    printf( " %d: initialize(r_val=%d)\n",myPID, r_val ); fflush(stdout);
  }
  r_val = -std::abs(r_val); // making sure non-positive (error-code is negative)
  MPI_Allreduce(MPI_IN_PLACE, &r_val, 1, MPI_INT, MPI_MIN, comm);
  if (msg_level > 0) {
    printf( " => r_val=%d\n",r_val ); fflush(stdout);
  }
  if (r_val == 0) {
    int level = 0;
    sc_GIDs[0] = getRowGIDsSubB(rowGIDsSub);
    while (level < num_level) {
      initialize_schur_complement(level, rowBegin, columns, sc_GIDs[level], sc_GIDs[level+1]);
      level++;
    }
    columnsUse.resize(0);
    columnsUse.shrink_to_fit();
  }
  if (msg_level > 0) {
    MPI_Barrier(comm);
    if (myPID == 0) printf(" Initialize done\n\n");
  }
  return r_val;
}

void D3Solver::getSubMatrices(const std::vector<double> & values,
                              std::vector<double> & in_valuesSub)
{
  communicateMatrixValues(values);
  const int num_recvs = values_recv.size();
  for (int i=0; i<num_recvs; i++) {
    for (size_t j=0; j<values_recv[i].size(); j++) {
      const int index1 = index_map_sub[i][j];
      const int index2 = old_to_new_indices[index1];
      in_valuesSub[index2] = values_recv[i][j];
    }
  }
}

void D3Solver::assign_values(const std::vector<double> & values_in) {
  assign_values(rowBeginOrig, values_in);
}

void D3Solver::assign_values(const std::vector<int> & rowBegin_in,
                             const std::vector<double> & values_in)
{
  if (num_extra_edges == 0 && matching_option == 0) {
    valuesPtr = &values_in;
  }
  else {
    // Note: extra columns for structural symmetry (= padded zeros)
    //       are at the end of each row (and hence no need to assign values)
    for (int i=0; i<numRows_proc; i++) {
      int index = rowBeginUse[i];
      for (int j=rowBegin_in[i]; j<rowBegin_in[i+1]; j++) {
        valuesUse[index++] = values_in[j];
      }
    }
    valuesPtr = &valuesUse;
  }
}

int D3Solver::factorize(const std::vector<double> & values_in)
{
  int r_val = 0;
  if (msg_level > 0) {
    MPI_Barrier(comm);
    if (myPID == 0) {
      printf( "\n -- D3Solver::factorize -- \n" ); fflush(stdout);
    }
  }
  double startTime = clockIt();
  if (matching_option == 0) {
    // No matching
    assign_values(values_in);
  } else {
    if (matching_option == 1) {
      // assign values
      assign_values(rowBeginRe,values_in);
#ifdef MATRIX_OUT
      {
        char filename[100];
        sprintf(filename, "lW_%d.dat", myPID);
        FILE *fp = fopen(filename, "w");
        const int num_rows_sub = rowBeginSub.size() - 1;
        for (int i=0; i<num_rows_sub; i++) {
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++)
            fprintf(fp,"%d %d %.16e\n",i,columnsSub[k],valuesSub[k]);
        }
        fclose(fp);
      }
#endif
    } else {
      // fill send-buffer
      int nnz = rowBeginOrig[numRows_proc];
      std::vector<double> sendbuf;
      sendbuf.resize(2*nnz, 0);
      for (int i=0; i<numRows_proc; i++) {
        int row = ipermMatching[startGID+i]; // perm origina i to row
        for (int p=0; p<numProcs; p++) {
          if (row >= fstRows[p] && row < fstRows[p+1]) {
            nnz = senddispls[p];
            for (int k=0; k<rowBeginOrig[i+1]-rowBeginOrig[i]; k++) {
              double val = values_in[rowBeginOrig[i]+k];
              sendbuf[nnz + 2*k+0] = double(row);
              sendbuf[nnz + 2*k+1] = val;
            }
            senddispls[p] += 2*(rowBeginOrig[i+1]-rowBeginOrig[i]);
            break;
          }
        }
      }

      // shift back
      for (int p=numProcs; p>0; p--) {
        senddispls[p] = senddispls[p-1];
      }
      senddispls[0] = 0;

      // communicate !!
      nnz = recvdispls[numProcs];
      std::vector<double> recvbuf;
      recvbuf.resize(nnz, 0);
      MPI_Alltoallv(sendbuf.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
                    recvbuf.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                    comm);

      // put it into CSR
      for (int i=0; i<nnz; i+=2) {
        int    row = int(recvbuf[i])-startGID;
        double val = recvbuf[i+1];
        valuesRe[rowBeginRe[row]] = val;
        rowBeginRe[row] ++;
      }
      for (int i=numRows_proc; i>0; i--) rowBeginRe[i] = rowBeginRe[i-1];
      rowBeginRe[0] = 0;

      // assign values
      assign_values(rowBeginRe,valuesRe);
#ifdef MATRIX_OUT
      {
        char filename[100];
        sprintf(filename, "lW_%d.dat", myPID);
        FILE *fp = fopen(filename, "w");
        for (int i=0; i<numRows_proc; i++) {
          for (int k=rowBeginRe[i]; k<rowBeginRe[i+1]; k++)
            fprintf(fp,"%d %d %.16e\n",i,columnsRe[k],valuesRe[k]);
        }
        fclose(fp);
      }
#endif
    }
  }
  const std::vector<double> & values = *valuesPtr;
  getSubMatrices(values, valuesSub);
  timer_gather_matrices = clockIt() - startTime;
  
  output_sub_matrices(rowBeginSub, columnsSub, valuesSub);

  startTime = clockIt();
  bool verbose = (myPID == 0 && msg_level > 0);
  if (solvername == "") {
    if (pardiso_solver.getNumRows() != 0) {
#ifdef USE_INTEL_PARDISO
      // Pardiso numerical factorization
      r_val = pardiso_solver.factorize(valuesSub.data(), verbose);
      sc[0] = pardiso_solver.getSchurComplement();
#ifdef MATRIX_OUT
      {
        char filename[250];
        sprintf(filename,"S_P%d.dat", myPID);
        FILE *fp = fopen(filename,"w");;
        for (int k=0; k<sc[0].size(); k++) fprintf(fp,"%d %.16e\n",k,sc[0][k]);
        fclose(fp);
      }
#endif
#endif
    }
  } else {
    // Amesos2 numerical factorization
    size_t n = rowBeginSub.size()-1;
    if (msg_level > 0) {
      MPI_Barrier(comm);
      printf("%d: n=%d\n",myPID,int(n)); fflush(stdout);
      if (myPID == 0) {
        printf( " > Amesos2:factorize\n" ); fflush(stdout);
      }
    }
    if (n > 0) {
      // [D, E; F, C]
      Kokkos::deep_copy(E_view, 0);
      #ifdef D3S_DENSE_F
        Kokkos::deep_copy(F_view, 0);
      #else
        int nnzF = 0;
      #endif
      Kokkos::deep_copy(S_view, 0);
      int nnzD = 0;
      int n1 = 0;
      int n2 = 0;
      for (int i=0; i<n; i++) {
        int row = m_parts(i);
        if (row >= 0) {
          // [D, E]
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) {
            int col = m_parts(columnsSub[k]);
            if (col >= 0) {
              // interior D
              values_view_D(nnzD) = valuesSub[k];
              nnzD ++;
            } else {
              // separator->interior, G
              E_view(row, -col-1) = valuesSub[k];
            }
          }
          n1++;
        } else {
          // [H, C]
          for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) {
            int col = m_parts(columnsSub[k]);
            if (col >= 0) {
              // interior->separator, H
              #ifdef D3S_DENSE_F
                F_view(-row-1, col) = valuesSub[k];
              #else
                values_view_F(nnzF) = valuesSub[k];
                nnzF ++;
              #endif
            } else {
              // separtor, S
              S_view(-row-1, -col-1) = valuesSub[k];
            }
          }
          n2++;
        }
      }

      // wrap into Kokkos::CrsMatrix
      graph_t static_graph(colind_view_D, rowmap_view_D);
      crsmat_t crsmat("CrsMatrix", n1, values_view_D, static_graph);
#ifdef MATRIX_OUT
      {
        char filename[250];
        FILE *fp;
        /*sprintf(filename,"A%d.dat", myPID);
        fp = fopen(filename,"w");;
        for (int i=0; i<n; i++) for (int k=rowBeginSub[i]; k<rowBeginSub[i+1]; k++) fprintf(fp,"%d %d %.16e\n",1+i,1+columnsSub[k],valuesSub[k]);
        fclose(fp);*/
        sprintf(filename,"D%d.dat", myPID);
        fp = fopen(filename,"w");;
        for (int i=0; i<n1; i++) for (int k=rowmap_view_D(i); k<rowmap_view_D(i+1); k++) fprintf(fp,"%d %d %.16e\n",1+i,1+colind_view_D(k),values_view_D(k));
        fclose(fp);
        /*sprintf(filename,"E%d.dat", myPID);
        fp = fopen(filename,"w");;
        for (int i=0; i<n1; i++) {
          for (int j=0; j<n2; j++) fprintf(fp,"%.16e ",E_view(i,j));
          fprintf(fp,"\n");
        }
        fclose(fp);
        sprintf(filename,"F%d.dat", myPID);
        fp = fopen(filename,"w");;
        for (int i=0; i<n2; i++) {
          for (int j=0; j<n1; j++) fprintf(fp,"%.16e ",F_view(i,j));
          fprintf(fp,"\n");
        }
        fclose(fp);
        sprintf(filename,"C%d.dat", myPID);
        fp = fopen(filename,"w");;
        for (int i=0; i<n2; i++) {
          for (int j=0; j<n2; j++) fprintf(fp,"%.16e ",S_view(i,j));
          fprintf(fp,"\n");
        }
        fclose(fp);
        sprintf(filename,"part%d.dat", myPID);
        fp = fopen(filename,"w");;
        for (int i=0; i<n; i++) fprintf(fp,"%d\n",m_parts(i));
        fclose(fp);*/
      }
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      {
#ifdef D3S_USE_KOKKOS_BACKEND
        // kokkos-backend for numeric factorization
        amesos2_solver->setA(Teuchos::rcpFromRef(crsmat), Amesos2::SYMBFACT);
#else
        // wrap into Tpetra::CrsMatrix
        A = Teuchos::rcp(new MAT(crsmat, localMap,localMap));

        // keep symbolic, and do numeric
        amesos2_solver->setA(A, Amesos2::SYMBFACT);
#endif
        amesos2_solver->numericFactorization();

        // apply D^{-1} to right-interface
        // for local Schur complement
#ifdef D3S_USE_KOKKOS_BACKEND
        amesos2_solver->setB(Teuchos::rcpFromRef(E_view));
        amesos2_solver->setX(Teuchos::rcpFromRef(G_view));
#else
        auto E = Teuchos::rcp(new MV(localMap, E_view));
        auto G = Teuchos::rcp(new MV(localMap, G_view));
        amesos2_solver->setB(E);
        amesos2_solver->setX(G);
#endif
        amesos2_solver->solve();
      }
      // Form local Schur
      {
#ifdef MATRIX_OUT
        {
          char filename[250];
          sprintf(filename,"G%d.dat", myPID);
          FILE *fp = fopen(filename,"w");;
          for (int i=0; i<n1; i++) {
            for (int j=0; j<n2; j++) fprintf(fp,"%.16e ",G_view(i,j));
            fprintf(fp,"\n");
          }
          fclose(fp);
        }
#endif
        // S = S - H*G
        #ifdef D3S_DENSE_F
          KokkosBlas::gemm("N","N",
                           -1.0, F_view,
                                 G_view,
                            1.0, S_view);
        #else
          graph_t static_graph(colind_view_F, rowmap_view_F);
          crsmat_t crsmat("CrsMatrix", n1, values_view_F, static_graph);
          KokkosSparse::spmv("N", -1.0, crsmat, G_view, 1.0, S_view);
        #endif
#ifdef MATRIX_OUT
        {
          char filename[250];
          sprintf(filename,"S_A%d.dat", myPID);
          FILE *fp = fopen(filename,"w");;
          for (int i=0; i<n2; i++) {
            for (int j=0; j<n2; j++) fprintf(fp,"%.16e ",S_view(i,j));
            fprintf(fp,"\n");
          }
          fclose(fp);
        }
#endif
      }

      // TODO: Skip this
      // copy Schur complement into D3S internal data-structure
      //  + sc stores the Schur complement in *** row major **
      sc[0].resize(n2*n2);
      for (int i=0; i<n2; i++) {
        for (int j=0; j<n2; j++) sc[0][j+i*n2] = S_view(i,j);
      }
    }
  }
#ifdef MATRIX_OUT
  {
    char filename[250];
    sprintf(filename,"S%d.dat", myPID);
    FILE *fp = fopen(filename,"w");;
    int n2 = std::sqrt(sc[0].size());
    for (int i=0; i<n2; i++) {
      for (int j=0; j<n2; j++) fprintf(fp,"%.16e ",sc[0][j+i*n2]);
      fprintf(fp,"\n");
    }
    fclose(fp);
  }
#endif
  timer_pardiso_numeric = clockIt() - startTime;
  
  if (r_val == 0) {
    int level = 0;
    while (level < num_level) {
      startTime = clockIt();
      r_val = compute_schur_complement(level, values);
      timer_factor[level] = clockIt() - startTime;
      // check at each level;
      r_val = -std::abs(r_val); // making sure non-positive (error-code is negative)
      MPI_Allreduce(MPI_IN_PLACE, &r_val, 1, MPI_INT, MPI_MIN, comm);
      if (r_val != 0) break;
      level++;
    }
  }

  if (msg_level > 0) {
    MPI_Barrier(comm);
    if (myPID == 0) printf(" Factorize done\n\n");
  }
  return r_val;
}

void D3Solver::communicateMatrixValuesB(const int level,
                                        const std::vector<double> & values)
{
  std::vector<std::vector<double>> & values_send_l = values_send_B[level];
  std::vector<std::vector<int>>    & indices = values_send_B_index[level];
  const int num_send = values_send_l.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<values_send_l[i].size(); j++) {
      values_send_l[i][j] = values[indices[i][j]];
    }
  }
  communicateData(values_send_B[level], my_recv_PIDs_B[level], my_send_PIDs_B[level],
                  values_recv_B[level]);
}

void D3Solver::communicateRhsValuesB(const int level,
                                     const std::vector<double> & rhs)
{
  std::vector<std::vector<double>> & rhs_send_l = rhs_send_sep[level];
  std::vector<std::vector<int>>    & indices = rhs_send_sep_index[level];
  const int num_send = rhs_send_l.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<rhs_send_l[i].size(); j++) {
      rhs_send_l[i][j] = rhs[indices[i][j]];
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

int D3Solver::compute_schur_complement(const int level,
                                       const std::vector<double> & values)
{
  int r_val = 0;
  int numSub, mult, sep_start;
  get_level_ints(level, numSub, mult, sep_start);
  communicateMatrixValuesB(level, values);
  extractMatrixValues(level);
  bool output_mats = false;
  if (msg_level >= 2) output_mats = true;
  if (output_mats) {
    output_matrices("A_BB", rowBegin_B[level], columns_B[level], values_B[level], level);
  }
  int send_to_pid, recv_from_pid, recv_index;
  get_comm_data(level, numSub, mult, send_to_pid, recv_from_pid, recv_index);
  point_to_point_single(send_to_pid, recv_from_pid, sc[level], sc_recv[level], comm_level[level]);
  if (recv_index != -1) {
    const int num_rows = n1a[level] + n2a[level];
    // assemeble local schurs from neighbors
    assemble_dense(level, num_rows);
    // add original separator block
    add_sparse_contrib(level, num_rows);
    r_val = eliminate_separator(level);
    rhs_sep[level].resize(n1a[level]);
    if (output_mats) {
      output_dense_matrix("Sc", num_rows, level, AS[level]);
      if (n2a[level] > 0) {
        output_dense_matrix("Sc_red", n2a[level], level, sc[level+1]);
      }
    }
  }
  return r_val;
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

int D3Solver::solve_schur_complement(const int level,
                                     const std::vector<double> & rhs)
{
  int r_val = 0;
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
    r_val = eliminate_separator_rhs(level);
  }
  return r_val;
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

int D3Solver::solve(const std::vector<double> & rhs,
                          std::vector<double> & sol,
                    const int numRhs)
{
  ThrowAssert(true, numRhs == 1, "solver currently setup for only a single rhs");
  int lengthRhs = rhs.size();
  if (msg_level > 0) {
    MPI_Barrier(comm);
    if (myPID == 0) {
      printf( "\n -- D3Solver::solve(%dx%d) -- \n",lengthRhs,numRhs ); fflush(stdout);
    }
  }
#ifdef MATRIX_OUT
  {
    char filename[250];
    sprintf(filename,"RHS%d.dat", myPID);
    FILE *fp = fopen(filename, "w");
    for (int i=0; i<lengthRhs; i++) {
      fprintf(fp,"%d %.16e\n",i,rhs[i]);
    }
    fclose(fp);
  }
#endif
  std::vector<double> rhsRe;
  if (matching_option == 0) {
    getSubRhs(rhs);
  } else {
    rhsRe.resize(lengthRhs, 0);
    if (matching_option == 1) {
      getSubRhs(rhs);
    } else {
      permsolve(ipermMatching, rhs, rhsRe);
      getSubRhs(rhsRe);
#ifdef MATRIX_OUT
      {
        int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        char filename[250];
        sprintf(filename,"RHS%d_RE.dat", myRank);
        FILE *fp = fopen(filename, "w");
        for (int i=0; i<lengthRhs; i++) {
          fprintf(fp,"%d %.16e\n",i,rhsRe[i]);
        }
        fclose(fp);
      }
#endif
    }
  }
#ifdef MATRIX_OUT
  {
    char filename[250];
    sprintf(filename,"RHS_I%d.dat", myPID);
    FILE *fp = fopen(filename, "w");
    for (int i=0; i<rowsISub.size(); i++) {
      fprintf(fp,"%d %.16e\n",i,rhsI[i]);
    }
    fclose(fp);
  }
#endif

  // initialize rhs to be zero
  int num_rows = rowBeginSub.size() - 1;
  rhs_pardiso.assign(num_rows, 0);
  sol_pardiso.resize(num_rows);

  int r_val = 0;
  int n1 = rowsISub.size(); // interior
  int n2 = rowsBSub.size(); // boundary
  if (solvername == "") {
    // == PardisoMKL solves ==
    // copy interior rhs vector in
    for (size_t i=0; i<rowsISub.size(); i++) {
      rhs_pardiso[rowsISub[i]] = rhsI[i];
    }
#ifdef USE_INTEL_PARDISO
    // forward solve with Pardiso
    int phase = 331;
    pardiso_solver.solve(rhs_pardiso.data(), sol_pardiso.data(), phase);
#endif
#ifdef MATRIX_OUT
    {
      char filename[250];
      sprintf(filename,"solp1_%d.dat", myPID);
      FILE *fp = fopen(filename,"w");
      for (size_t i=0; i<rowsISub.size(); i++) fprintf(fp,"%.16e %.16e\n",rhs_pardiso[rowsISub[i]],sol_pardiso[rowsISub[i]]);
      fclose(fp);
    }
#endif
    // copy interior solution back for backward solve
    //  (schur complement part is zero)
    for (size_t i=0; i<rowsISub.size(); i++) {
      const int row = rowsISub[i];
      rhs_pardiso[row] = sol_pardiso[row];
    }
    // copy boundary solution back for schur complement solve
    rhs_sc[0].resize(rowsBSub.size());
    for (size_t i=0; i<rowsBSub.size(); i++) {
      rhs_sc[0][i] = sol_pardiso[rowsBSub[i]];
    }
  } else {
    // Amesos2 solves
    // * copy vectors into views
    // copy interior rhs vector in
    for (size_t i=0; i<rowsISub.size(); i++) {
      rhs_pardiso[i] = rhsI[i];
    }
    // solve with interior
    {
#ifdef D3S_USE_KOKKOS_BACKEND
      // kokkos-backend for solve
      Kokkos::resize(X_view, n1, numRhs);
      Kokkos::resize(B_view, n1, numRhs);
      for (int j=0; j<numRhs; j++) {
        for (int i=0; i<n1; i++) B_view(i,j) = rhs_pardiso[i+j*n1];
      }

      amesos2_solver->setB(Teuchos::rcpFromRef(B_view));
      amesos2_solver->setX(Teuchos::rcpFromRef(X_view));
#else
      using array_type = Teuchos::ArrayView<const double>;
      Teuchos::RCP<array_type> arrayB = rcp(new array_type(rhs_pardiso));
      Teuchos::RCP<array_type> arrayX = rcp(new array_type(sol_pardiso));
      const Teuchos::ArrayView<const array_type> viewB (arrayB.getRawPtr(), numRhs);
      const Teuchos::ArrayView<const array_type> viewX (arrayX.getRawPtr(), numRhs);
      // * wrap vectors into MV
      auto B = Teuchos::rcp(new MV(localMap, viewB, numRhs));
      auto X = Teuchos::rcp(new MV(localMap, numRhs));
      //auto X = Teuchos::rcp(new MV(localMap, viewX, numRhs));
      // * call Amesos2 solve for the interior solve
      amesos2_solver->setB(B);
      amesos2_solver->setX(X);
#endif

      // do solve
      amesos2_solver->solve();

      // copying out (TODO: fix)
#ifdef D3S_USE_KOKKOS_BACKEND
      for (int j=0; j<numRhs; j++) {
        for (int i=0; i<n1; i++) sol_pardiso[i+j*n1] = X_view(i,j);
      }
#else
      {
        auto localX = X->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (int j=0; j<numRhs; j++) {
          for (int i=0; i<n1; i++) sol_pardiso[i+j*n1] = localX(i,j);
        }
      }
#endif
    }
#ifdef MATRIX_OUT
    {
      char filename[250];
      sprintf(filename,"sol1_%d.dat", myPID);
      FILE *fp = fopen(filename,"w");
      for (int i=0; i<n1; i++) fprintf(fp,"%.16e %.16e\n",rhs_pardiso[i],sol_pardiso[i]);
      fclose(fp);
    }
#endif

    // * update rhs for the Schur solve, b2 -= F*x1
    {
      rhs_sc[0].resize(rowsBSub.size());
      UnmanagedViewType X1 (&sol_pardiso[0], n1, numRhs);
      UnmanagedViewType B2 (&rhs_sc[0][0],   n2, numRhs);
      #ifdef D3S_DENSE_F
        KokkosBlas::gemm("N","N",
                         -1.0, F_view,
                               X1,
                          0.0, B2);
      #else
        graph_t static_graph(colind_view_F, rowmap_view_F);
        crsmat_t crsmat("CrsMatrix", n1, values_view_F, static_graph);
        KokkosSparse::spmv("N", -1.0, crsmat, X1, 0.0, B2);
      #endif
    }
  }

  // forward solve Schur complement
  //  (solve_schur_complement will communicate original rhs to add to rhs_sc[0])
  int level = 0;
  while (level < num_level) {
    const double startTime = clockIt();
    if (matching_option == 0 || matching_option == 1) {
      r_val = solve_schur_complement(level, rhs);
    } else {
      r_val = solve_schur_complement(level, rhsRe);
    }
    timer_solve[level] += clockIt() - startTime;
    // check at each level;
    r_val = -std::abs(r_val); // making sure non-positive (error-code is negative)
    MPI_Allreduce(MPI_IN_PLACE, &r_val, 1, MPI_INT, MPI_MIN, comm);
    if (r_val != 0) return;

    level++;
  }

  // backward  solve Schur complement
  //  (at this point, we have the root separator solution)
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
  ThrowAssert(true, rowsBSub.size() == rhs_sc[0].size(), "inconsistent sizes");
  if (solvername == "") {
    // copy in boundary solution to rhs
    for (size_t i=0; i<rowsBSub.size(); i++) {
      const int row = rowsBSub[i];
      rhs_pardiso[row] = rhs_sc[0][i];
    }
#ifdef USE_INTEL_PARDISO
    int phase = 333; // backward solve
    pardiso_solver.solve(rhs_pardiso.data(), sol_pardiso.data(), phase);
#endif
    // copy back interior solution
    for (size_t i=0; i<rowsISub.size(); i++) {
      rhsI[i] = sol_pardiso[rowsISub[i]];
    }
  } else {
    // With Amesos2, the interior part of U is I.
#ifdef MATRIX_OUT
    {
      char filename[250];
      sprintf(filename,"sol2_%d.dat", myPID);
      FILE *fp = fopen(filename,"w");
      for (size_t i=0; i<num_rows; i++) fprintf(fp,"%d: %.16e %.16e\n",i,rhs_pardiso[i],sol_pardiso[i]);
      fclose(fp);;
    }
#endif
    // * update rhs for the interior solve, b1 -= G*x2
    int n1 = rowsISub.size(); // interior
    int n2 = rowsBSub.size(); // boundary
    UnmanagedViewType B1 (&sol_pardiso[0], n1, numRhs);
    UnmanagedViewType X2 (&rhs_sc[0][0], n2, numRhs);
    KokkosBlas::gemm("N","N",
                     -1.0, G_view,
                           X2,
                      1.0, B1);
#ifdef MATRIX_OUT
    {
      char filename[250];
      sprintf(filename,"solI3_%d.dat", myPID);
      FILE *fp = fopen(filename,"w");
      for (size_t i=0; i<num_rows; i++) fprintf(fp,"%d: %.16e %.16e\n",i,rhs_pardiso[i],sol_pardiso[i]);
      fclose(fp);
      sprintf(filename,"solB3_%d.dat", myPID);
      fp = fopen(filename,"w");
      for (size_t i=0; i<n2; i++) fprintf(fp,"%.16e \n",rhs_sc[0][i]);
      fclose(fp);
    }
#endif
    // copy back interior solution
    for (size_t i=0; i<rowsISub.size(); i++) {
      rhsI[i] = sol_pardiso[i];
    }
  }
  putSubSol(sol);
#ifdef MATRIX_OUT
  if (myPID == 0) {
    printf("SOL=[\n");
    for (size_t i=0; i<sol.size(); i++) printf("%d %e\n",i,sol[i]);
    printf("];\n");
  }
#endif

  if (matching_option == 1) {
    permsolve(permMatching, sol, rhsRe);
    for (size_t i=0; i<sol.size(); i++) {
      sol[i] = rhsRe[i];
    }
  }
  if (msg_level > 0) {
    MPI_Barrier(comm);
    if (myPID == 0) printf(" Solve done\n\n");
  }
  return r_val;
}

void D3Solver::permsolve(const std::vector<int> perm,
                         const std::vector<double> & rhs,
                               std::vector<double> & rhsRe) {
  int lengthRhs = rhs.size();
  std::vector<int> sendcounts_rhs;
  sendcounts_rhs.resize(numProcs, 0);
  for (int i=0; i<lengthRhs; i++) {
    int row = perm[startGID+i]; // perm origina i to row
    for (int p=0; p<numProcs; p++) {
      if (row >= fstRows[p] && row < fstRows[p+1]) {
        sendcounts_rhs[p] ++;
        break;
      }
    }
  }
  std::vector<int> senddispls_rhs;
  senddispls_rhs.resize(numProcs+1, 0);
  for (int p=0; p<numProcs; p++) {
    sendcounts_rhs[p] *= 2;
    senddispls_rhs[p+1] = senddispls_rhs[p]+sendcounts_rhs[p];
  }

  std::vector<int> recvcounts_rhs;
  std::vector<int> recvdispls_rhs;
  recvcounts_rhs.resize(numProcs, 0);
  MPI_Alltoall(sendcounts_rhs.data(), 1, MPI_INT, recvcounts_rhs.data(), 1, MPI_INT, comm);
  recvdispls_rhs.resize(numProcs+1, 0);
  for (int p=0; p<numProcs; p++) {
    recvdispls_rhs[p+1] = recvdispls_rhs[p] + recvcounts_rhs[p];
  }

  // fill send-buffer
  std::vector<double> sendbuf;
  sendbuf.resize(2*lengthRhs, 0);
  for (int i=0; i<lengthRhs; i++) {
    int row = perm[startGID+i]; // perm origina i to row
    for (int p=0; p<numProcs; p++) {
      if (row >= fstRows[p] && row < fstRows[p+1]) {
        int nnz = senddispls_rhs[p];
        sendbuf[nnz + 0] = double(row);
        sendbuf[nnz + 1] = rhs[i];
        senddispls_rhs[p] += 2;
        break;
      }
    }
  }
  // shift back
  for (int p=numProcs; p>0; p--) {
    senddispls_rhs[p] = senddispls_rhs[p-1];
  }
  senddispls_rhs[0] = 0;

  // communicate !!
  std::vector<double> recvbuf;
  recvbuf.resize(2*lengthRhs, 0);
  MPI_Alltoallv(sendbuf.data(), sendcounts_rhs.data(), senddispls_rhs.data(), MPI_DOUBLE,
                recvbuf.data(), recvcounts_rhs.data(), recvdispls_rhs.data(), MPI_DOUBLE,
                comm);

  for (int i=0; i<lengthRhs; i++) {
    int    row = int(recvbuf[2*i])-startGID;
    double val = recvbuf[2*i+1];
    rhsRe[row] = val;
  }
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
  ThrowAssert(true, sep_map[level].size() == rhs_sc[level].size(), "unequal sizes");
  int length = sep_map[level].size();
  for (int i=0; i<length; i++) {
    const int row = sep_map[level][i];
    rhs[row] = rhs_sc[level][i];
  }
  ThrowAssert(true, sep_map_recv[level].size() == rhs_sc_recv[level].size(), "unequal sizes");
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
  ThrowAssert(true, num_rowsB == int(sep_map_B[level].size()), "unequal lengths");
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
  int num_additional = 0;
  {
    auto iter = std::unique(not_in_sep.begin(), not_in_sep.end());
    not_in_sep.erase(iter, not_in_sep.end());
    // add potentially missing rows to not_in_sep (this is rare but verified it is possible)
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
  if (msg_level < 2) return;
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
  if (msg_level < 2) return;
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

void D3Solver::initialize_schur_complement(const int level,
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
  
  extractMatrixStructures(level, row_GIDs_recv, column_counts_recv, column_GIDs_recv,
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

int D3Solver::eliminate_separator(const int level)
{
  const MKL_INT n1 = n1a[level];
  const MKL_INT n2 = n2a[level];
  assign_matrix_blocks(level);
  const double startTime = clockIt();

  MKL_INT info = 0;
  MKL_INT matrix_layout = LAPACK_COL_MAJOR;
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    ipiv[level].resize(n1);
    //printf("C=[\n");
    //for (int i=0; i<n1; i++) {
    //  for (int j=0; j<n1; j++) printf("%.16e ",A11[level][i+j*n1]);
    //}
    //printf("];\n");
    info = LAPACKE_dgetrf(matrix_layout, n1, n1, A11[level].data(), n1,
                          ipiv[level].data());
    ThrowAssert(false, info == 0, "error in call to LAPACKE_dgetrf");
    if (info != 0) {
      fprintf(stderr, "DGETRF(%dx%d) failed with info=%d in D3S::eliminate_separator\n",n1,n1,info);
      return info;
    }
#endif
  }
  if (n2 == 0) {
    timer_factor_dla[level] = clockIt() - startTime;
    return info;
  }
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    info = LAPACKE_dgetrs(matrix_layout, 'N', n1, n2, A11[level].data(), n1,
                          ipiv[level].data(), A12[level].data(), n1);
    ThrowAssert(false, info == 0, "error in call to LAPACKE_dgetrs");
    if (info != 0) {
      fprintf(stderr, "DGETRS(%dx%d) failed with info=%d in D3S::eliminate_separator\n",n1,n2,info);
      return info;
    }
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
  return info;
}

int D3Solver::eliminate_separator_rhs(const int level)
{
  const double startTime = clockIt();
  const MKL_INT n1 = n1a[level];
  const MKL_INT n2 = n2a[level];
  const MKL_INT n = n1 + n2; // leading dimension of AS_rhs[level]
  const MKL_INT num_rhs = 1;
  double* rhs = AS_rhs[level].data();

  int info = 0;
  int matrix_layout = LAPACK_COL_MAJOR;
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    info = LAPACKE_dgetrs(matrix_layout, 'N', n1, num_rhs, A11[level].data(), n1,
                          ipiv[level].data(), rhs, n);
    ThrowAssert(false, info == 0, "error in call to LAPACKE_dgetrs");
    if (info != 0) {
      fprintf(stderr, "DGETRS(%dx%d) with ldb=%d failed with info=%d in D3S::eliminate_separator_rhs(n1=%d, n2=%d)\n",
              n1,num_rhs,n,info,n1,n2);
      return info;
    }
#endif
  }
  if (n2 == 0) {
    timer_solve_dla[level] += clockIt() - startTime;
    return info;
  }
  rhs_sc[level+1].resize(n2);
  double* C = rhs_sc[level+1].data();
  for (int i=0; i<n2; i++) C[i] = rhs[n1+i];
  if (n1 > 0) {
#ifdef USE_INTEL_PARDISO
    double alpha(-1), beta(1);
    CBLAS_LAYOUT layout = CblasColMajor;
    cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n2, num_rhs, n1, alpha,
                A21[level].data(), n2, rhs, n, beta, C, n2);
#endif
  }
  timer_solve_dla[level] += clockIt() - startTime;
  return info;
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

