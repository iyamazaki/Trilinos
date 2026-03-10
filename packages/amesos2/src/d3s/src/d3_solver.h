#pragma once

#include <vector>
#include <mpi.h>

#include "metis.h"
#include "throwAssert.h"
#include "mpi.h"
#include "gather_to_root_simple.h"

#include "Amesos2_Solver.hpp"

#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "KokkosSparse_CrsMatrix.hpp"
#include "trilinos_btf_decl.h"

// Note: logic still needed to not define USE_INTEL_PARDISO for non-Intel builds
#define USE_INTEL_PARDISO

#ifdef USE_INTEL_PARDISO
  #include "sc_pardiso.h"
#endif

#ifndef D3SOLVER_HPP
#define D3SOLVER_HPP

class D3Solver
{
public:
  D3Solver(MPI_Comm commIn);

  ~D3Solver();

  void setNumThreads(const int num_threadsIn);
  void setOrderingOption(const int matching_optionIn, const int reorder_optionIn);
  void setVerbose(const int msg_levelIn, const int debug_levelIn);
  void setInteriorSolverName(const std::string solvername_in);

  int initialize(const std::vector<int> & rowBegin_in,
                 const std::vector<int> & columns_in,
                 const int startGID_in,
                 const int numProcSolver_in);
  
  int factorize(const std::vector<double> & values);
  
  int solve(const std::vector<double> & rhs,
                  std::vector<double> & sol,
            const int numRhs=1);

  void gatherScatterSol(std::vector<double> & sol,
                        std::vector<double> & solAll) const;
  
  void output_timers() const;
  
 private:
  inline double clockIt() const;

  int getLocalID(const int gID,
                 const std::vector<int> & svec,
                 const bool do_not_throw=false) const;

  int getLocalID(const int gID,
                 const int* array,
                 const int length,
                 const bool do_not_throw) const;
  
  int getLocalID_unsorted(const int gID,
                          const std::vector<int> & vec) const;
  
  int getLocalID_col(const int gID,
                     const std::vector<int> & vec) const;
  
  void getDispls(const std::vector<int> & numEntriesProc,
                 std::vector<int> & displs) const;
  
  void getGraphForMetis(const std::vector<int> & rowBegin,
                        const std::vector<int> & columns,
                        std::vector<idx_t> & rowperm,
                        std::vector<idx_t> & irowperm,
                        std::vector<idx_t> & rowBeginMetis,
                        std::vector<idx_t> & columnsMetis,
                        std::vector<std::pair<int,int>> & additional_edges);

  void getLevelsAndLocations(const int numProc,
                             std::vector<int> & level,
                             std::vector<int> & location) const;
  
  void extractRowSubIDs(const std::vector<int> & node_begin,
                        const std::vector<int> & node_sub_id,
                        const std::vector<idx_t> & iperm,
                        std::vector<int> & row_sub_id) const;
  
  void checkRowSubIDs(const std::vector<int> & rowSubIDs,
                      const std::vector<idx_t> & rowperm,
                      const std::vector<int> & rowBegin,
                      const std::vector<int> & columns) const;
  
  void get_separators(const std::vector<int> & rowSubIDs,
                      std::vector<int> & sepIDs,
                      std::vector<int> & sepBegin,
                      std::vector<int> & sepRows) const;
  
  void getGraphTranspose(const std::vector<int> & rowBegin,
                         const std::vector<int> & columns,
                         std::vector<int> & rowBeginT,
                         std::vector<int> & columnsT) const;
  
  void getRowSubIDs(const std::vector<int> & rowBegin,
                    const std::vector<int> & columns);
  
  void communicateMatrixData(const std::vector<int> & activeSubs,
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
                             std::vector<int> & my_recv_PIDs);
  
  void communicateMatrixValues(const std::vector<double> & values);
  
  void communicateMatrixValuesB(const int level,
                                const std::vector<double> & values);
  
  void communicateRhsData(const std::vector<int> & activeSubs,
                          const std::vector<std::vector<int>> & num_rows_send_rhs);
  
  std::vector<int> myReceives(const std::vector<int> & mySends);
  
  template <typename T>
  void communicateData(const std::vector<std::vector<T>> & data_send,
                       const std::vector<int> & my_recv_PIDs,
                       const std::vector<int> & my_send_PIDs,
                       std::vector<std::vector<T>> & data_recv,
                       const bool reverse_comm=false);

  void output_rows(const std::string name,
                   const std::vector<int> & rows);
  
  void output_sub_matrices(const std::vector<int> & rowBegin,
                           const std::vector<int> & columns,
                           const std::vector<double> & values);
  
  void phase1(const std::vector<int> & rowBegin,
              const std::vector<int> & columns,
              std::vector<std::vector<int>> & num_rows_recv,
              std::vector<std::vector<int>> & row_GIDs_recv,
              std::vector<std::vector<int>> & column_counts_recv,                      
              std::vector<std::vector<int>> & column_GIDs_recv);
  
  void phase1_rhs();
  
  void phase2(const int level,
              const std::vector<int> & rowBegin,
              const std::vector<int> & columns,
              const std::vector<int> & separators,
              const int delta_pid,
              std::vector<std::vector<int>> & num_rows_recv,
              std::vector<std::vector<int>> & row_GIDs_recv,
              std::vector<std::vector<int>> & column_counts_recv,
              std::vector<std::vector<int>> & column_GIDs_recv);
  
  void generateSubMatrices(const std::vector<std::vector<int>> & row_GIDs_recv,
                           const std::vector<std::vector<int>> & column_counts_recv,
                           const std::vector<std::vector<int>> & column_GIDs_recv,
                           std::vector<int> & rowBegin,
                           std::vector<int> & columns,
                           std::vector<double> & values,
                           std::vector<int> & rowGIDs);
  
  void extractMatrixStructures(const int level,
                     const std::vector<std::vector<int>> & row_GIDs_recv,
                     const std::vector<std::vector<int>> & column_counts_recv,
                     const std::vector<std::vector<int>> & column_GIDs_recv,
                     std::vector<int> & rowBegin,
                     std::vector<int> & columns,
                     std::vector<double> & values,
                     std::vector<int> & rowGIDs);
  
  void extractMatrixValues(const int level);
  
  void extractRhs(const int level);
  
  void output_dense_matrix(const std::string prefix,
                           const int numRows,
                           const int level,
                           const std::vector<double> & A) const;
  
  void output_matrices(const std::string prefix,
                       const std::vector<int> & rowBegin,
                       const std::vector<int> & columns,
                       const std::vector<double> & values,
                       const int level) const;
  
  std::vector<int> getSubRows(const std::vector<std::vector<int>> & row_GIDs_recv) const;
  
  void getSubMatrices(const std::vector<int> & rowBegin,
                      const std::vector<int> & columns,
                      std::vector<int> & rowBeginSub,
                      std::vector<int> & columnsSub,
                      std::vector<double> & valuesSub,
                      std::vector<int> & rowGIDsSub);
  
  void getSubMatrices(const std::vector<double> & values,
                      std::vector<double> & valuesSub);
  
  std::vector<int> getRowGIDsSubB(const std::vector<int> & rowGIDsSub);
  
  void get_comm_pairs(const int num_proc_here,
                      const int myPID_here,
                      int & send_to_pid,
                      int & recv_from_pid,
                      int & recv_index) const;

  void get_schur_gids(const int send_pid,
                      const int recv_pid,
                      const std::vector<int> & sourceGIDs,
                      std::vector<int> & targetGIDs,
                      MPI_Comm comm_here);
  
  template <typename T>
  void point_to_point_single(const int send_to_pid,
                             const int recv_from_pid,
                             const std::vector<T> & send_data,
                             std::vector<T> & recv_data,
                             MPI_Comm comm_here);
    
  void initialize_schur_complement(const int level,
                                   const std::vector<int> & rowBegin,
                                   const std::vector<int> & columns,
                                   const std::vector<int> & rowGIDsSubB,
                                   std::vector<int> & not_in_sep);
  
  int compute_schur_complement(const int level,
                               const std::vector<double> & values);
  
  int solve_schur_complement(const int level,
                             const std::vector<double> & rhs);
  
  void assemble_rhs(const int level);
  
  void assemble_dense(const int level,
                      const int n);

  void assemble_dense(const int level,
                      const std::vector<int> & gIDs,
                      const std::vector<int> & gIDs_recv,
                      const int sep_number,
                      const std::vector<int> & rowGIDsB,
                      std::vector<int> & not_in_sep);
  
  void assign_matrix_blocks(const int level);
  
  void resize_vectors();
  
  int eliminate_separator(const int level);
  
  int eliminate_separator_rhs(const int level);
  
  void convert_to_row_major(const std::vector<double> & A_col_major,
                            const int num_rows,
                            const int num_cols,
                            std::vector<double> & A_row_major) const;
  
  bool determine_valid_row(const int gID,
                           const std::vector<int> & separators) const;
  
  void getSubRhs(const std::vector<double> & rhs);
  
  void putSubSol(std::vector<double> & sol);
  
  void gatherSubRhsI();
  
  void scatterSubSolI();
  
  int get_num_rows_sep(const int sep_number) const;
  
  void sort_cols_and_indices(int* cols,
                             int* indices,
                             std::pair<int,int>* col_index_pairs,
                             const int num_cols);
  
  void add_sparse_contrib(const int level,
                          const int sep_number,
                          const std::vector<int> & not_in_sep,
                          const std::vector<int> & rowGIDsB);

  void add_sparse_contrib(const int level,
                          const int num_rows);

  void add_sep_contrib(const int level);
  
  void phase2_rhs(const int level,
                  const std::vector<int> & separators,
                  const int delta_pid,
                  const int sep_number);
  
  void communicateRhsValuesB(const int level,
                             const std::vector<double> & rhs);
  
  void communicateRhsData(const std::vector<int> & activeSubs,
                          const std::vector<std::vector<int>> & num_rows_send,
                          const std::vector<std::vector<int>> & row_GIDs_send,
                          std::vector<std::vector<int>> & row_GIDs_recv,
                          std::vector<int> & my_send_PIDs,
                          std::vector<int> & my_recv_PIDs);
  
  void get_level_ints(const int level,
                      int & numSub,
                      int & mult,
                      int & sep_start) const;
  
  void communicate_solution(const int level,
                            std::vector<double> & sol);
  
  void sort_and_add_zero_diags(std::vector<int> & rowBegin,
                               std::vector<int> & columns);
  
  int get_sep_number(const int sep_start,
                     const int recv_index) const;
  
  void permsolve(const std::vector<int> perm,
                 const std::vector<double> & rhs,
                       std::vector<double> & rhsRe);
  void backsolve(const int level);
  
  void scatter_sol(const int level);
  
  void get_comm_data(const int level,
                     int & send_to_pid,
                     int & recv_from_pid,
                     int & recv_index) const;
  
  void get_comm_data(const int level,
                     const int numSub,
                     const int mult,
                     int & send_to_pid,
                     int & recv_from_pid,
                     int & recv_index) const;
  
  void get_color_and_key(const int numSub,
                         const int mult,
                         int & color,
                         int & key) const;
  
  int setNumProcSolver(const int numProcSolver_in);
  
  void process_names(std::string & all_names,
                     const int numProc,
                     const int max_length,
                     int & num_nodes);
  
  void get_best_ranks(const int node,
                      const int sub_start,
                      const int num_subs_per_node,
                      const std::vector<std::vector<int>> & node_pids,
                      const std::vector<int> & nnz_proc,
                      std::vector<int> & best_ranks) const;
  
  int get_best_node(const int sub_start,
                    const int num_subs_per_node,
                    const std::vector<std::vector<int>> & node_pids,
                    const std::vector<int> & nnz_proc,
                    std::vector<bool> & node_flag) const;

  void assign_graph(const std::vector<int> & rowBegin,
                    const std::vector<int> & columns,
                    const std::vector<int> & extraEdges);
  void assign_graph(const std::vector<int> & rowBegin_in,
                    const std::vector<int> & columns_in,
                    const std::vector<int> & rowBegin,
                    const std::vector<int> & columns,
                    const std::vector<int> & extraEdges);
  
  void update_graph(const std::vector<int> & rowBegin,
                    const std::vector<int> & columns,
                    const std::vector<int> & extraEdges);
  
  void assign_values(const std::vector<double> & values_in);
  void assign_values(const std::vector<int> & rowBegin_in,
                     const std::vector<double> & values_in);
  
  int get_proc_for_row(const int row,
                       const std::vector<int> & numRowsAll,
                       const int numProc,
                       int & first_proc,
                       int & first_row) const;
  
  void scatter_additional_edges(const std::vector<std::pair<int,int>> & additional_edges,
                                std::vector<int> & extraEdges);
  
  int num_nodes_use(const int num_nodes) const;

  std::vector<int> gather_nnz_proc(const std::vector<int> & rowBegin) const;
  
  std::vector<std::vector<int>> gather_node_pids(int & num_nodes);

  void assignTargetMPIs(const std::vector<int> & rowBegin);
  
  void output_time(const std::string & message,
                   const double time) const;
  
  void getProcName();
 
private:

  MPI_Comm comm;
  int myPID, numProcs, numProcSolver, num_threads;
  int numRows_global, numRows_proc, startGID, num_level;

  bool robust_option;
  int matching_option, reorder_option;
  int msg_level;

  int structurally_symmetric, num_extra_edges;
  std::vector<int> permMatching, ipermMatching;

  // 1D block row after row-matching
  std::vector<int> rowBeginRe, columnsRe;
  std::vector<double> valuesRe;

  // comm for matching
  std::vector<int> fstRows;
  std::vector<int> sendcounts;
  std::vector<int> senddispls;
  std::vector<int> recvcounts;
  std::vector<int> recvdispls;


  std::vector<int> rowSubIDs, sepIDs, sepRows, sepBegin, rowsB, targetMPIs,
    rowsISub, rowsBSub;
  std::vector<int> rowBeginSub, columnsSub, my_send_PIDs_sub, my_recv_PIDs_sub,
    my_send_PIDs_rhs, my_recv_PIDs_rhs, old_to_new_indices, n1a, n2a;

  std::vector<int> rowBeginUse, columnsUse;
  std::vector<int> rowBeginOrig;
  std::vector<double> valuesSub, rhsSub, rhsI, rhs_pardiso, sol_pardiso,
    timer_factor, timer_factor_dla, timer_solve, timer_solve_dla, valuesUse;
  std::vector<std::vector<int>> sep_map, sep_map_recv, rhs_index_send, sc_GIDs,
    values_send_index, index_map_sub, sep_map_B,
    rowBegin_B, columns_B, my_send_PIDs_B, my_recv_PIDs_B, my_send_PIDs_sep,
    my_recv_PIDs_sep;
  std::vector<std::vector<std::vector<int>>> values_send_B_index,
    index_map_B, rhs_send_sep_index, rhs_recv_sep_index;
  std::vector<std::vector<std::vector<double>>> values_send_B, values_recv_B,
    rhs_send_sep, rhs_recv_sep;
  std::vector<std::vector<double>> AS, values_send, values_recv, AS_rhs,
    A11, A12, A21, A22;
  // Schur complement
  std::vector<std::vector<double>> sc, sc_recv;
  // rhs vectors
  std::vector<std::vector<double>> rhs_send, rhs_recv, rhs_sc, rhs_sc_recv,
    rhs_sep, values_B;
  std::vector<MPI_Comm> comm_level;
  std::string node_name;
  std::vector<std::string> node_names;
  double timer_pardiso_numeric=0, timer_pardiso_symbolic=0,
    timer_gather_matrices=0;
  const std::vector<int> *rowBeginPtr, *columnsPtr;
  const std::vector<double> *valuesPtr;

  #ifdef USE_INTEL_PARDISO
    std::vector<std::vector<MKL_INT>> ipiv;
    sc_pardiso pardiso_solver;
  #endif

  // Interior Amesos2 solver
  int debug_level_interior;
  std::string solvername;
  using SC = double;
  using LO = Tpetra::Map<>::local_ordinal_type;
  using GO =  Tpetra::Map<>::global_ordinal_type;
  using NO = Tpetra::Map<>::node_type;
  using MAT = Tpetra::CrsMatrix<SC,LO,GO>;
  using MV = Tpetra::MultiVector<SC,LO,GO>;
  using map_type = Tpetra::Map<LO, GO, NO>;
  Kokkos::View<int*> m_parts;
  Teuchos::RCP<const map_type > localMap;
  Teuchos::RCP<MAT> A;
  Teuchos::RCP<MV> X;
  Teuchos::RCP<MV> B;

#define D3S_USE_KOKKOS_BACKEND
#ifdef D3S_USE_KOKKOS_BACKEND
  using execution_space = typename NO::execution_space;
  using memory_space = typename NO::memory_space;
  using device_t = Kokkos::Device<execution_space, memory_space>;

  using crsmat_t = KokkosSparse::CrsMatrix<double, int, device_t>;
  using mv_view_t = Kokkos::View<double**, Kokkos::LayoutLeft, device_t>;
  Teuchos::RCP<Amesos2::Solver<crsmat_t,mv_view_t>> amesos2_solver;
#else
  using crsmat_t = typename MAT::local_matrix_device_type;
  using mv_view_t = typename MV::host_view_type::non_const_type;
  Teuchos::RCP<Amesos2::Solver<MAT,MV>> amesos2_solver;
#endif
  using graph_t = typename crsmat_t::StaticCrsGraphType;
  using rowmap_view_t = typename graph_t::row_map_type::non_const_type;
  using colind_view_t = typename graph_t::entries_type::non_const_type;
  using values_view_t = typename crsmat_t::values_type::non_const_type;

  using UnmanagedViewType = Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  // [D, G; H, S]
  // D in csrmat
  rowmap_view_t rowmap_view_D;
  colind_view_t colind_view_D;
  values_view_t values_view_D;
  // F in csrmat
  rowmap_view_t rowmap_view_F;
  colind_view_t colind_view_F;
  values_view_t values_view_F;

  mv_view_t E_view, F_view;
  mv_view_t G_view, S_view;

  mv_view_t X_view;
  mv_view_t B_view;
};

#endif //D3SOLVER_HPP
