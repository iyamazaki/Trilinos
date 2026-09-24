
#include <vector>

#include "Teuchos_RCP.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultMpiComm.hpp"

#ifndef D3S_COMMUNICATION_HELPER_H
#define D3S_COMMUNICATION_HELPER_H

  // utility kernels for communication
  int getLocalID(const int gID,
                 const std::vector<int> & svec,
                 const bool do_not_throw=false);

  int getLocalID(const int gID,
                 const int* array,
                 const int length,
                 const bool do_not_throw);

  int getLocalID_unsorted(const int gID,
                          const std::vector<int> & vec);

  std::vector<int> myReceives(const std::vector<int> & mySends,
                              const std::vector<int> & targetMPIs,
                              Teuchos::RCP<Teuchos::MpiComm<int>> comm);


  // communication kernels
  template <typename T>
  void communicateData(const std::vector<std::vector<T>> & data_send,
                       const std::vector<int> & my_recv_PIDs,
                       const std::vector<int> & my_send_PIDs,
                       std::vector<std::vector<T>> & data_recv,
                       Teuchos::RCP<Teuchos::MpiComm<int>> comm,
                       const bool reverse_comm=false);

  template <typename SC>
  void communicateMatrixData(const std::vector<int> & activeSubs,
                             const std::vector<std::vector<int>> & num_rows_send,
                             const std::vector<std::vector<int>> & row_GIDs_send,
                             const std::vector<std::vector<int>> & column_counts_send,
                             const std::vector<std::vector<int>> & column_GIDs_send,
                             const std::vector<std::vector<SC>>  & values_send_here,
                             std::vector<std::vector<int>> & num_rows_recv,
                             std::vector<std::vector<int>> & row_GIDs_recv,
                             std::vector<std::vector<int>> & column_counts_recv,
                             std::vector<std::vector<int>> & column_GIDs_recv,
                             std::vector<std::vector<SC>>  & values_recv_here,
                             std::vector<int> & my_send_PIDs,
                             std::vector<int> & my_recv_PIDs,
                             const std::vector<int> & targetMPIs,
                             Teuchos::RCP<Teuchos::MpiComm<int>> comm);

  template <typename SC>
  void communicateMatrixValues(const std::vector<SC> & values,
                               const std::vector<std::vector<int>> & values_send_index,
                                     std::vector<int> & my_send_PIDs_sub,
                                     std::vector<std::vector<SC>> & values_send,
                               const std::vector<int> & my_recv_PIDs_sub,
                                     std::vector<std::vector<SC>> & values_recv,
                               Teuchos::RCP<Teuchos::MpiComm<int>> comm);

  template <typename SC>
  void communicateRhsData(const std::vector<int> & activeSubs,
                          const std::vector<int> & targetMPIs,
                          const std::vector<std::vector<int>> & num_rows_send_rhs,
                                std::vector<int> & my_send_PIDs_rhs,
                                std::vector<int> & my_recv_PIDs_rhs,
                                std::vector<std::vector<SC>> & rhs_recv,
                          Teuchos::RCP<Teuchos::MpiComm<int>> comm);


  void communicateRhsData(const std::vector<int> & activeSubs,
                          const std::vector<int> & targetMPIs,
                          const std::vector<std::vector<int>> & num_rows_send,
                          const std::vector<std::vector<int>> & row_GIDs_send,
                          std::vector<std::vector<int>> & row_GIDs_recv,
                          std::vector<int> & my_send_PIDs,
                          std::vector<int> & my_recv_PIDs,
                          Teuchos::RCP<Teuchos::MpiComm<int>> comm);

  template <typename SC>
  void communicateMatrixValuesB(const int level, const std::vector<SC> & values,
                                std::vector<std::vector<std::vector<int>>> & values_send_B_index,
                                std::vector<std::vector<std::vector<SC>>> & values_send_B,
                                std::vector<std::vector<std::vector<SC>>> & values_recv_B,
                                std::vector<std::vector<int>> & my_send_PIDs_B,
                                std::vector<std::vector<int>> & my_recv_PIDs_B,
                                Teuchos::RCP<Teuchos::MpiComm<int>> comm);

  template <typename SC>
  void communicateRhsValuesB(const int level, const std::vector<SC> & rhs,
		             std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
		             std::vector<std::vector<std::vector<SC>>> & rhs_send_sep,
		             std::vector<std::vector<std::vector<SC>>> & rhs_recv_sep,
			     std::vector<std::vector<int>> & my_send_PIDs_sep,
			     std::vector<std::vector<int>> & my_recv_PIDs_sep,
                             Teuchos::RCP<Teuchos::MpiComm<int>> comm);

  template <typename SC>
  void communicate_solution(const int level, std::vector<SC> & sol,
                            std::vector<std::vector<SC>> & AS_rhs,
		            std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
		            std::vector<std::vector<std::vector<int>>> & rhs_recv_sep_index,
                            std::vector<std::vector<std::vector<SC>>> & rhs_send_sep,
                            std::vector<std::vector<std::vector<SC>>> & rhs_recv_sep,
                            std::vector<std::vector<int>> & my_send_PIDs_sep,
                            std::vector<std::vector<int>> & my_recv_PIDs_sep,
                            Teuchos::RCP<Teuchos::MpiComm<int>> comm);
#endif

