
#include "throwAssert.h"
#include "communication_helper.h"


// utility kernels for communication
int getLocalID(const int gID,
               const std::vector<int> & svec,
               const bool do_not_throw)
{
  return getLocalID(gID, svec.data(), svec.size(), do_not_throw);
}

int getLocalID(const int gID,
               const int* array,
               const int length,
               const bool do_not_throw)
{
  auto it = std::lower_bound(array, array+length, gID);
  const bool valid = (it != array+length) && (*it == gID);
  if ((valid == false) && do_not_throw) {
    return -1;
  }
  if (valid == false) {
    std::cout << " Invalid getLocalID: " << gID << std::endl;
  }
  ThrowAssert(true, valid, "index not found");
  return std::distance(array, it);
}

int getLocalID_unsorted(const int gID,
                        const std::vector<int> & vec)
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

std::vector<int> myReceives(const std::vector<int> & mySends,
                            const std::vector<int> & targetMPIs,
                            Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  int numProc = comm->getSize();
  int myPID = comm->getRank();
  std::vector<int> sendArray(numProc, 0);
  for (size_t i=0; i<mySends.size(); i++) {
    sendArray[targetMPIs[mySends[i]]] = 1;
  }
  const int n = numProc*numProc;
  std::vector<int> gatherArrayRoot(n);
  int root = 0;
  Teuchos::gather<int,int>(sendArray.data(), numProc,gatherArrayRoot.data(), numProc,
                           root, *comm);
  Teuchos::broadcast<int, int>(*comm, root, n, gatherArrayRoot.data());
  std::vector<int> myRecvs;
  for (int j=0; j<numProc; j++) {
    if (gatherArrayRoot[myPID+numProc*j] == 1) myRecvs.push_back(j);
  }
  return myRecvs;
}

// communication kernels
template <typename T>
void communicateData(const std::vector<std::vector<T>> & data_send,
                     const std::vector<int> & my_recv_PIDs,
                     const std::vector<int> & my_send_PIDs,
                           std::vector<std::vector<T>> & data_recv,
                     Teuchos::RCP<Teuchos::MpiComm<int>> comm,
                     const bool reverse_comm)
{
  const int myPID = comm->getRank();
  const int num_recvs = my_recv_PIDs.size();
  const int num_sends = my_send_PIDs.size();
  ThrowAssert(true, my_recv_PIDs.size() == data_recv.size(), "incompatible sizes");
  ThrowAssert(true, my_send_PIDs.size() == data_send.size(), "incompatible sizes");
  int numProc =  comm->getSize();
  const int tag = 0;
  Teuchos::Array<Teuchos::RCP<Teuchos::CommRequest<int>>> send_requests(num_sends);
  Teuchos::Array<Teuchos::RCP<Teuchos::CommRequest<int>>> recv_requests(num_recvs);
  Teuchos::Array<Teuchos::RCP<Teuchos::CommStatus<int>>> statuses(numProc);
  // communicate data
  bool has_ownership = false;
  int actual_num_sends(0), actual_num_recvs(0);
  for (int i=0; i<num_recvs; i++) {
    // don't receive data from self
    if (my_recv_PIDs[i] != myPID) {
      const int count = data_recv[i].size();
      T * data = const_cast<T*>(data_recv[i].data());
      recv_requests[actual_num_recvs++]
        = Teuchos::ireceive<int, T>(Teuchos::ArrayRCP<T>(data, 0, count, has_ownership),
                                    my_recv_PIDs[i], tag, *comm);
    }
  }
  for (int i=0; i<num_sends; i++) {
    // don't send data to self, but do copy over data
    if (my_send_PIDs[i] != myPID) {
      const int count = data_send[i].size();
      T * data = const_cast<T*>(data_send[i].data());
      send_requests[actual_num_sends++]
        = Teuchos::isend<int, T>(Teuchos::ArrayRCP<T>(data, 0, count, has_ownership),
                                 my_send_PIDs[i], tag, *comm);
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
  Teuchos::waitAll(*comm, send_requests(0, num_sends), statuses(0, num_sends));
  Teuchos::waitAll(*comm, recv_requests(0, num_recvs), statuses(0, num_recvs));
}

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
                           Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  const int numActive = activeSubs.size();
  my_recv_PIDs = myReceives(activeSubs, targetMPIs, comm);
  my_send_PIDs.resize(numActive);
  for (int i=0; i<numActive; i++) {
    my_send_PIDs[i] = targetMPIs[activeSubs[i]];
  }

  const int num_recvs = my_recv_PIDs.size();
  // number of rows
  num_rows_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) num_rows_recv[i].resize(1);
  communicateData(num_rows_send, my_recv_PIDs, my_send_PIDs, num_rows_recv, comm);
  // row numbers
  row_GIDs_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) row_GIDs_recv[i].resize(num_rows_recv[i][0]);
  communicateData(row_GIDs_send, my_recv_PIDs, my_send_PIDs, row_GIDs_recv, comm);
  // column counts for rows
  column_counts_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) column_counts_recv[i].resize(num_rows_recv[i][0]);
  communicateData(column_counts_send, my_recv_PIDs, my_send_PIDs, column_counts_recv, comm);
  // column GIDs and values
  column_GIDs_recv.resize(num_recvs);
  values_recv_here.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) {
    int num_terms = 0;
    for (int j=0; j<num_rows_recv[i][0]; j++) num_terms += column_counts_recv[i][j];
    column_GIDs_recv[i].resize(num_terms);
    values_recv_here[i].resize(num_terms);
  }
  communicateData(column_GIDs_send, my_recv_PIDs, my_send_PIDs, column_GIDs_recv, comm);
}

template <typename SC>
void communicateMatrixValues(const std::vector<SC> & values,
                             const std::vector<std::vector<int>> & values_send_index,
                                   std::vector<int> & my_send_PIDs_sub,
                                   std::vector<std::vector<SC>> & values_send,
                             const std::vector<int> & my_recv_PIDs_sub,
                                   std::vector<std::vector<SC>> & values_recv,
                             Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  const int num_send = values_send.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<values_send[i].size(); j++) {
      values_send[i][j] = values[values_send_index[i][j]];
    }
  }
  communicateData(values_send, my_recv_PIDs_sub, my_send_PIDs_sub, values_recv, comm);
}

template <typename SC>
void communicateRhsData(const std::vector<int> & activeSubs,
                        const std::vector<int> & targetMPIs,
                        const std::vector<std::vector<int>> & num_rows_send_rhs,
                              std::vector<int> & my_send_PIDs_rhs,
                              std::vector<int> & my_recv_PIDs_rhs,
                              std::vector<std::vector<SC>> & rhs_recv,
                        Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  const int numActive = activeSubs.size();
  my_recv_PIDs_rhs = myReceives(activeSubs, targetMPIs, comm);
  my_send_PIDs_rhs.resize(numActive);
  for (int i=0; i<numActive; i++) {
    my_send_PIDs_rhs[i] = targetMPIs[activeSubs[i]];
  }
  const int num_recvs = my_recv_PIDs_rhs.size();
  // number of rows
  std::vector<std::vector<int>> num_rows_recv_rhs(num_recvs);
  for (int i=0; i<num_recvs; i++) num_rows_recv_rhs[i].resize(1);
  communicateData(num_rows_send_rhs, my_recv_PIDs_rhs, my_send_PIDs_rhs, num_rows_recv_rhs, comm);
  // rhs values
  rhs_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) rhs_recv[i].resize(num_rows_recv_rhs[i][0]);
}

void communicateRhsData(const std::vector<int> & activeSubs,
                        const std::vector<int> & targetMPIs,
                        const std::vector<std::vector<int>> & num_rows_send,
                        const std::vector<std::vector<int>> & row_GIDs_send,
                        std::vector<std::vector<int>> & row_GIDs_recv,
                        std::vector<int> & my_send_PIDs,
                        std::vector<int> & my_recv_PIDs,
                        Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  const int numActive = activeSubs.size();
  my_recv_PIDs = myReceives(activeSubs, targetMPIs, comm);
  my_send_PIDs.resize(numActive);
  for (int i=0; i<numActive; i++) {
    my_send_PIDs[i] = targetMPIs[activeSubs[i]];
  }

  const int num_recvs = my_recv_PIDs.size();
  // number of rows
  std::vector<std::vector<int>> num_rows_recv(num_recvs);
  for (int i=0; i<num_recvs; i++) num_rows_recv[i].resize(1);
  communicateData(num_rows_send, my_recv_PIDs, my_send_PIDs, num_rows_recv, comm);
  // row numbers
  row_GIDs_recv.resize(num_recvs);
  for (int i=0; i<num_recvs; i++) row_GIDs_recv[i].resize(num_rows_recv[i][0]);
  communicateData(row_GIDs_send, my_recv_PIDs, my_send_PIDs, row_GIDs_recv, comm);
}

template <typename SC>
void communicateMatrixValuesB(const int level, const std::vector<SC> & values,
                              std::vector<std::vector<std::vector<int>>> & values_send_B_index,
                              std::vector<std::vector<std::vector<SC>>> & values_send_B,
                              std::vector<std::vector<std::vector<SC>>> & values_recv_B,
                              std::vector<std::vector<int>> & my_send_PIDs_B,
                              std::vector<std::vector<int>> & my_recv_PIDs_B,
                              Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  std::vector<std::vector<SC>>  & values_send_l = values_send_B[level];
  std::vector<std::vector<int>> & indices = values_send_B_index[level];
  const int num_send = values_send_l.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<values_send_l[i].size(); j++) {
      values_send_l[i][j] = values[indices[i][j]];
    }
  }
  communicateData(values_send_B[level], my_recv_PIDs_B[level], my_send_PIDs_B[level],
                  values_recv_B[level], comm);
}

template <typename SC>
void communicateRhsValuesB(const int level, const std::vector<SC> & rhs,
                           std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
                           std::vector<std::vector<std::vector<SC>>> & rhs_send_sep,
                           std::vector<std::vector<std::vector<SC>>> & rhs_recv_sep,
                           std::vector<std::vector<int>> & my_send_PIDs_sep,
                           std::vector<std::vector<int>> & my_recv_PIDs_sep,
                           Teuchos::RCP<Teuchos::MpiComm<int>> comm)
{
  std::vector<std::vector<SC>>  & rhs_send_l = rhs_send_sep[level];
  std::vector<std::vector<int>> & indices = rhs_send_sep_index[level];
  const int num_send = rhs_send_l.size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<rhs_send_l[i].size(); j++) {
      rhs_send_l[i][j] = rhs[indices[i][j]];
    }
  }
  communicateData(rhs_send_sep[level], my_recv_PIDs_sep[level],
                  my_send_PIDs_sep[level], rhs_recv_sep[level], comm);
}

template <typename SC>
void communicate_solution(const int level, std::vector<SC> & sol,
                          std::vector<std::vector<SC>> & AS_rhs,
                          std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
                          std::vector<std::vector<std::vector<int>>> & rhs_recv_sep_index,
                          std::vector<std::vector<std::vector<SC>>> & rhs_send_sep,
                          std::vector<std::vector<std::vector<SC>>> & rhs_recv_sep,
                          std::vector<std::vector<int>> & my_send_PIDs_sep,
                          std::vector<std::vector<int>> & my_recv_PIDs_sep,
                          Teuchos::RCP<Teuchos::MpiComm<int>> comm)
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
                  my_recv_PIDs_sep[level], rhs_send_sep[level], comm, reverse_comm);
  // unload separator solutions back into sol
  const int num_send = rhs_send_sep[level].size();
  for (int i=0; i<num_send; i++) {
    for (size_t j=0; j<rhs_send_sep_index[level][i].size(); j++) {
      const int index = rhs_send_sep_index[level][i][j];
      sol[index] = rhs_send_sep[level][i][j];
    }
  }
}


// ..ETI..
template
void communicateData<int>(const std::vector<std::vector<int>> & data_send,
                          const std::vector<int> & my_recv_PIDs,
                          const std::vector<int> & my_send_PIDs,
                                std::vector<std::vector<int>> & data_recv,
                                Teuchos::RCP<Teuchos::MpiComm<int>> comm,
                          const bool reverse_comm);
template
void communicateData<double>(const std::vector<std::vector<double>> & data_send,
                             const std::vector<int> & my_recv_PIDs,
                             const std::vector<int> & my_send_PIDs,
                                   std::vector<std::vector<double>> & data_recv,
                                   Teuchos::RCP<Teuchos::MpiComm<int>> comm,
                             const bool reverse_comm);
template
void communicateMatrixData<double>(const std::vector<int> & activeSubs,
                                   const std::vector<std::vector<int>>     & num_rows_send,
                                   const std::vector<std::vector<int>>     & row_GIDs_send,
                                   const std::vector<std::vector<int>>     & column_counts_send,
                                   const std::vector<std::vector<int>>     & column_GIDs_send,
                                   const std::vector<std::vector<double>>  & values_send_here,
                                         std::vector<std::vector<int>>     & num_rows_recv,
                                         std::vector<std::vector<int>>     & row_GIDs_recv,
                                         std::vector<std::vector<int>>     & column_counts_recv,
                                         std::vector<std::vector<int>>     & column_GIDs_recv,
                                         std::vector<std::vector<double>>  & values_recv_here,
                                         std::vector<int> & my_send_PIDs,
                                         std::vector<int> & my_recv_PIDs,
                                   const std::vector<int> & targetMPIs,
                                   Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateMatrixValues<double>(const std::vector<double> & values,
                                     const std::vector<std::vector<int>> & values_send_index,
                                           std::vector<int> & my_send_PIDs_sub,
                                           std::vector<std::vector<double>> & values_send,
                                     const std::vector<int> & my_recv_PIDs_sub,
                                           std::vector<std::vector<double>> & values_recv,
                                     Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateRhsData<double>(const std::vector<int> & activeSubs,
                                const std::vector<int> & targetMPIs,
                                const std::vector<std::vector<int>> & num_rows_send_rhs,
                                      std::vector<int> & my_send_PIDs_rhs,
                                      std::vector<int> & my_recv_PIDs_rhs,
                                      std::vector<std::vector<double>> & rhs_recv,
                                Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateMatrixValuesB<double>(const int level, const std::vector<double> & values,
                                      std::vector<std::vector<std::vector<int>>> & values_send_B_index,
                                      std::vector<std::vector<std::vector<double>>> & values_send_B,
                                      std::vector<std::vector<std::vector<double>>> & values_recv_B,
                                      std::vector<std::vector<int>> & my_send_PIDs_B,
                                      std::vector<std::vector<int>> & my_recv_PIDs_B,
                                      Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateRhsValuesB<double>(const int level, const std::vector<double> & rhs,
                                   std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
                                   std::vector<std::vector<std::vector<double>>> & rhs_send_sep,
                                   std::vector<std::vector<std::vector<double>>> & rhs_recv_sep,
                                   std::vector<std::vector<int>> & my_send_PIDs_sep,
                                   std::vector<std::vector<int>> & my_recv_PIDs_sep,
                                   Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicate_solution<double>(const int level, std::vector<double> & sol,
                                  std::vector<std::vector<double>> & AS_rhs,
                                  std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
                                  std::vector<std::vector<std::vector<int>>> & rhs_recv_sep_index,
                                  std::vector<std::vector<std::vector<double>>> & rhs_send_sep,
                                  std::vector<std::vector<std::vector<double>>> & rhs_recv_sep,
                                  std::vector<std::vector<int>> & my_send_PIDs_sep,
                                  std::vector<std::vector<int>> & my_recv_PIDs_sep,
                                  Teuchos::RCP<Teuchos::MpiComm<int>> comm);
#ifdef HAVE_TEUCHOS_INST_FLOAT
template
void communicateData<float>(const std::vector<std::vector<float>> & data_send,
                            const std::vector<int> & my_recv_PIDs,
                            const std::vector<int> & my_send_PIDs,
                                  std::vector<std::vector<float>> & data_recv,
                                  Teuchos::RCP<Teuchos::MpiComm<int>> comm,
                            const bool reverse_comm);
template
void communicateMatrixData<float>(const std::vector<int> & activeSubs,
                                  const std::vector<std::vector<int>>    & num_rows_send,
                                  const std::vector<std::vector<int>>    & row_GIDs_send,
                                  const std::vector<std::vector<int>>    & column_counts_send,
                                  const std::vector<std::vector<int>>    & column_GIDs_send,
                                  const std::vector<std::vector<float>>  & values_send_here,
                                        std::vector<std::vector<int>>    & num_rows_recv,
                                        std::vector<std::vector<int>>    & row_GIDs_recv,
                                        std::vector<std::vector<int>>    & column_counts_recv,
                                        std::vector<std::vector<int>>    & column_GIDs_recv,
                                        std::vector<std::vector<float>>  & values_recv_here,
                                        std::vector<int> & my_send_PIDs,
                                        std::vector<int> & my_recv_PIDs,
                                  const std::vector<int> & targetMPIs,
template
void communicateMatrixValues<float>(const std::vector<float> & values,
                                    const std::vector<std::vector<int>> & values_send_index,
                                          std::vector<int> & my_send_PIDs_sub,
                                          std::vector<std::vector<float>> & values_send,
                                    const std::vector<int> & my_recv_PIDs_sub,
                                          std::vector<std::vector<float>> & values_recv,
                                    Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateRhsData<float>(const std::vector<int> & activeSubs,
                               const std::vector<int> & targetMPIs,
                               const std::vector<std::vector<int>> & num_rows_send_rhs,
                                     std::vector<int> & my_send_PIDs_rhs,
                                     std::vector<int> & my_recv_PIDs_rhs,
                                     std::vector<std::vector<float>> & rhs_recv,
                               Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateMatrixValuesB<float>(const int level, const std::vector<float> & values,
                                     std::vector<std::vector<std::vector<int>>> & values_send_B_index,
                                     std::vector<std::vector<std::vector<float>>> & values_send_B,
                                     std::vector<std::vector<std::vector<float>>> & values_recv_B,
                                     std::vector<std::vector<int>> & my_send_PIDs_B,
                                     std::vector<std::vector<int>> & my_recv_PIDs_B,
                                     Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicateRhsValuesB<float>(const int level, const std::vector<float> & rhs,
                                  std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
                                  std::vector<std::vector<std::vector<float>>> & rhs_send_sep,
                                  std::vector<std::vector<std::vector<float>>> & rhs_recv_sep,
                                  std::vector<std::vector<int>> & my_send_PIDs_sep,
                                  std::vector<std::vector<int>> & my_recv_PIDs_sep,
                                  Teuchos::RCP<Teuchos::MpiComm<int>> comm);
template
void communicate_solution<float>(const int level, std::vector<float> & sol,
                                 std::vector<std::vector<float>> & AS_rhs,
                                 std::vector<std::vector<std::vector<int>>> & rhs_send_sep_index,
                                 std::vector<std::vector<std::vector<int>>> & rhs_recv_sep_index,
                                 std::vector<std::vector<std::vector<float>>> & rhs_send_sep,
                                 std::vector<std::vector<std::vector<float>>> & rhs_recv_sep,
                                 std::vector<std::vector<int>> & my_send_PIDs_sep,
                                 std::vector<std::vector<int>> & my_recv_PIDs_sep,
                                 Teuchos::RCP<Teuchos::MpiComm<int>> comm);
#endif
