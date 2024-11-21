// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_MATRIXMATRIX_CUDA_DEF_HPP
#define TPETRA_MATRIXMATRIX_CUDA_DEF_HPP

#include "Tpetra_Details_IntRowPtrHelper.hpp"

#ifdef HAVE_TPETRA_INST_CUDA
namespace Tpetra {
namespace MMdetails {

/*********************************************************************************************************/
// MMM KernelWrappers for Partial Specialization to CUDA
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
struct KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType> {
    static inline void mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                  const LocalOrdinalViewType & Acol2Brow,
                                                  const LocalOrdinalViewType & Acol2Irow,
                                                  const LocalOrdinalViewType & Bcol2Ccol,
                                                  const LocalOrdinalViewType & Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                  const std::string& label = std::string(),
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

#if 1
    static inline void mult_A_B_newmatrix_kernel_wrapper_host(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                  const LocalOrdinalViewType & Acol2Brow,
                                                  const LocalOrdinalViewType & Acol2Irow,
                                                  const LocalOrdinalViewType & Bcol2Ccol,
                                                  const LocalOrdinalViewType & Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                  const std::string& label = std::string(),
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);
#endif

   static inline void mult_A_B_reuse_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                  const LocalOrdinalViewType & Acol2Brow,
                                                  const LocalOrdinalViewType & Acol2Irow,
                                                  const LocalOrdinalViewType & Bcol2Ccol,
                                                  const LocalOrdinalViewType & Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                  const std::string& label = std::string(),
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

};

// Jacobi KernelWrappers for Partial Specialization to Cuda
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal, class LocalOrdinalViewType>
struct KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType> {
    static inline void jacobi_A_B_newmatrix_kernel_wrapper(Scalar omega,
                                                           const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> & Dinv,
                                                           CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                           CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                           const LocalOrdinalViewType & Acol2Brow,
                                                           const LocalOrdinalViewType & Acol2Irow,
                                                           const LocalOrdinalViewType & Bcol2Ccol,
                                                           const LocalOrdinalViewType & Icol2Ccol,
                                                           CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                           Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                           const std::string& label = std::string(),
                                                           const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

    static inline void jacobi_A_B_reuse_kernel_wrapper(Scalar omega,
                                                       const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> & Dinv,
                                                       CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                       CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                       const LocalOrdinalViewType & Acol2Brow,
                                                       const LocalOrdinalViewType & Acol2Irow,
                                                       const LocalOrdinalViewType & Bcol2Ccol,
                                                       const LocalOrdinalViewType & Icol2Ccol,
                                                       CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                       Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                       const std::string& label = std::string(),
                                                       const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

    static inline void jacobi_A_B_newmatrix_KokkosKernels(Scalar omega,
                                                          const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> & Dinv,
                                                          CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                          CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                          const LocalOrdinalViewType & Acol2Brow,
                                                          const LocalOrdinalViewType & Acol2Irow,
                                                          const LocalOrdinalViewType & Bcol2Ccol,
                                                          const LocalOrdinalViewType & Icol2Ccol,
                                                          CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                          Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                          const std::string& label = std::string(),
                                                          const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);
};


/*********************************************************************************************************/
// AB NewMatrix Kernel wrappers (KokkosKernels/CUDA Version)
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType>::mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & Acol2Brow,
                                                                                               const LocalOrdinalViewType & Acol2Irow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {


#if 1//def HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<TimeMonitor> MM = rcp(new TimeMonitor(*(TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaWrapper")))));
#endif
  // Node-specific code
  typedef Tpetra::KokkosCompat::KokkosCudaWrapperNode Node;
  std::string nodename("Cuda");

  // Lots and lots of typedefs
  using Teuchos::RCP;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_device_type KCRS;
  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  using int_view_t = Kokkos::View<int *, typename lno_view_t::array_layout, typename lno_view_t::memory_space, typename lno_view_t::memory_traits>;
  typedef typename graph_t::row_map_type::const_type  c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;
  //typedef typename graph_t::row_map_type::const_type lno_view_t_const;

  // Options
  int team_work_size = 16;  // Defaults to 16 as per Deveci 12/7/16 - csiefer
  std::string myalg("SPGEMM_KK_MEMORY");
  if(!params.is_null()) {
    if(params->isParameter("cuda: algorithm"))
      myalg = params->get("cuda: algorithm",myalg);
    if(params->isParameter("cuda: team work size"))
      team_work_size = params->get("cuda: team work size",team_work_size);
  }

  // KokkosKernelsHandle
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
       typename lno_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type,
       typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space > KernelHandle;
  using IntKernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
       typename int_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type,
       typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space >;

  // Grab the  Kokkos::SparseCrsMatrices
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixDevice();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixDevice();

  c_lno_view_t Arowptr = Amat.graph.row_map,
               Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries,
                       Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values,
                      Bvals = Bmat.values;

  // Get the algorithm mode
  std::string alg = nodename+std::string(" algorithm");
  //  printf("DEBUG: Using kernel: %s\n",myalg.c_str());
  if(!params.is_null() && params->isParameter(alg)) myalg = params->get(alg,myalg);
  KokkosSparse::SPGEMMAlgorithm alg_enum = KokkosSparse::StringToSPGEMMAlgorithm(myalg);

  // Merge the B and Bimport matrices
  KCRS Bmerged = Tpetra::MMdetails::merge_matrices(Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C.getColMap()->getLocalNumElements());


  // if Kokkos Kernels is going to use the cuSparse TPL for SpGEMM, this matrix
  // needs to be sorted
  // Try to mirror the Kokkos Kernels internal SpGEMM TPL use logic
  // inspired by https://github.com/trilinos/Trilinos/pull/11709
  // and https://github.com/kokkos/kokkos-kernels/pull/2008
#if defined(KOKKOS_ENABLE_CUDA) \
 && defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE) \
 && ((CUDA_VERSION < 11000) || (CUDA_VERSION >= 11040))
  if constexpr (std::is_same_v<typename device_t::execution_space, Kokkos::Cuda>) {
    if (!KokkosSparse::isCrsGraphSorted(Bmerged.graph.row_map, Bmerged.graph.entries)) {
      KokkosSparse::sort_crs_matrix(Bmerged);
    }
  }
#endif

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaCore"))));
#endif

  // Do the multiply on whatever we've got
  typename KernelHandle::nnz_lno_t AnumRows = Amat.numRows();
  typename KernelHandle::nnz_lno_t BnumRows = Bmerged.numRows();
  typename KernelHandle::nnz_lno_t BnumCols = Bmerged.numCols();

  // regardless of whether integer row ptrs are used, need to ultimately produce the row pointer, entries, and values of the expected types
  lno_view_t      row_mapC (Kokkos::ViewAllocateWithoutInitializing("non_const_lno_row"), AnumRows + 1);
  lno_nnz_view_t  entriesC;
  scalar_view_t   valuesC;

  Tpetra::Details::IntRowPtrHelper<decltype(Bmerged)> irph(Bmerged.nnz(), Bmerged.graph.row_map);
  const bool useIntRowptrs = 
     irph.shouldUseIntRowptrs() && 
     Aview.origMatrix->getApplyHelper()->shouldUseIntRowptrs();

  if (useIntRowptrs) {
Teuchos::RCP< Teuchos::Time > locTimer_ = Teuchos::TimeMonitor::getNewCounter ("TpetraExt_MatrixMatrix::CUDA::spgemm-1");
Teuchos::TimeMonitor LocTimer(*locTimer_);

    IntKernelHandle kh;
    kh.create_spgemm_handle(alg_enum);
    kh.set_team_work_size(team_work_size);

    int_view_t int_row_mapC (Kokkos::ViewAllocateWithoutInitializing("non_const_int_row"), AnumRows + 1);

    auto Aint = Aview.origMatrix->getApplyHelper()->getIntRowptrMatrix(Amat);
    auto Bint = irph.getIntRowptrMatrix(Bmerged);
    KokkosSparse::Experimental::spgemm_symbolic(&kh,AnumRows,BnumRows,BnumCols,Aint.graph.row_map,Aint.graph.entries,false,Bint.graph.row_map,Bint.graph.entries,false, int_row_mapC);

    size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
      valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
    }
    KokkosSparse::Experimental::spgemm_numeric(&kh,AnumRows,BnumRows,BnumCols,Aint.graph.row_map,Aint.graph.entries,Aint.values,false,Bint.graph.row_map,Bint.graph.entries,Bint.values,false,int_row_mapC,entriesC,valuesC);
    // transfer the integer rowptrs back to the correct rowptr type
    Kokkos::parallel_for(int_row_mapC.size(), KOKKOS_LAMBDA(int i){ row_mapC(i) = int_row_mapC(i);});
    kh.destroy_spgemm_handle();

  } else {
Teuchos::RCP< Teuchos::Time > locTimer_ = Teuchos::TimeMonitor::getNewCounter ("TpetraExt_MatrixMatrix::CUDA::spgemm-2");
Teuchos::TimeMonitor LocTimer(*locTimer_);
    KernelHandle kh;
    kh.create_spgemm_handle(alg_enum);
    kh.set_team_work_size(team_work_size);

    KokkosSparse::Experimental::spgemm_symbolic(&kh,AnumRows,BnumRows,BnumCols,Amat.graph.row_map,Amat.graph.entries,false,Bmerged.graph.row_map,Bmerged.graph.entries,false,row_mapC);

    size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
      valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
    }

    KokkosSparse::Experimental::spgemm_numeric(&kh,AnumRows,BnumRows,BnumCols,Amat.graph.row_map,Amat.graph.entries,Amat.values,false,Bmerged.graph.row_map,Bmerged.graph.entries,Bmerged.values,false,row_mapC,entriesC,valuesC);

    kh.destroy_spgemm_handle();
  }


#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaSort"))));
#endif

  // Sort & set values
  if (params.is_null() || params->get("sort entries",true))
    Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
  C.setAllValues(row_mapC,entriesC,valuesC);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaESFC"))));
#endif

  // Final Fillcomplete
  RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));
  RCP<const Export<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > dummyExport;
  C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
}


#if 1
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType>::mult_A_B_newmatrix_kernel_wrapper_host(
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & targetMapToOrigRow,
                                                                                               const LocalOrdinalViewType & targetMapToImportRow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params)
{
#if 1//def HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix Cuda-Host"))));
#endif

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;

  using Node = Tpetra::KokkosCompat::KokkosCudaWrapperNode;


  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>     map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getLocalNumRows();
  size_t n = Ccolmap->getLocalNumElements();
  size_t b_max_nnz_per_row = Bview.origMatrix->getLocalMaxNumRowEntries();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
    b_max_nnz_per_row = std::max(b_max_nnz_per_row,Bview.importMatrix->getLocalMaxNumRowEntries());
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  RCP<TimeMonitor> MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix SerialCore - Compare"))));
#endif

  // Classic csr assembly (low memory edition)
  //
  // mfh 27 Sep 2016: C_estimate_nnz does not promise an upper bound.
  // The method loops over rows of A, and may resize after processing
  // each row.  Chris Siefert says that this reflects experience in
  // ML; for the non-threaded case, ML found it faster to spend less
  // effort on estimation and risk an occasional reallocation.
  size_t CSR_alloc = std::max(C_estimate_nnz(*Aview.origMatrix, *Bview.origMatrix), n);
  lno_view_t Crowptr(Kokkos::ViewAllocateWithoutInitializing("Crowptr"),m+1);
  lno_nnz_view_t Ccolind(Kokkos::ViewAllocateWithoutInitializing("Ccolind"),CSR_alloc);
  scalar_view_t Cvals(Kokkos::ViewAllocateWithoutInitializing("Cvals"),CSR_alloc);

  // mfh 27 Sep 2016: The c_status array is an implementation detail
  // of the local sparse matrix-matrix multiply routine.

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();
  std::vector<size_t> c_status(n, ST_INVALID);

  // mfh 27 Sep 2016: Here is the local sparse matrix-matrix multiply
  // routine.  The routine computes C := A * (B_local + B_remote).
  //
  // For column index Aik in row i of A, targetMapToOrigRow[Aik] tells
  // you whether the corresponding row of B belongs to B_local
  // ("orig") or B_remote ("Import").

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {
    // mfh 27 Sep 2016: m is the number of rows in the input matrix A
    // on the calling process.
    Crowptr[i] = CSR_ip;

    // mfh 27 Sep 2016: For each entry of A in the current row of A
    for (size_t k = Arowptr[i]; k < Arowptr[i+1]; k++) {
      LO Aik  = Acolind[k]; // local column index of current entry of A
      const SC Aval = Avals[k];   // value of current entry of A
      if (Aval == SC_ZERO)
        continue; // skip explicitly stored zero values in A

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
        // corresponding to the current entry of A is populated, then
        // the corresponding row of B is in B_local (i.e., it lives on
        // the calling process).

        // Local matrix
        size_t Bk = static_cast<size_t> (targetMapToOrigRow[Aik]);

        // mfh 27 Sep 2016: Go through all entries in that row of B_local.
        for (size_t j = Browptr[Bk]; j < Browptr[Bk+1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
            // New entry
            c_status[Cij]   = CSR_ip;
            Ccolind[CSR_ip] = Cij;
            Cvals[CSR_ip]   = Aval*Bvals[j];
            CSR_ip++;

          } else {
            Cvals[c_status[Cij]] += Aval*Bvals[j];
          }
        }

      } else {
        // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
        // corresponding to the current entry of A NOT populated (has
        // a flag "invalid" value), then the corresponding row of B is
        // in B_local (i.e., it lives on the calling process).

        // Remote matrix
        size_t Ik = static_cast<size_t> (targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik+1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip){
            // New entry
            c_status[Cij]   = CSR_ip;
            Ccolind[CSR_ip] = Cij;
            Cvals[CSR_ip]   = Aval*Ivals[j];
            CSR_ip++;
          } else {
            Cvals[c_status[Cij]] += Aval*Ivals[j];
          }
        }
      }
    }

    // Resize for next pass if needed
    if (i+1 < m && CSR_ip + std::min(n,(Arowptr[i+2]-Arowptr[i+1])*b_max_nnz_per_row) > CSR_alloc) {
      CSR_alloc *= 2;
      Kokkos::resize(Ccolind,CSR_alloc);
      Kokkos::resize(Cvals,CSR_alloc);
    }
    OLD_ip = CSR_ip;
  }

  Crowptr[m] = CSR_ip;

  // Downward resize
  Kokkos::resize(Ccolind,CSR_ip);
  Kokkos::resize(Cvals,CSR_ip);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  {
  auto MM3(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix Final Sort")));
#endif

  // Final sort & set of CRS arrays
  if (params.is_null() || params->get("sort entries",true))
    Import_Util::sortCrsEntries(Crowptr,Ccolind, Cvals);
  C.setAllValues(Crowptr,Ccolind, Cvals);


#ifdef HAVE_TPETRA_MMM_TIMINGS
  }
  auto MM4(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix ESFC")));
  {
#endif

  // Final FillComplete
  //
  // mfh 27 Sep 2016: So-called "expert static fill complete" bypasses
  // Import (from domain Map to column Map) construction (which costs
  // lots of communication) by taking the previously constructed
  // Import object.  We should be able to do this without interfering
  // with the implementation of the local part of sparse matrix-matrix
  // multply above.
  RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));
  RCP<const Export<LO,GO,NO> > dummyExport;
  C.expertStaticFillComplete(Bview. origMatrix->getDomainMap(), Aview. origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
#ifdef HAVE_TPETRA_MMM_TIMINGS
  }
  MM2 = Teuchos::null;
#endif
}
#endif

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType>::mult_A_B_reuse_kernel_wrapper(
               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & targetMapToOrigRow_dev,
                                                                                               const LocalOrdinalViewType & targetMapToImportRow_dev,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol_dev,
                                                                                               const LocalOrdinalViewType & Icol2Ccol_dev,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

  // FIXME: Right now, this is a cut-and-paste of the serial kernel
  typedef Tpetra::KokkosCompat::KokkosCudaWrapperNode Node;

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Reuse SerialCore"))));
  Teuchos::RCP<Teuchos::TimeMonitor> MM2;
#endif
  using Teuchos::RCP;
  using Teuchos::rcp;


  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>     map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();

  // Since this is being run on Cuda, we need to fence because the below code will use UVM
  // typename graph_t::execution_space().fence();
  
  // KDDKDD UVM Without UVM, need to copy targetMap arrays to host.
  // KDDKDD UVM Ideally, this function would run on device and use 
  // KDDKDD UVM KokkosKernels instead of this host implementation.
  auto targetMapToOrigRow = 
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           targetMapToOrigRow_dev);
  auto targetMapToImportRow = 
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           targetMapToImportRow_dev);
  auto Bcol2Ccol =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           Bcol2Ccol_dev);
  auto Icol2Ccol = 
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           Icol2Ccol_dev);

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getLocalNumRows();
  size_t n = Ccolmap->getLocalNumElements();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixHost();
  const KCRS & Cmat = C.getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map,
               Browptr = Bmat.graph.row_map,
               Crowptr = Cmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, 
                       Bcolind = Bmat.graph.entries, 
                       Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM2 = Teuchos::null; MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix SerialCore - Compare"))));
#endif

  // Classic csr assembly (low memory edition)
  // mfh 27 Sep 2016: The c_status array is an implementation detail
  // of the local sparse matrix-matrix multiply routine.

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  std::vector<size_t> c_status(n, ST_INVALID);

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {
    // First fill the c_status array w/ locations where we're allowed to
    // generate nonzeros for this row
    OLD_ip = Crowptr[i];
    CSR_ip = Crowptr[i+1];
    for (size_t k = OLD_ip; k < CSR_ip; k++) {
      c_status[Ccolind[k]] = k;

      // Reset values in the row of C
      Cvals[k] = SC_ZERO;
    }

    for (size_t k = Arowptr[i]; k < Arowptr[i+1]; k++) {
      LO Aik  = Acolind[k];
      const SC Aval = Avals[k];
      if (Aval == SC_ZERO)
        continue;

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // Local matrix
        size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow[Aik]);

        for (size_t j = Browptr[Bk]; j < Browptr[Bk+1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
            "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

          Cvals[c_status[Cij]] += Aval * Bvals[j];
        }

      } else {
        // Remote matrix
        size_t Ik = Teuchos::as<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik+1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
            "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

          Cvals[c_status[Cij]] += Aval * Ivals[j];
        }
      }
    }
  }

  C.fillComplete(C.getDomainMap(), C.getRangeMap());
}

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType>::jacobi_A_B_newmatrix_kernel_wrapper(Scalar omega,
                                                                                               const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> & Dinv,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & Acol2Brow,
                                                                                               const LocalOrdinalViewType & Acol2Irow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

#ifdef HAVE_TPETRA_MMM_TIMINGS
    std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
    using Teuchos::TimeMonitor;
    Teuchos::RCP<TimeMonitor> MM;
#endif

  // Node-specific code
  using Teuchos::RCP;

  // Options
  //int team_work_size = 16;  // Defaults to 16 as per Deveci 12/7/16 - csiefer // unreferenced
  std::string myalg("KK");
  if(!params.is_null()) {
    if(params->isParameter("cuda: jacobi algorithm"))
      myalg = params->get("cuda: jacobi algorithm",myalg);
  }

  if(myalg == "MSAK") {
    ::Tpetra::MatrixMatrix::ExtraKernels::jacobi_A_B_newmatrix_MultiplyScaleAddKernel(omega,Dinv,Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C,Cimport,label,params);
  }
  else if(myalg == "KK") {
    jacobi_A_B_newmatrix_KokkosKernels(omega,Dinv,Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C,Cimport,label,params);
  }
  else {
    throw std::runtime_error("Tpetra::MatrixMatrix::Jacobi newmatrix unknown kernel");
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix CudaESFC"))));
#endif

  // Final Fillcomplete
  RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));

  // NOTE: MSAK already fillCompletes, so we have to check here
  if(!C.isFillComplete()) {
    RCP<const Export<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > dummyExport;
    C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
  }

}



/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType>::jacobi_A_B_reuse_kernel_wrapper(Scalar omega,
                                                                                               const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> & Dinv,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & targetMapToOrigRow_dev,
                                                                                               const LocalOrdinalViewType & targetMapToImportRow_dev,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol_dev,
                                                                                               const LocalOrdinalViewType & Icol2Ccol_dev,
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

  // FIXME: Right now, this is a cut-and-paste of the serial kernel
  typedef Tpetra::KokkosCompat::KokkosCudaWrapperNode Node;

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse CudaCore"))));
  Teuchos::RCP<Teuchos::TimeMonitor> MM2;
#endif
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;
  typedef typename scalar_view_t::memory_space scalar_memory_space;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>     map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();

  // Since this is being run on Cuda, we need to fence because the below host code will use UVM
  // KDDKDD typename graph_t::execution_space().fence();

  // KDDKDD UVM Without UVM, need to copy targetMap arrays to host.
  // KDDKDD UVM Ideally, this function would run on device and use 
  // KDDKDD UVM KokkosKernels instead of this host implementation.
  auto targetMapToOrigRow = 
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           targetMapToOrigRow_dev);
  auto targetMapToImportRow = 
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           targetMapToImportRow_dev);
  auto Bcol2Ccol =
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           Bcol2Ccol_dev);
  auto Icol2Ccol = 
       Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                           Icol2Ccol_dev);
  
 
  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getLocalNumRows();
  size_t n = Ccolmap->getLocalNumElements();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrixHost();
  const KCRS & Cmat = C.getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map, Crowptr = Cmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries, Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr = lclB.graph.row_map;
    Icolind = lclB.graph.entries;
    Ivals   = lclB.values;
  }

  // Jacobi-specific inner stuff
  auto Dvals =
       Dinv.template getLocalView<scalar_memory_space>(Access::ReadOnly);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM2 = Teuchos::null; MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse CudaCore - Compare"))));
#endif

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  std::vector<size_t> c_status(n, ST_INVALID);

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {

    // First fill the c_status array w/ locations where we're allowed to
    // generate nonzeros for this row
    OLD_ip = Crowptr[i];
    CSR_ip = Crowptr[i+1];
    for (size_t k = OLD_ip; k < CSR_ip; k++) {
      c_status[Ccolind[k]] = k;

      // Reset values in the row of C
      Cvals[k] = SC_ZERO;
    }

    SC minusOmegaDval = -omega*Dvals(i,0);

    // Entries of B
    for (size_t j = Browptr[i]; j < Browptr[i+1]; j++) {
      Scalar Bval = Bvals[j];
      if (Bval == SC_ZERO)
        continue;
      LO Bij = Bcolind[j];
      LO Cij = Bcol2Ccol[Bij];

      TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
        std::runtime_error, "Trying to insert a new entry into a static graph");

      Cvals[c_status[Cij]] = Bvals[j];
    }

    // Entries of -omega * Dinv * A * B
    for (size_t k = Arowptr[i]; k < Arowptr[i+1]; k++) {
      LO Aik  = Acolind[k];
      const SC Aval = Avals[k];
      if (Aval == SC_ZERO)
        continue;

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // Local matrix
        size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow[Aik]);

        for (size_t j = Browptr[Bk]; j < Browptr[Bk+1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry into a static graph");

          Cvals[c_status[Cij]] += minusOmegaDval * Aval * Bvals[j];
        }

      } else {
        // Remote matrix
        size_t Ik = Teuchos::as<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik+1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
            std::runtime_error, "Trying to insert a new entry into a static graph");

          Cvals[c_status[Cij]] += minusOmegaDval * Aval * Ivals[j];
        }
      }
    }
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM2= Teuchos::null;
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse ESFC"))));
#endif

  C.fillComplete(C.getDomainMap(), C.getRangeMap());

}

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers2<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode,LocalOrdinalViewType>::jacobi_A_B_newmatrix_KokkosKernels(Scalar omega,
												const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> & Dinv,
											        CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Aview,
												CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& Bview,
												const LocalOrdinalViewType & Acol2Brow,
												const LocalOrdinalViewType & Acol2Irow,
												const LocalOrdinalViewType & Bcol2Ccol,
												const LocalOrdinalViewType & Icol2Ccol,
												CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode>& C,
												Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > Cimport,
												const std::string& label,
												const Teuchos::RCP<Teuchos::ParameterList>& params) {

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<TimeMonitor> MM;
#endif

  // Check if the diagonal entries exist in debug mode
  const bool debug = Tpetra::Details::Behavior::debug();
  if(debug) {

    auto rowMap = Aview.origMatrix->getRowMap();
    Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosCudaWrapperNode> diags(rowMap);
    Aview.origMatrix->getLocalDiagCopy(diags);
    size_t diagLength = rowMap->getLocalNumElements();
    Teuchos::Array<Scalar> diagonal(diagLength);
    diags.get1dCopy(diagonal());

    for(size_t i = 0; i < diagLength; ++i) {
      TEUCHOS_TEST_FOR_EXCEPTION(diagonal[i] == Teuchos::ScalarTraits<Scalar>::zero(), 
				 std::runtime_error, 
				 "Matrix A has a zero/missing diagonal: " << diagonal[i] << std::endl <<
				 "KokkosKernels Jacobi-fused SpGEMM requires nonzero diagonal entries in A" << std::endl);
    }
  }

  // Usings
  using device_t = typename Tpetra::KokkosCompat::KokkosCudaWrapperNode::device_type;
  using matrix_t = typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode>::local_matrix_device_type;
  using graph_t = typename matrix_t::StaticCrsGraphType;
  using lno_view_t = typename graph_t::row_map_type::non_const_type;
  using int_view_t = Kokkos::View<int *,
                                  typename lno_view_t::array_layout,
                                  typename lno_view_t::memory_space,
                                  typename lno_view_t::memory_traits>;
  using c_lno_view_t = typename graph_t::row_map_type::const_type;
  using lno_nnz_view_t = typename graph_t::entries_type::non_const_type;
  using scalar_view_t = typename matrix_t::values_type::non_const_type;

  // KokkosKernels handle 
  using handle_t = typename KokkosKernels::Experimental::KokkosKernelsHandle<
    typename lno_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type,
    typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space >;

  using int_handle_t = typename KokkosKernels::Experimental::KokkosKernelsHandle<
    typename int_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type,
    typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space >;

  // Merge the B and Bimport matrices
  const matrix_t Bmerged = Tpetra::MMdetails::merge_matrices(Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C.getColMap()->getLocalNumElements());

  // Get the properties and arrays of input matrices
  const matrix_t & Amat = Aview.origMatrix->getLocalMatrixDevice();
  const matrix_t & Bmat = Bview.origMatrix->getLocalMatrixDevice();

  typename handle_t::nnz_lno_t AnumRows = Amat.numRows();
  typename handle_t::nnz_lno_t BnumRows = Bmerged.numRows();
  typename handle_t::nnz_lno_t BnumCols = Bmerged.numCols();

  // Arrays of the output matrix
  lno_view_t row_mapC (Kokkos::ViewAllocateWithoutInitializing("row_mapC"), AnumRows + 1);
  lno_nnz_view_t entriesC;
  scalar_view_t valuesC;

  // Options
  int team_work_size = 16; 
  std::string myalg("SPGEMM_KK_MEMORY");
  if(!params.is_null()) {
    if(params->isParameter("cuda: algorithm"))
      myalg = params->get("cuda: algorithm",myalg);
    if(params->isParameter("cuda: team work size"))
      team_work_size = params->get("cuda: team work size",team_work_size);
  }

  // Get the algorithm mode
  std::string alg("Cuda algorithm");
  if(!params.is_null() && params->isParameter(alg)) myalg = params->get(alg,myalg);
  KokkosSparse::SPGEMMAlgorithm alg_enum = KokkosSparse::StringToSPGEMMAlgorithm(myalg);

  // decide whether to use integer-typed row pointers for this spgemm
  Tpetra::Details::IntRowPtrHelper<decltype(Bmerged)> irph(Bmerged.nnz(), Bmerged.graph.row_map);
  const bool useIntRowptrs = 
    irph.shouldUseIntRowptrs() && 
    Aview.origMatrix->getApplyHelper()->shouldUseIntRowptrs();

  if (useIntRowptrs) {
    int_handle_t kh;
    kh.create_spgemm_handle(alg_enum);
    kh.set_team_work_size(team_work_size);

    int_view_t int_row_mapC (Kokkos::ViewAllocateWithoutInitializing("int_row_mapC"), AnumRows + 1);

    auto Aint = Aview.origMatrix->getApplyHelper()->getIntRowptrMatrix(Amat);
    auto Bint = irph.getIntRowptrMatrix(Bmerged);

    KokkosSparse::Experimental::spgemm_symbolic(&kh, AnumRows, BnumRows, BnumCols,
                  Aint.graph.row_map, Aint.graph.entries, false,
                  Bint.graph.row_map, Bint.graph.entries, false,
                  int_row_mapC);

    size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
      valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
    }

    // even though there is no TPL for this, we have to use the same handle that was used in the symbolic phase,
    // so need to have a special int-typed call for this as well.
    KokkosSparse::Experimental::spgemm_jacobi(&kh, AnumRows, BnumRows, BnumCols,
                Aint.graph.row_map, Aint.graph.entries, Amat.values, false,
                Bint.graph.row_map, Bint.graph.entries, Bint.values, false,
                int_row_mapC, entriesC, valuesC,
                omega, Dinv.getLocalViewDevice(Access::ReadOnly));
    // transfer the integer rowptrs back to the correct rowptr type
    Kokkos::parallel_for(int_row_mapC.size(), KOKKOS_LAMBDA(int i){ row_mapC(i) = int_row_mapC(i);});
    kh.destroy_spgemm_handle();
  } else {
    handle_t kh;
    kh.create_spgemm_handle(alg_enum);
    kh.set_team_work_size(team_work_size);

    KokkosSparse::Experimental::spgemm_symbolic(&kh, AnumRows, BnumRows, BnumCols,
                  Amat.graph.row_map, Amat.graph.entries, false,
                  Bmerged.graph.row_map, Bmerged.graph.entries, false,
                  row_mapC);

    size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
      valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
    }

    KokkosSparse::Experimental::spgemm_jacobi(&kh, AnumRows, BnumRows, BnumCols,
                Amat.graph.row_map, Amat.graph.entries, Amat.values, false,
                Bmerged.graph.row_map, Bmerged.graph.entries, Bmerged.values, false,
                row_mapC, entriesC, valuesC,
                omega, Dinv.getLocalViewDevice(Access::ReadOnly));
    kh.destroy_spgemm_handle();
  }



#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix CudaSort"))));
#endif

  // Sort & set values
  if (params.is_null() || params->get("sort entries",true))
    Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
  C.setAllValues(row_mapC,entriesC,valuesC);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = Teuchos::null; MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix CudaESFC"))));
#endif

  // Final Fillcomplete
  Teuchos::RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));
  Teuchos::RCP<const Export<LocalOrdinal,GlobalOrdinal,Tpetra::KokkosCompat::KokkosCudaWrapperNode> > dummyExport;
  C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
}

  }//MMdetails
}//Tpetra

#endif//CUDA

#endif
