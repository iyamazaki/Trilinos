// @HEADER
// *****************************************************************************
//           Amesos2: Templated Direct Sparse Solver Package
//
// Copyright 2011 NTESS and the Amesos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/**
   \file   Amesos2_D3S_def.hpp
   \author Siva Rajamanickam <srajama@sandia.gov>

   \brief  Definitions for the Amesos2 D3S solver interface
*/


#ifndef AMESOS2_D3S_DEF_HPP
#define AMESOS2_D3S_DEF_HPP

#include <Teuchos_Tuple.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include "Amesos2_SolverCore_def.hpp"
#include "Amesos2_D3S_decl.hpp"

namespace Amesos2 {


template <class Matrix, class Vector>
D3S<Matrix,Vector>::D3S(
  Teuchos::RCP<const Matrix> A,
  Teuchos::RCP<Vector>       X,
  Teuchos::RCP<const Vector> B )
  : SolverCore<Amesos2::D3S,Matrix,Vector>(A, X, B)
  , transFlag_(0)
  , is_contiguous_(true)
  , use_gather_(true)
  , solvername_("")
  , msg_level_(0)
  , num_threads_(1)
  , matching_option_(0)
  , reorder_option_(2)
  , debug_level_(0)
{
  // Matrix info
  Teuchos::RCP<const Teuchos::Comm<int> > matComm = this->matrixA_->getComm ();
  const global_ordinal_type indexBase = this->matrixA_->getRowMap ()->getIndexBase ();
  numRows_ = this->matrixA_->getLocalNumRows();
  numProcSolver_ = matComm->getSize();

  // rowmap for loadA (to have locally contiguous)
  d3s_rowmap_ =
    Teuchos::rcp (new map_type (this->globalNumRows_, numRows_, indexBase, matComm));
  d3s_contig_rowmap_ = Teuchos::rcp (new map_type (0, 0, indexBase, matComm));
  d3s_contig_colmap_ = Teuchos::rcp (new map_type (0, 0, indexBase, matComm));
  startGID_ = d3s_rowmap_->getMinGlobalIndex();

  // get MPI Comm
  TEUCHOS_TEST_FOR_EXCEPTION(
      matComm.is_null (), std::logic_error, "Amesos2::D3S "
      "constructor: The matrix's communicator is null!");
  Teuchos::RCP<const Teuchos::MpiComm<int> > matMpiComm =
    Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> > (matComm);
  TEUCHOS_TEST_FOR_EXCEPTION(
    matMpiComm.is_null (), std::logic_error, "Amesos2::D3S "
    "constructor: The matrix's communicator is not an MpiComm!");
  TEUCHOS_TEST_FOR_EXCEPTION(
    matMpiComm->getRawMpiComm ().is_null (), std::logic_error, "Amesos2::"
    "D3S constructor: The matrix's communicator claims to be a "
    "Teuchos::MpiComm<int>, but its getRawPtrComm() method returns "
    "Teuchos::null!  This means that the underlying MPI_Comm doesn't even "
    "exist, which likely implies that the Teuchos::MpiComm was constructed "
    "incorrectly.  It means something different than if the MPI_Comm were "
    "MPI_COMM_NULL.");
  MPI_Comm D3SComm = *(matMpiComm->getRawMpiComm ());
  D3SComm_ = MPI_Comm_c2f(D3SComm);

  solver = Teuchos::rcp (new D3Solver(D3SComm));
}


template <class Matrix, class Vector>
D3S<Matrix,Vector>::~D3S( )
{
}

template <class Matrix, class Vector>
bool
D3S<Matrix,Vector>::single_proc_optimization() const {
  return (this->root_ && (this->matrixA_->getComm()->getSize() == 1) && is_contiguous_);
}

template<class Matrix, class Vector>
int
D3S<Matrix,Vector>::preOrdering_impl()
{
  /* TODO: Define what it means for D3S
   */
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor preOrderTimer(this->timers_.preOrderTime_);
#endif

  return(0);
}


template <class Matrix, class Vector>
int
D3S<Matrix,Vector>::symbolicFactorization_impl()
{
  int info = 0;
  try {
    solver->setNumThreads(num_threads_);
    solver->setOrderingOption(matching_option_, reorder_option_);
    solver->setVerbose(msg_level_, debug_level_);
    solver->setInteriorSolverName(solvername_);

    int nnz = colind_view_.extent(0);
    std::vector<int> rowBegin(rowptr_view_.data(), rowptr_view_.data()+(numRows_+1));
    std::vector<int> columns (colind_view_.data(), colind_view_.data()+(nnz));
    info = solver->initialize(rowBegin, columns, startGID_, numProcSolver_);
  } catch (...) {
    info = -1;
  }

  /* All processes should have the same error code */
  Teuchos::broadcast(*(this->matrixA_->getComm()), 0, &info);

  TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error,
      "D3S symbolic factorization failed(info="+std::to_string(info)+")");

  return(0);
}


template <class Matrix, class Vector>
int
D3S<Matrix,Vector>::numericFactorization_impl()
{
  using Teuchos::as;

  // Cleanup old L and U matrices if we are not reusing a symbolic
  // factorization.  Stores and other data will be allocated in gstrf.
  // Only rank 0 has valid pointers, TODO: for D3S

  int info = 0;
  try { // Do factorization
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor numFactTimer(this->timers_.numFactTime_);
#endif
    int nnz = nzvals_view_.extent(0);
    std::vector<d3s_dtype> values(nzvals_view_.data(), nzvals_view_.data()+(nnz));
    info = function_map::factorize(solver, values);
  } catch (...) {
    info = -1;
  }

  /* All processes should have the same error code */
  Teuchos::broadcast(*(this->matrixA_->getComm()), 0, &info);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error,
      "D3S numeric factorization failed(info="+std::to_string(info)+")");

  return(info);
}

template <class Matrix, class Vector>
int
D3S<Matrix,Vector>::solve_impl(
 const Teuchos::Ptr<MultiVecAdapter<Vector> >  X,
 const Teuchos::Ptr<const MultiVecAdapter<Vector> > B) const
{
  using Teuchos::as;
  int ierr = 0; // returned error code

  // Get B data
  const local_ordinal_type ld_rhs = this->matrixA_->getLocalNumRows();
  nrhs_ = as<int>(X->getGlobalNumVectors());

  const size_t val_store_size = as<size_t>(ld_rhs * nrhs_);
  bvals_.resize(val_store_size);
  xvals_.resize(val_store_size);
  tvals_.resize(val_store_size);
  {
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor mvConvTimer( this->timers_.vecConvTime_ );
    Teuchos::TimeMonitor redistTimer( this->timers_.vecRedistTime_ );
#endif

    Util::get_1d_copy_helper<
      MultiVecAdapter<Vector>,
      d3s_dtype>::do_get(B, bvals_(),
        as<size_t>(ld_rhs),
        Teuchos::ptrInArg(*d3s_rowmap_));
  }

  try {
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor solveTimer(this->timers_.solveTime_);
#endif
    for (int j=0; j<nrhs_; j++) {
      std::vector<d3s_dtype> rhs (bvals_.getRawPtr()+(j*ld_rhs), bvals_.getRawPtr()+((j+1)*ld_rhs));
      std::vector<d3s_dtype> sol (tvals_.getRawPtr()+(j*ld_rhs), tvals_.getRawPtr()+((j+1)*ld_rhs));

      ierr = function_map::solve(solver, rhs, sol);

      //TODO
      for(int i=0; i<numRows_; i++) xvals_[i + j*ld_rhs] = sol[i];
    }
  } catch (...) {
    ierr = -1;
  }

  /* All processes should have the same error code */
  Teuchos::broadcast(*(this->matrixA_->getComm()), 0, &ierr);

  TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0, std::runtime_error,
      "D3S solve failed(ierr="+std::to_string(ierr)+")");

  /* Get values to X */
  {
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor redistTimer(this->timers_.vecRedistTime_);
#endif

    Util::put_1d_data_helper<
    MultiVecAdapter<Vector>,
      d3s_dtype>::do_put(X, xvals_(),
        as<size_t>(ld_rhs),
        Teuchos::ptrInArg(*d3s_rowmap_));
  }
  return(ierr);
}


template <class Matrix, class Vector>
bool
D3S<Matrix,Vector>::matrixShapeOK_impl() const
{
  // The D3S factorization routines can handle square as well as
  // rectangular matrices, but D3S can only apply the solve routines to
  // square matrices, so we check the matrix for squareness.
  return( this->matrixA_->getGlobalNumRows() == this->matrixA_->getGlobalNumCols() );
}


template <class Matrix, class Vector>
void
D3S<Matrix,Vector>::setParameters_impl(const Teuchos::RCP<Teuchos::ParameterList> & parameterList )
{
  using Teuchos::RCP;
  using Teuchos::getIntegralValue;
  using Teuchos::ParameterEntryValidator;

  RCP<const Teuchos::ParameterList> valid_params = getValidParameters_impl();

  transFlag_ = this->control_.useTranspose_ ? 1: 0;
  // The D3S transpose option can override the Amesos2 option
  if( parameterList->isParameter("Trans") ){
    RCP<const ParameterEntryValidator> trans_validator = valid_params->getEntry("Trans").validator();
    parameterList->getEntry("Trans").setValidator(trans_validator);

    transFlag_ = getIntegralValue<int>(*parameterList, "Trans");
  }

  if( parameterList->isParameter("MessageLevel") ){
    msg_level_ = parameterList->get<int>("MessageLevel");
  }
  if( parameterList->isParameter("DebugLevel") ){
    debug_level_ = parameterList->get<int>("DebugLevel");
  }

  if( parameterList->isParameter("InteriorSolverName") ){
    solvername_ = parameterList->get<std::string>("InteriorSolverName");
  }

  if( parameterList->isParameter("NumThreads") ){
    num_threads_ = parameterList->get<int>("NumThreads");
  }

  if( parameterList->isParameter("MatchingOption") ){
    matching_option_ = parameterList->get<int>("MatchingOption");
  }

  if( parameterList->isParameter("OrderingOption") ){
    reorder_option_ = parameterList->get<int>("OrderingOption");
  }

  if( parameterList->isParameter("IsContiguous") ){
    is_contiguous_ = parameterList->get<bool>("IsContiguous");
  }
}


template <class Matrix, class Vector>
Teuchos::RCP<const Teuchos::ParameterList>
D3S<Matrix,Vector>::getValidParameters_impl() const
{
  using std::string;
  using Teuchos::tuple;
  using Teuchos::ParameterList;
  using Teuchos::setStringToIntegralParameter;

  static Teuchos::RCP<const Teuchos::ParameterList> valid_params;

  if( is_null(valid_params) )
  {
    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();

    pl->set("Equil", true, "Whether to equilibrate the system before solve, does nothing now");
    pl->set("IsContiguous", true, "Whether GIDs contiguous");

    pl->set("InteriorSolverName", "", "Name of the solver used for solving interior/leaf problem");
    pl->set("MessageLevel", 0, "Message Level");
    pl->set("DebugLevel", 0, "Debug Message Level");
    pl->set("NumThreads", 1, "Number of threads");
    pl->set("MatchingOption", 0, "Matching option (0 none, 1 cardinarity)");
    pl->set("OrderingOption", 2, "Reordering option (0 minumum degree, 2 nested dissection, 3 parallel OpenMP ND)");

    setStringToIntegralParameter<int>("Trans", "NOTRANS",
                                      "Solve for the transpose system or not",
                                      tuple<string>("NOTRANS","TRANS","CONJ"),
                                      tuple<string>("Solve with transpose",
                                                    "Do not solve with transpose",
                                                    "Solve with the conjugate transpose"),
                                      tuple<int>(0, 1, 2),
                                      pl.getRawPtr());
    valid_params = pl;
  }

  return valid_params;
}


template <class Matrix, class Vector>
bool
D3S<Matrix,Vector>::loadA_impl(EPhase current_phase)
{
  using Teuchos::as;
#ifdef HAVE_AMESOS2_TIMERS
  Teuchos::TimeMonitor convTimer(this->timers_.mtxConvTime_);
#endif

  // D3S does not need matrix data in the pre-ordering phase
  if( current_phase == PREORDERING ) return( false );

  // is_contiguous         : input is contiguous
  // CONTIGUOUS_AND_ROOTED : input is not contiguous, so make output contiguous
  // TODO:
  EDistribution dist_option = (true ? DISTRIBUTED_NO_OVERLAP : ((is_contiguous_ == true) ? ROOTED : CONTIGUOUS_AND_ROOTED));
  if (dist_option == DISTRIBUTED_NO_OVERLAP && !is_contiguous_) {
    // Neeed to form contiguous matrix
    // Only reinex GIDs
    d3s_rowmap_ = this->matrixA_->getRowMap(); // use original map to redistribute vectors in solve
    Teuchos::RCP<const MatrixAdapter<Matrix> > contig_mat = this->matrixA_->reindex(d3s_contig_rowmap_, d3s_contig_colmap_, current_phase);
    // Copy into local views
    if (current_phase == SYMBFACT) { 
        Kokkos::resize(nzvals_temp_, contig_mat->getLocalNNZ());
        Kokkos::resize(nzvals_view_, contig_mat->getLocalNNZ());
        Kokkos::resize(colind_view_, contig_mat->getLocalNNZ());
        Kokkos::resize(rowptr_view_, contig_mat->getLocalNumRows() + 1);
    }
    int nnz_ret = 0;
    {
#ifdef HAVE_AMESOS2_TIMERS
      Teuchos::TimeMonitor mtxRedistTimer( this->timers_.mtxRedistTime_ );
#endif
      Util::get_crs_helper_kokkos_view<MatrixAdapter<Matrix>,
        host_value_type_array,host_ordinal_type_array, host_size_type_array >::do_get(
                                         contig_mat.ptr(),
                                         nzvals_temp_, colind_view_, rowptr_view_,
                                         nnz_ret,
                                         ptrInArg(*(contig_mat->getRowMap())),
                                         #if 1
                                         DISTRIBUTED_NO_OVERLAP,
                                         #else
                                         ROOTED,
                                         #endif
                                         SORTED_INDICES);
      Kokkos::deep_copy(nzvals_view_, nzvals_temp_);
    }
  } else {
    if (current_phase == SYMBFACT) {
      if (dist_option == DISTRIBUTED_NO_OVERLAP) {
        Kokkos::resize(nzvals_temp_, this->matrixA_->getLocalNNZ());
        Kokkos::resize(nzvals_view_, this->matrixA_->getLocalNNZ());
        Kokkos::resize(colind_view_, this->matrixA_->getLocalNNZ());
        Kokkos::resize(rowptr_view_, this->matrixA_->getLocalNumRows() + 1);
      } else {
        if( this->root_ ) {
          Kokkos::resize(nzvals_temp_, this->matrixA_->getGlobalNNZ());
          Kokkos::resize(nzvals_view_, this->matrixA_->getGlobalNNZ());
          Kokkos::resize(colind_view_, this->matrixA_->getGlobalNNZ());
          Kokkos::resize(rowptr_view_, this->matrixA_->getGlobalNumRows() + 1);
        } else {
          Kokkos::resize(nzvals_temp_, 0);
          Kokkos::resize(nzvals_view_, 0);
          Kokkos::resize(colind_view_, 0);
          Kokkos::resize(rowptr_view_, 0);
        }
      }
    }
    int nnz_ret = 0;
    {
#ifdef HAVE_AMESOS2_TIMERS
      Teuchos::TimeMonitor mtxRedistTimer( this->timers_.mtxRedistTime_ );
#endif
      Util::get_crs_helper_kokkos_view<MatrixAdapter<Matrix>,
        host_value_type_array,host_ordinal_type_array, host_size_type_array >::do_get(
                                         this->matrixA_.ptr(),
                                         nzvals_temp_, colind_view_, rowptr_view_,
                                         nnz_ret,
                                         ptrInArg(*(this->matrixA_->getRowMap())),
                                         dist_option,
                                         SORTED_INDICES);
      Kokkos::deep_copy(nzvals_view_, nzvals_temp_);
    }
  }
  return true;
}


template <class Matrix, class Vector>
void
D3S<Matrix,Vector>::describe_impl(Teuchos::FancyOStream &out,
                                  const Teuchos::EVerbosityLevel verbLevel) const
{
  out << " D3S current parameters:" << std::endl;
  out << std::endl;
}

template<class Matrix, class Vector>
const char* D3S<Matrix,Vector>::name = "D3S";


} // end namespace Amesos2

#endif  // AMESOS2_D3S_DEF_HPP
