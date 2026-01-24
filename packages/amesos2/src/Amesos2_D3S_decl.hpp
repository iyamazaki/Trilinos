// @HEADER
// *****************************************************************************
//           Amesos2: Templated Direct Sparse Solver Package
//
// Copyright 2011 NTESS and the Amesos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/**
  \file   Amesos2_D3S_decl.hpp
  \author Siva Rajamanickam <srajama@sandia.gov>

  \brief  Amesos2 D3S declarations.
*/


#ifndef AMESOS2_D3S_DECL_HPP
#define AMESOS2_D3S_DECL_HPP

#include "Amesos2_SolverTraits.hpp"
#include "Amesos2_SolverCore.hpp"
#include "Amesos2_D3S_FunctionMap.hpp"

namespace Amesos2 {


/** \brief Amesos2 interface to the D3S package.
 *
 * See the \ref D3S_parameters "summary of D3S parameters"
 * supported by this Amesos2 interface.
 *
 * \ingroup amesos2_solver_interfaces
 */
template <class Matrix,
          class Vector>
class D3S : public SolverCore<Amesos2::D3S, Matrix, Vector>
{
  friend class SolverCore<Amesos2::D3S,Matrix,Vector>; // Give our base access
                                                          // to our private
                                                          // implementation funcs
public:

  /// Name of this solver interface.
  static const char* name;      // declaration. Initialization outside.

  typedef D3S<Matrix,Vector>                                       type;
  typedef SolverCore<Amesos2::D3S,Matrix,Vector>             super_type;

  // Since typedef's are not inheritted, go grab them
  typedef typename VectorTraits<Vector>::scalar_t        vector_scalar_type;
  typedef typename super_type::scalar_type                      scalar_type;
  typedef typename super_type::local_ordinal_type        local_ordinal_type;
  typedef typename super_type::global_ordinal_type      global_ordinal_type;
  typedef typename super_type::global_size_type            global_size_type;
  typedef typename super_type::node_type                          node_type;
  typedef Tpetra::Map<local_ordinal_type,
                                 global_ordinal_type,
                                 node_type>                        map_type;

  typedef TypeMap<Amesos2::D3S,scalar_type>                        type_map;

  /*
   * The D3S interface will need two other typedef's, which are:
   * - the D3S type that corresponds to scalar_type and
   * - the corresponding type to use for magnitude
   */
  typedef typename type_map::type                                 d3s_type;
  typedef typename type_map::dtype                               d3s_dtype;

  typedef FunctionMap<Amesos2::D3S,d3s_dtype>                  function_map;

  typedef Matrix                                                matrix_type;
  typedef MatrixAdapter<matrix_type>                    matrix_adapter_type;

  /// \name Constructor/Destructor methods
  //@{

  /**
   * \brief Initialize from Teuchos::RCP.
   *
   * \warning Should not be called directly!  Use instead
   * Amesos2::create() to initialize a D3S interface.
   */
  D3S(Teuchos::RCP<const Matrix> A,
          Teuchos::RCP<Vector>       X,
          Teuchos::RCP<const Vector> B);


  /// Destructor
  ~D3S( );

  //@}

private:

 /**
  * \brief can we optimize size_type and ordinal_type for straight pass through,
  * also check that is_contiguous_ flag set to true
  */
  bool single_proc_optimization() const;

  /**
   * \brief Performs pre-ordering on the matrix to increase efficiency.
   *
   * D3S does not support pre-ordering, so this method does nothing.
   */
  int preOrdering_impl();


  /**
   * \brief Perform symbolic factorization of the matrix using D3S.
   *
   * Called first in the sequence before numericFactorization.
   *
   * \throw std::runtime_error D3S is not able to factor the matrix.
   */
  int symbolicFactorization_impl();


  /**
   * \brief D3S specific numeric factorization
   *
   * \throw std::runtime_error D3S is not able to factor the matrix
   */
  int numericFactorization_impl();


  /**
   * \brief D3S specific solve.
   *
   * Uses the symbolic and numeric factorizations, along with the RHS
   * vector \c B to solve the sparse system of equations.  The
   * solution is placed in X.
   *
   * \throw std::runtime_error D3S is not able to solve the system.
   *
   * \callgraph
   */
  int solve_impl(const Teuchos::Ptr<MultiVecAdapter<Vector> >       X,
                 const Teuchos::Ptr<const MultiVecAdapter<Vector> > B) const;


  /**
   * \brief Determines whether the shape of the matrix is OK for this solver.
   */
  bool matrixShapeOK_impl() const;


  /**
   * Currently, the following D3S parameters/options are
   * recognized and acted upon:
   *
   * <ul>
   *   <li> \c "Trans" : { \c "NOTRANS" | \c "TRANS" |
   *     \c "CONJ" }.  Specifies whether to solve with the transpose system.</li>
   *   <li> \c "Equil" : { \c true | \c false }.  Specifies whether
   *     the solver to equilibrate the matrix before solving.</li>
   *   <li> \c "IterRefine" : { \c "NO" | \c "SINGLE" | \c "DOUBLE" | \c "EXTRA"
   *     }. Specifies whether to perform iterative refinement, and in
   *     what precision to compute the residual.</li>
   *   <li> \c "SymmetricMode" : { \c true | \c false }.</li>
   *   <li> \c "DiagPivotThresh" : \c double value. Specifies the threshold
   *     used for a diagonal to be considered an acceptable pivot.</li>
   *   <li> \c "ColPerm" which takes one of the following:
   *     <ul>
   *     <li> \c "NATURAL" : natural ordering.</li>
   *     <li> \c "MMD_AT_PLUS_A" : minimum degree ordering on the structure of
   *       \f$ A^T + A\f$ .</li>
   *     <li> \c "MMD_ATA" : minimum degree ordering on the structure of
   *       \f$ A T A \f$ .</li>
   *     <li> \c "COLAMD" : approximate minimum degree column ordering.
   *       (default)</li>
   *     </ul>
   * </ul>
   */
  void setParameters_impl(
    const Teuchos::RCP<Teuchos::ParameterList> & parameterList );


  /**
   * Hooked in by Amesos2::SolverCore parent class.
   *
   * \return a const Teuchos::ParameterList of all valid parameters for this
   * solver.
   */
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters_impl() const;


  /**
   * \brief Reads matrix data into internal structures
   *
   * \param [in] current_phase an indication of which solution phase this
   *                           load is being performed for.
   *
   * \return \c true if the matrix was loaded, \c false if not
   */
  bool loadA_impl(EPhase current_phase);

  typedef Kokkos::DefaultHostExecutionSpace HostExecSpaceType;
  typedef Kokkos::View<int*,       HostExecSpaceType>    host_size_type_array;
  typedef Kokkos::View<int*,       HostExecSpaceType>    host_ordinal_type_array;
  typedef Kokkos::View<d3s_dtype*, HostExecSpaceType>    host_value_type_array;

private:

  /// Stores the values of the nonzero entries for D3S
  host_value_type_array nzvals_view_;
  host_value_type_array nzvals_temp_;
  /// Stores the location in \c Ai_ and Aval_ that starts row j
  host_ordinal_type_array colind_view_;
  /// Stores the row indices of the nonzero entries
  host_size_type_array rowptr_view_;

  mutable int nrhs_;
  /// Persisting, contiguous, 1D store for X
  mutable Teuchos::Array<d3s_dtype> xvals_;
  mutable Teuchos::Array<d3s_dtype> tvals_;
  /// Persisting, contiguous, 1D store for B
  mutable Teuchos::Array<d3s_dtype> bvals_;

  /// Transpose flag
  /// 0: Non-transpose, 1: Transpose, 2: Conjugate-transpose
  int transFlag_;

  bool is_contiguous_;
  bool use_gather_;

  std::string solvername_;
  int msg_level_;
  int num_threads_;
  int matching_option_;
  int reorder_option_;
  int debug_level_;

  int numProcSolver_;
  int numRows_;
  int startGID_;
  MPI_Fint D3SComm_;
  Teuchos::RCP<const map_type> d3s_rowmap_;
  Teuchos::RCP<const map_type> d3s_contig_rowmap_;
  Teuchos::RCP<const map_type> d3s_contig_colmap_;

  Teuchos::RCP<D3Solver> solver;
};                              // End class D3S


// Specialize solver_traits struct for D3S
template <>
struct solver_traits<D3S> {
#ifdef HAVE_TEUCHOS_COMPLEX
  typedef Meta::make_list6<float,
                           double,
                           Kokkos::complex<float>,
                           Kokkos::complex<double>,
                           std::complex<float>,
                           std::complex<double> > supported_scalars;
#else
  typedef Meta::make_list2<float, double> supported_scalars;
#endif
};

template <typename Scalar, typename LocalOrdinal, typename ExecutionSpace>
struct solver_supports_matrix<D3S,
  KokkosSparse::CrsMatrix<Scalar, LocalOrdinal, ExecutionSpace>> {
  static const bool value = true;
};

} // end namespace Amesos2

#endif  // AMESOS2_D3S_DECL_HPP
