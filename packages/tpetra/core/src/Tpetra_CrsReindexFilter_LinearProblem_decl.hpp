// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_CRSREINDEXFILTER_LINEARPROBLEM_DECL_HPP
#define TPETRA_CRSREINDEXFILTER_LINEARPROBLEM_DECL_HPP

/// \file Tpetra_CrsReindexFilter_LinearProblem_decl.hpp
/// \brief Declaration of the Tpetra::Transform class

//#include "Epetra_ConfigDefs.h"
//#include "Epetra_Object.h"
//#include "Epetra_CrsMatrix.h"
//#include "Epetra_MapColoring.h"
//#include "Epetra_SerialDenseVector.h"

//#include "Teuchos_RCP.hpp"
//#include "Teuchos_ArrayRCP.hpp"
//#include "Tpetra_RowMatrix_fwd.hpp"
#include "Tpetra_LinearProblem.hpp"
#include "Tpetra_Transform.hpp"


#include "Teuchos_DataAccess.hpp"

#include "Tpetra_Core.hpp"                      //  REMOVE TEMP!!
#include "Tpetra_Vector_decl.hpp"
#include "Tpetra_MultiVector_decl.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"
#include "Tpetra_RowMatrix_decl.hpp"


//class Epetra_LinearProblem;
//class Epetra_Map;
//class Epetra_MultiVector;
//class Epetra_Import;
//class Epetra_Export;
//class Epetra_IntVector;


namespace Tpetra {

  /// \class CrsReindexFilter_LinearProblem
  /// \brief A class for explicitly eliminating matrix rows and columns from a LinearProblem.
  ///
  /// The Epetra_CrsSingletonFilter class takes an existing Epetra_LinearProblem
  /// object, analyzes it structure and explicitly eliminates singleton
  /// rows and columns from the matrix and appropriately modifies the RHS
  /// and LHS of the linear problem.  The result of this process is a
  /// reindexed system of equations that is itself an Epetra_LinearProblem
  /// object.  The reindexed system can then be solved using any solver
  /// that is understands an Epetra_LinearProblem.  The solution for the
  /// original system is obtained by calling ComputeOriginalSolution().
  /// 
  /// Singleton rows are defined to be rows that have a single nonzero
  /// entry in the matrix.  The equation associated with this row can be
  /// explicitly eliminated because it involved only one variable.  For
  /// example if row i has a single nonzero value in column j, call it
  /// A(i,j), we can explicitly solve for x(j) = b(i)/A(i,j), where b(i)
  /// is the ith entry of the RHS and x(j) is the jth entry of the LHS.
  /// 
  /// Singleton columns are defined to be columns that have a single
  /// nonzero entry in the matrix.  The variable associated with this
  /// column is originaly dependent, meaning that the solution for all other
  /// variables does not depend on it.  If this entry is A(i,j) then the
  /// ith row and jth column can be removed from the system and x(j) can
  /// be solved after the solution for all other variables is determined.
  /// 
  /// By removing singleton rows and columns, we can often produce a
  /// reindexed system that is smaller and far less dense, and in general
  /// having better numerical properties.
  /// 
  /// The basic procedure for using this class is as follows:
  /// <ol>
  /// <li> Construct original problem: Construct and Epetra_LinearProblem
  ///      containing the "original" matrix, RHS and LHS.  This is done 
  ///      outside of Epetra_CrsSingletonFilter class.  Presumably,
  ///      you have some reason to believe that this system may contain
  ///      singletons.
  /// <li> Construct an Epetra_CrsSingletonFilter instance:  Constructor
  ///      needs no arguments.
  /// <li> Analyze matrix: Invoke the Analyze() method, passing in the
  ///      Tpetra::RowMatrix object from your original linear problem
  ///      mentioned in the first step above.
  /// <li> Go/No Go decision to construct reindexed problem:
  ///      Query the results of the Analyze method using the SingletonsDetected()
  ///      method.  This method returns "true" if there were singletons
  ///      found in the matrix.  You can also query any of the other
  ///      methods in the Filter Statistics section to determine if you
  ///      want to proceed with the construction of the reindexed system.
  /// <li> Construct reindexed problem: 
  ///      If, in the previous step, you determine that you want to proceed
  ///      with the construction of the reindexed problem, you should next
  ///      call the ConstructReindexedProblem() method, passing in the original
  ///      linear problem object from the first step.  This method will
  ///      use the information from the Analyze() method to construct a
  ///      reduce problem that has explicitly eliminated the singleton
  ///      rows, solved for the corresponding LHS values and updated the
  ///      RHS.  This step will also remove singleton columns from the
  ///      reindexed system.  Once the solution of the reindexed problem is
  ///      is computed (via any solver that understands an Epetra_LinearProblem),
  ///      you should call the ComputeOriginalSolution() method to compute
  ///      the LHS values assocaited with the singleton columns.
  /// <li> Solve reindexed problem: Obtain a RCP to the reindexed problem
  ///      using the ReindexedProblem() method.  Using the solver of your
  ///      choice, solve the reindexed system.
  /// <li> Compute solution to original problem:  Once the solution the reindexed
  ///      problem is determined, the ComputeOriginalSolution() method will
  ///      place the reindexed solution values into the appropriate locations
  ///      of the original solution LHS and then compute the values associated
  ///      with column singletons.  At this point, you have a complete
  ///      solution to the original original problem.
  /// <li> Solve a subsequent original problem that differs from the original
  ///      problem only in values: It is often the case that the structure
  ///      of a problem will be the same for a sequence of linear problems.
  ///      In this case, the UpdateReindexedProblem() method can be useful.
  ///      After going through the above process one time, if you have a
  ///      linear problem that is structural \e identical to the previous
  ///      problem, you can minimize memory and time costs by using the
  ///      UpdateReindexedProblem() method, passing in the subsequent
  ///      problem.  Once you have called the UpdateReindexedProblem()
  ///      method, you can then solve the reduce problem problem as you
  ///      wish, and then compute the original solution as before.  The RCP
  ///      generated by ReindexedProblem() will not change when
  ///      UpdateReindexedProblem() is called.
  /// </ol>

  template <class Scalar,
            class LocalOrdinal,
            class GlobalOrdinal,
            class Node>
  class CrsReindexFilter_LinearProblem :
    public SameTypeTransform<Tpetra::LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  {
        
   public:
    //! @name Typedefs
    //@{

    using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using crs_matrix_type = CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using row_matrix_type = RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using multivector_type = MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using vector_type = Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using vector_type_int = Vector<int, LocalOrdinal, GlobalOrdinal, Node>;
    using vector_type_LO = Vector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>;
    using linear_problem_type = LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    using scalar_type = Tpetra::Vector<>::scalar_type;
    using local_ordinal_type = Tpetra::Vector<>::local_ordinal_type;
    using global_ordinal_type = Tpetra::Vector<>::global_ordinal_type;

    using OriginalType      = typename Transform<linear_problem_type, linear_problem_type>::OriginalType;
    using OriginalConstType = typename Transform<linear_problem_type, linear_problem_type>::OriginalConstType;
    using NewType           = typename Transform<linear_problem_type, linear_problem_type>::NewType;
    using NewConstType      = typename Transform<linear_problem_type, linear_problem_type>::NewConstType;

    using nonconst_local_inds_host_view_type = typename row_matrix_type::nonconst_local_inds_host_view_type;
    using nonconst_values_host_view_type     = typename row_matrix_type::nonconst_values_host_view_type;
    //@}
 
    //@{ \name Constructors/Destructor.
    /// \brief Constructor.
    CrsReindexFilter_LinearProblem( bool verbose = false );
 
    /// \brief Destructor
    virtual ~CrsReindexFilter_LinearProblem();
    //@}
  
    ///
    NewType operator()( const OriginalType & originalLinearProblem );
  
    ///
    void fwd();
  
    ///
    void rvs();


    //@{ \name Attribute Access Methods.

    /// \brief Returns RCP to the original original Tpetra::LinearProblem.
    Teuchos::RCP<linear_problem_type> OriginalProblem() const {return(OriginalProblem_);}

    /// \brief Returns RCP to the derived reindexed Tpetra::LinearProblem.
    Teuchos::RCP<linear_problem_type> ReindexedProblem() const {return(ReindexedProblem_);}

    /// \brief! Returns RCP to Tpetra::RowMatrix from original problem.
    Teuchos::RCP<row_matrix_type> OriginalMatrix() const {return(OriginalRowMatrix_);}

    /// \brief Returns RCP to Tpetra::CrsMatrix from reindexed problem.
    Teuchos::RCP<row_matrix_type> ReindexedMatrix() const {return(ReindexedProblem_->getMatrix());}

    //! Returns RCP to Tpetra::MapColoring object: color 0 rows are part of reindexed system.
    //Epetra_MapColoring * RowMapColors() const {return(RowMapColors_);}

    //! Returns RCP to Tpetra::MapColoring object: color 0 columns are part of reindexed system.
    //Epetra_MapColoring * ColMapColors() const {return(ColMapColors_);}

    //! Returns RCP to Tpetra::Map describing the reindexed system row distribution.
    Teuchos::RCP<const map_type> ReindexedMatrixRowMap() const {return(ReindexedProblem_->getMatrix()->getRowMap());}

    //! Returns RCP to Tpetra::Map describing the reindexed system column distribution.
    Teuchos::RCP<const map_type> ReindexedMatrixColMap() const {return(ReindexedProblem_->getMatrix()->getColMap());}

    //! Returns RCP to Tpetra::Map describing the domain map for the reindexed system.
    Teuchos::RCP<const map_type> ReindexedMatrixDomainMap() const {return(ReindexedProblem_->getMatrix()->getDomainMap());}

    //! Returns RCP to Tpetra::Map describing the range map for the reindexed system.
    Teuchos::RCP<const map_type> ReindexedMatrixRangeMap() const {return(ReindexedProblem_->getMatrix()->getRangeMap());}

    //@}
    //
   protected:

    //  This RCP will be null if original matrix is not a CrsMatrix.
    Teuchos::RCP<crs_matrix_type> OriginalCrsMatrix() const {return(OriginalCrsMatrix_);}

    Teuchos::RCP<const map_type> OriginalMatrixRowMap() const {return(OriginalMatrix()->getRowMap());}
    Teuchos::RCP<const map_type> OriginalMatrixColMap() const {return(OriginalMatrix()->getColMap());}
    Teuchos::RCP<const map_type> OriginalMatrixDomainMap() const {return(OriginalMatrix()->getDomainMap());}
    Teuchos::RCP<const map_type> OriginalMatrixRangeMap() const {return(OriginalMatrix()->getRangeMap());}

   protected:
    bool verbose_;

    Teuchos::RCP<linear_problem_type> OriginalProblem_;
    Teuchos::RCP<row_matrix_type> OriginalRowMatrix_;
    Teuchos::RCP<crs_matrix_type> OriginalCrsMatrix_;
    Teuchos::RCP<multivector_type> OriginalRHS_;
    Teuchos::RCP<multivector_type> OriginalLHS_;

    Teuchos::RCP<linear_problem_type> ReindexedProblem_;
    Teuchos::RCP<crs_matrix_type> ReindexedCrsMatrix_;
    Teuchos::RCP<multivector_type> ReindexedRHS_;
    Teuchos::RCP<multivector_type> ReindexedLHS_;

   private:
  };

} //namespace Tpetra

#endif //  TPETRA_CRSREINDEXFILTER_LINEARPROBLEM_DECL_HPP
