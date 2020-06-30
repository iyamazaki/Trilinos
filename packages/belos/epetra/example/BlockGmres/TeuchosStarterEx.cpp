#include "Teuchos_BLAS.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "BelosMultiVecTraits.hpp"



typedef double ScalarType;
typedef Teuchos::ScalarTraits<ScalarType> SCT;
typedef typename SCT::magnitudeType MagnitudeType;

int main(){

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   Teuchos::BLAS<int,ScalarType> blas;

   int i, j, k, lda, ldq, ldv, ldr, ldt;
   int m = 250, n = 40;
   double norma, norma2;
   MagnitudeType orth, repres, nrmA;

   lda = m, ldq = m, ldv = m, ldr = n, ldt = n;


   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > A; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > Q; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > V; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > T; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 

   if (A == Teuchos::null) {
     A = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n) );
   }  
   if (Q == Teuchos::null) {
     Q = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n) );
   }
   if (V == Teuchos::null) {
     V = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n) );
   }
   if (T == Teuchos::null) {
     T = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,1) );
   }

   A->random();
   for( i=0; i<n; i++){
      blas.COPY( m, &(*A)(0,i), 1, &(*V)(0,i), 1);
   }
   nrmA = A->normFrobenius();


//   Teuchos::RCP<Belos::MultiVecTraits<int,ScalarType>> AA; 
//   if (AA == Teuchos::null) {
//      AA = Teuchos::rcp( new Belos::MultiVecTraits<int,ScalarType>() );
//   }
//   AA = Teuchos::rcp<Belos::MultiVecTraits<int,ScalarType>>(A,n);
//   AA->Print(AA,std::cout);

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;

   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n,true) );
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );



   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Step 0 -- initialization step

   norma2 = blas.DOT( m-1, &(*V)(1,0), 1, &(*V)(1,0), 1);
   norma = sqrt( (*V)(0,0) * (*V)(0,0) + norma2 );

   (*T)(0,0) = ( (*V)(0,0) > 0 ) ? ( (*V)(0,0) + norma ) : ( (*V)(0,0) - norma ); // Using T(0,0) as a workspace
   blas.SCAL( m-1, ( 1.0e+00 / (*T)(0,0) ), &(*V)(1,0), 1 );
   (*T)(0,0) = ( 2.0e+00 ) / ( (1.0e+00) + norma2 / ( (*T)(0,0) * (*T)(0,0) ) ); 

   (*V)(0,0) = ( (*V)(0,0) > 0 ) ? ( - norma ) : ( + norma );

   (*Q)(0,0) = 1.0e+00 - (*T)(0,0);
   for( i=1; i<m; i++ ){ (*Q)(i,0) = - ( (*V)(i,0) ) * ( (*T)(0,0) ); }
 
   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // QR factorization of the matrix A --> A is copied in V for storage

   for( j=1; j<n; j++){

      // Step 1: Project and Update
      blas.COPY( j, &(*V)(0,j), 1, &(*work)(0,0), 1);
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::TRANS, Teuchos::UNIT_DIAG, j, &(*V)(0,0), ldv, &(*work)(0,0), 1 );
      blas.GEMV( Teuchos::TRANS, m-j, j, 1.0e+00, &(*V)(j,0), ldv, &(*V)(j,j), 1, 1.0e+00, &(*work)(0,0), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 );
      blas.GEMV( Teuchos::NO_TRANS, m-j, j, -1.0e+00, &(*V)(j,0), ldv, &(*work)(0,0), 1, 1.0e+00, &(*V)(j,j), 1 );
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::UNIT_DIAG, j, &(*V)(0,0), ldv, &(*work)(0,0), 1 );

      // Step 2: Update V, tau and R
      for( i=0; i<j; i++ ){ (*V)(i,j) = (*V)(i,j) - (*work)(i,0); }
      norma2 =  blas.DOT( m-(j+1), &(*V)(j+1,j), 1, &(*V)(j+1,j), 1);
      norma = sqrt( (*V)(j,j) * (*V)(j,j) + norma2 );
      (*work)(0,0) = ( (*V)(j,j) > 0 ) ? ( (*V)(j,j) + norma ) : ( (*V)(j,j) - norma );
      (*T)(j,j) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*work)(0,0) * (*work)(0,0) ) );
      blas.SCAL( m-(j+1), ( 1.0e+00 / (*work)(0,0) ), &(*V)(j+1,j), 1 );
      (*V)(j,j) = ( (*V)(j,j) > 0 ) ? ( - norma ) : ( + norma );
 
      // Step 3: Construct Q
      (*Q)(j,j) = 1.0e+00 - (*T)(j,j);
      for( i=j+1; i<m; i++ ){ (*Q)(i,j) = - ( (*V)(i,j) ) * ( (*T)(j,j) ); }
      for( i=0; i<j; i++ ){ (*Q)(i,j) = 0.0e+00; }

      blas.GEMV( Teuchos::TRANS, m-j, j, 1.0e+00, &(*V)(j,0), ldv, &(*Q)(j,j), 1, 0.0e+00, &(*T)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );
      blas.GEMV( Teuchos::NO_TRANS, m-j, j, -1.0e+00, &(*V)(j,0), ldv, &(*T)(0,j), 1, 1.0e+00, &(*Q)(j,j), 1 );
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::UNIT_DIAG, j, &(*V)(0,0), ldv, &(*T)(0,j), 1 );
      blas.AXPY( j, -1.0e+00, &(*T)(0,j), 1, &(*Q)(0,j), 1 );

      // Step 4: Construct T
      blas.COPY( j, &(*V)(j,0), ldv, &(*T)(0,j), 1);
      blas.GEMV( Teuchos::TRANS, m-(j+1), j, 1.0e+00, &(*V)(j+1,0), ldv, &(*V)(j+1,j), 1, 1.0e+00, &(*T)(0,j), 1 );
      blas.SCAL( j, -(*T)(j,j), &(*T)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );

   }

   // Orthogonality Check
   orth_check->putScalar();
   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, n, m, 1.0e+00, &(*Q)(0,0), ldq, 0.0e+00, &(*orth_check)(0,0), n);
   for( i=0; i<n; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   orth = orth_check->normFrobenius();

   // Representativity check
   for( i = 0; i < n; i++ ){ blas.COPY( m, &(*Q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, n, 1.0e+00, &(*V)(0,0), ldv, &(*repres_check)(0,0), m );
   for( k=0; k<n; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
   repres = repres_check->normFrobenius();

   printf("m = %3d, n = %3d,  ",m,n);
   printf("|| I - Q'Q || = %3.3e, ", orth);
   printf("|| A - QR || / ||A|| = %3.3e \n", repres/nrmA); 

}




