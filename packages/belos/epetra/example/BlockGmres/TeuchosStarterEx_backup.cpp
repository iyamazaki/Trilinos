#include "Teuchos_BLAS.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"

typedef double ScalarType;
typedef Teuchos::ScalarTraits<ScalarType> SCT;
typedef typename SCT::magnitudeType MagnitudeType;

int main(){

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   Teuchos::BLAS<int,ScalarType> blas;

   int i, j, k, lda, ldq, ldv, ldr, ldt;
   int m = 150, n = 150;
   lda = m, ldq = m, ldv = m, ldr = n, ldt = n;

   double norma, norma2, orth_val, repres_val;
   MagnitudeType orth, repres;

   // Create an empty matrix, then hard set it to dimensions ( m x n )
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > A; 
   if (A == Teuchos::null) {
     A = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n) );
   } //rcp creates a RCP from (new stuff) 

   // Generate random matrix and print
   A->random();
   //A->print(std::cout);

   // Initialize the variable R --> Does this allocate space for it?
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > R; 
   // Create square-matrix for QR factorization
   R = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );  
   //R->print(std::cout);

   // Create a for loop -- putting upper-triangular entries of A into R
   //for( i = 0; i < n; i++ ){
   //   blas.COPY( n-i, &(*A)(i,i), lda, &(*R)(i,i), ldr);
   //}
   //R->print(std::cout);
  
   // Setting matrices below to random so I know if I am doing something wrong
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > Q; 
   if (Q == Teuchos::null) {
     Q = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n) );
   }
//   Q->random();
   Q = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n,true) );  

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > V; 
   if (V == Teuchos::null) {
     V = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n) );
   }
   // Copy A into V -- Keeping A for representativity check at end
   // Copying column-wise -- is this better for memory heirarchy?
   for( i=0; i<n; i++){
      blas.COPY( m, &(*A)(0,i), 1, &(*V)(0,i), 1);
   }

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > T; 
   if (T == Teuchos::null) {
     T = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }
//   T->random();
   T = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );  

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;
   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n,true) );

//   for(i=0;i<n;i++){ for(j=0;j<m;j++){ (*repres_check)(j,i) = (*A)(j,i) - (*V)(j,i); }}
//   repres_val = repres_check->normFrobenius();
//   printf("\n\n %3.3e \n\n",repres_val);

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Now I have the main matrices. Time to do a QR factorization
   // Note: I am not coding for parrallel processing, I will do 
   // this later after I start to become more comfortable. I'm 
   // sure there will be better ways to do most things I do here
   //
   //
   // Step 1 -- initialization step

//   j = 0;
   
   blas.COPY( 1, &(*A)(0,0), 1, &(*R)(0,0), 1); 
   norma2 = blas.DOT( m-1, &(*V)(0+1,0), 1, &(*V)(0+1,0), 1);
   norma = sqrt( (*R)(0,0) * (*R)(0,0) + norma2 );

   (*R)(0,0) = ( (*R)(0,0) > 0 ) ? ( (*R)(0,0) + norma ) : ( (*R)(0,0) - norma ) ;
   (*T)(0,0) = ( 2.0e+00 ) / ( (1.0e+00) + norma2 / ( (*R)(0,0) * (*R)(0,0) ) ); 
   (*V)(0,0) = 1.0e+00; 
   blas.SCAL( m-1, ( 1.0e+00 / (*R)(0,0) ), &(*V)(0+1,0), 1 );
   (*R)(0,0) = ( (*R)(0,0) > 0 ) ? ( - norma ) : ( + norma );

   (*Q)(0,0) = 1.0e+00 - (*T)(0,0);
   for( i = 1 ; i < m ; i++ ){ (*Q)(i,0) = - ( (*V)(i,0) ) * ( (*T)(0,0) ); }
 
   // Orthogonality Check
   orth_check->putScalar();
   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, 1, m, 1.0e+00, &(*Q)(0,0), ldq, 0.0e+00, &(*orth_check)(0,0), n);
   for( i=0; i<1; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   orth_val = orth_check->normFrobenius();
   printf("\n|| I - Q''Q || = %3.3e, ", orth_val);

   // Representativity check
   repres_check->putScalar();
   for( i = 0; i < 1; i++ ){ blas.COPY( m, &(*Q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, 1, 1.0e+00, &(*R)(0,0), ldr, &(*repres_check)(0,0), m );
   for( k=0; k<1; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
   repres_val = repres_check->normFrobenius();
   printf("|| A - QR || = %3.3e \n", repres_val);



   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////
   //
   // Step 2 - next column
   //

//   j = 1;

   blas.GEMV( Teuchos::TRANS, m, 1, 1.0e+00, &(*V)(0,0), ldv, &(*V)(0,1), 1, 0.0e+00, &(*R)(0,1), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, 1, &(*T)(0,0), ldt, &(*R)(0,1), 1 );
   blas.GEMV( Teuchos::NO_TRANS, m, 1, -1.0e+00, &(*V)(0,0), ldv, &(*R)(0,1), 1, 1.0e+00, &(*V)(0,1), 1 );
   for( i=0; i<1; i++ ){ (*R)(i,1) = (*V)(i,1); }

   (*R)(1,1) = (*V)(1,1);
   norma2 =  blas.DOT( m-(1+1), &(*V)(1+1,1), 1, &(*V)(1+1,1), 1);
   norma = sqrt( (*R)(1,1) * (*R)(1,1) + norma2 );
   (*R)(1,1) = ( (*R)(1,1) > 0 ) ? ( (*R)(1,1) + norma ) : ( (*R)(1,1) - norma );
   (*T)(1,1) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*R)(1,1) * (*R)(1,1) ) );
   for( i=0; i<1; i++ ){ (*V)(i,1) = +0.0e+00; }
   (*V)(1,1) = 1.0e+00;
   blas.SCAL( m-(1+1), ( 1.0e+00 / (*R)(1,1) ), &(*V)(1+1,1), 1 );
   (*R)(1,1) = ( (*R)(1,1) > 0 ) ? ( - norma ) : ( + norma );

   (*Q)(1,1) = 1.0e+00 - (*T)(1,1);
   for( i = 1+1; i < m ; i++ ){ (*Q)(i,1) = - ( (*V)(i,1) ) * ( (*T)(1,1) ); }
   for( i = 0; i < 1 ; i++ ){ (*Q)(i,1) = 0.0e+00; }

   blas.GEMV( Teuchos::TRANS, m, 1, 1.0e+00, &(*V)(0,0), ldv, &(*Q)(0,1), 1, 0.0e+00, &(*T)(0,1), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, 1, &(*T)(0,0), ldt, &(*T)(0,1), 1 );
   blas.GEMV( Teuchos::NO_TRANS, m, 1, -1.0e+00, &(*V)(0,0), ldv, &(*T)(0,1), 1, 1.0e+00, &(*Q)(0,1), 1 );

//   for( i=0; i<1; i++){ (*T)(i,1) = (*V)(1,i); }
   blas.GEMV( Teuchos::TRANS, m, 1, 1.0e+00, &(*V)(0,0), ldv, &(*V)(0,1), 1, 0.0e+00, &(*T)(0,1), 1 );
   blas.SCAL( 1, -(*T)(1,1), &(*T)(0,1), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, 1, &(*T)(0,0), ldt, &(*T)(0,1), 1 );
 
   // Orthogonality Check
   orth_check->putScalar();
   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, n, m, 1.0e+00, &(*Q)(0,0), ldq, 0.0e+00, &(*orth_check)(0,0), n);
   for( i=0; i<2; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   orth_val = orth_check->normFrobenius();
   printf("\n|| I - Q''Q || = %3.3e, ", orth_val);

   // Representativity check
   repres_check->putScalar();
   for( i = 0; i < 2; i++ ){ blas.COPY( m, &(*Q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, 2, 1.0e+00, &(*R)(0,0), ldr, &(*repres_check)(0,0), m );
   for( k=0; k<2; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
   repres_val = repres_check->normFrobenius();
   printf("|| A - QR || = %3.3e \n", repres_val);

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////
   //
   // Step 3 - next column
   //

//   j = 2;
/*
   blas.GEMV( Teuchos::TRANS, m, 2, 1.0e+00, &(*V)(0,0), ldv, &(*V)(0,2), 1, 0.0e+00, &(*R)(0,2), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, 2, &(*T)(0,0), ldt, &(*R)(0,2), 1 );
   blas.GEMV( Teuchos::NO_TRANS, m, 2, -1.0e+00, &(*V)(0,0), ldv, &(*R)(0,2), 1, 1.0e+00, &(*V)(0,2), 1 );
   for( i=0; i<2; i++ ){ (*R)(i,2) = (*V)(i,2); }

   (*R)(2,2) = (*V)(2,2);
   norma2 =  blas.DOT( m-(2+1), &(*V)(2+1,2), 1, &(*V)(2+1,2), 1);
   norma = sqrt( (*R)(2,2) * (*R)(2,2) + norma2 );
   (*R)(2,2) = ( (*R)(2,2) > 0 ) ? ( (*R)(2,2) + norma ) : ( (*R)(2,2) - norma );
   (*T)(2,2) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*R)(2,2) * (*R)(2,2) ) );
   for( i=0; i<2; i++ ){ (*V)(i,2) = +0.0e+00; }
   (*V)(2,2) = 1.0e+00;
   blas.SCAL( m-(2+1), ( 1.0e+00 / (*R)(2,2) ), &(*V)(2+1,2), 1 );
   (*R)(2,2) = ( (*R)(2,2) > 0 ) ? ( - norma ) : ( + norma );

   (*Q)(2,2) = 1.0e+00 - (*T)(2,2);
   for( i = 2+1; i < m ; i++ ){ (*Q)(i,2) = - ( (*V)(i,2) ) * ( (*T)(2,2) ); }
   for( i = 0; i < 2 ; i++ ){ (*Q)(i,2) = 0.0e+00; }

   blas.GEMV( Teuchos::TRANS, m, 2, 1.0e+00, &(*V)(0,0), ldv, &(*Q)(0,2), 1, 0.0e+00, &(*T)(0,2), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, 2, &(*T)(0,0), ldt, &(*T)(0,2), 1 );
   blas.GEMV( Teuchos::NO_TRANS, m, 2, -1.0e+00, &(*V)(0,0), ldv, &(*T)(0,2), 1, 1.0e+00, &(*Q)(0,2), 1 );

   for( i=0; i<2; i++){ (*T)(i,2) = (*V)(2,i); }
   blas.GEMV( Teuchos::TRANS, m-(2+1), 2, 1.0e+00, &(*V)(2+1,0), ldv, &(*V)(2+1,2), 1, 1.0e+00, &(*T)(0,2), 1 );
   blas.SCAL( 2, -(*T)(2,2), &(*T)(0,2), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, 2, &(*T)(0,0), ldt, &(*T)(0,2), 1 );
   
   // Orthogonality Check
   orth_check->putScalar();
   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, 3, m, 1.0e+00, &(*Q)(0,0), ldq, 0.0e+00, &(*orth_check)(0,0), n);
   for( i=0; i<3; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   orth_val = orth_check->normFrobenius();
   printf("\n|| I - Q''Q || = %3.3e, ", orth_val);

   // Representativity check
   repres_check->putScalar();
   for( i = 0; i < 3; i++ ){ blas.COPY( m, &(*Q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, 3, 1.0e+00, &(*R)(0,0), ldr, &(*repres_check)(0,0), m );
   for( k=0; k<3; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
   repres_val = repres_check->normFrobenius();
   printf("|| A - QR || = %3.3e \n", repres_val);
*/   
  
   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////
   //
   // Step 4
   //

//   j = 3;

   for( j=2; j<n; j++){

      blas.GEMV( Teuchos::TRANS, m, j, 1.0e+00, &(*V)(0,0), ldv, &(*V)(0,j), 1, 0.0e+00, &(*R)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*R)(0,j), 1 );
      blas.GEMV( Teuchos::NO_TRANS, m, j, -1.0e+00, &(*V)(0,0), ldv, &(*R)(0,j), 1, 1.0e+00, &(*V)(0,j), 1 );
      for( i=0; i<j; i++ ){ (*R)(i,j) = (*V)(i,j); }

      (*R)(j,j) = (*V)(j,j);
      norma2 =  blas.DOT( m-(j+1), &(*V)(j+1,j), 1, &(*V)(j+1,j), 1);
      norma = sqrt( (*R)(j,j) * (*R)(j,j) + norma2 );
      (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( (*R)(j,j) + norma ) : ( (*R)(j,j) - norma );
      (*T)(j,j) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*R)(j,j) * (*R)(j,j) ) );
      for( i=0; i<j; i++ ){ (*V)(i,j) = +0.0e+00; }
      (*V)(j,j) = 1.0e+00; 
      blas.SCAL( m-(j+1), ( 1.0e+00 / (*R)(j,j) ), &(*V)(j+1,j), 1 );
      (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( - norma ) : ( + norma );
 
      (*Q)(j,j) = 1.0e+00 - (*T)(j,j);
      for( i = j+1; i < m ; i++ ){ (*Q)(i,j) = - ( (*V)(i,j) ) * ( (*T)(j,j) ); }
      for( i = 0; i < j ; i++ ){ (*Q)(i,j) = 0.0e+00; }

      blas.GEMV( Teuchos::TRANS, m, j, 1.0e+00, &(*V)(0,0), ldv, &(*Q)(0,j), 1, 0.0e+00, &(*T)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );
      blas.GEMV( Teuchos::NO_TRANS, m, j, -1.0e+00, &(*V)(0,0), ldv, &(*T)(0,j), 1, 1.0e+00, &(*Q)(0,j), 1 );

      for( i=0; i<j; i++){ (*T)(i,j) = (*V)(j,i); }
      blas.GEMV( Teuchos::TRANS, m-(j+1), j, 1.0e+00, &(*V)(j+1,0), ldv, &(*V)(j+1,j), 1, 1.0e+00, &(*T)(0,j), 1 );
      blas.SCAL( j, -(*T)(j,j), &(*T)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );

      // Orthogonality Check
      orth_check->putScalar();
      blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, j+1, m, 1.0e+00, &(*Q)(0,0), ldq, 0.0e+00, &(*orth_check)(0,0), n);
      for( i=0; i<j+1; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
      orth_val = orth_check->normFrobenius();
      printf("\n|| I - Q''Q || = %3.3e, ", orth_val);

      // Representativity check
      for( i = 0; i < j+1; i++ ){ blas.COPY( m, &(*Q)(0,i), 1, &(*repres_check)(0,i), 1); }
      blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, j+1, 1.0e+00, &(*R)(0,0), ldr, &(*repres_check)(0,0), m );
      for( k=0; k<j+1; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
      repres_val = repres_check->normFrobenius();
      printf("|| A - QR || = %3.3e \n", repres_val);
   
   }

}



/*
   blas.GEMV( Teuchos::TRANS, m, 3, 1.0e+00, &(*V)(0,0), ldv, &(*V)(0,3), 1, 0.0e+00, &(*R)(0,3), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, 3, &(*T)(0,0), ldt, &(*R)(0,3), 1 );
   blas.GEMV( Teuchos::NO_TRANS, m, 3, -1.0e+00, &(*V)(0,0), ldv, &(*R)(0,3), 1, 1.0e+00, &(*V)(0,3), 1 );
   for( i=0; i<3; i++ ){ (*R)(i,3) = (*V)(i,3); }

   (*R)(3,3) = (*V)(3,3);
   norma2 =  blas.DOT( m-(3+1), &(*V)(3+1,3), 1, &(*V)(3+1,3), 1);
   norma = sqrt( (*R)(3,3) * (*R)(3,3) + norma2 );
   (*R)(3,3) = ( (*R)(3,3) > 0 ) ? ( (*R)(3,3) + norma ) : ( (*R)(3,3) - norma );
   (*T)(3,3) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*R)(3,3) * (*R)(3,3) ) );
   for( i=0; i<3; i++ ){ (*V)(i,3) = +0.0e+00; }
   (*V)(3,3) = 1.0e+00; 
   blas.SCAL( m-(3+1), ( 1.0e+00 / (*R)(3,3) ), &(*V)(3+1,3), 1 );
   (*R)(3,3) = ( (*R)(3,3) > 0 ) ? ( - norma ) : ( + norma );

   (*Q)(3,3) = 1.0e+00 - (*T)(3,3);
   for( i = 3+1; i < m ; i++ ){ (*Q)(i,3) = - ( (*V)(i,3) ) * ( (*T)(3,3) ); }
   for( i = 0; i < 3 ; i++ ){ (*Q)(i,3) = 0.0e+00; }

   blas.GEMV( Teuchos::TRANS, m, 3, 1.0e+00, &(*V)(0,0), ldv, &(*Q)(0,3), 1, 0.0e+00, &(*T)(0,3), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, 3, &(*T)(0,0), ldt, &(*T)(0,3), 1 );
   blas.GEMV( Teuchos::NO_TRANS, m, 3, -1.0e+00, &(*V)(0,0), ldv, &(*T)(0,3), 1, 1.0e+00, &(*Q)(0,3), 1 );

   for( i=0; i<3; i++){ (*T)(i,3) = (*V)(3,i); }
   blas.GEMV( Teuchos::TRANS, m-(3+1), 3, 1.0e+00, &(*V)(3+1,0), ldv, &(*V)(3+1,3), 1, 1.0e+00, &(*T)(0,3), 1 );
   blas.SCAL( 3, -(*T)(3,3), &(*T)(0,3), 1 );
   blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, 3, &(*T)(0,0), ldt, &(*T)(0,3), 1 );

   // Orthogonality Check
   orth_check->putScalar();
   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, 4, m, 1.0e+00, &(*Q)(0,0), ldq, 0.0e+00, &(*orth_check)(0,0), n);
   for( i=0; i<4; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   orth_val = orth_check->normFrobenius();
   printf("\n|| I - Q''Q || = %3.3e, ", orth_val);

   // Representativity check
   for( i = 0; i < 4; i++ ){ blas.COPY( m, &(*Q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, 4, 1.0e+00, &(*R)(0,0), ldr, &(*repres_check)(0,0), m );
   for( k=0; k<4; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
   repres_val = repres_check->normFrobenius();
   printf("|| A - QR || = %3.3e \n", repres_val);
*/
