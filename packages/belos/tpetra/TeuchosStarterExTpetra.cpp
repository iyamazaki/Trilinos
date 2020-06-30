#include "Teuchos_BLAS.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"


// I/O for Harwell-Boeing files
#include <Tpetra_MatrixIO.hpp>

//#include <Teuchos_CommandLineProcessor.hpp>
//#include <Teuchos_ParameterList.hpp>
//#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include "BelosMultiVecTraits.hpp"
#include "BelosTpetraAdapter.hpp"

// I've added
#include <vector>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_CommHelpers.hpp>

using Tpetra::Operator;
using Tpetra::CrsMatrix;
using Tpetra::MultiVector;
using Teuchos::RCP;

// I've added
using Teuchos::outArg;
using std::endl;
  using Teuchos::REDUCE_SUM;
  using Teuchos::reduceAll;


typedef double ScalarType;
typedef Teuchos::ScalarTraits<ScalarType> SCT;
typedef typename SCT::magnitudeType MagnitudeType;
typedef Tpetra::Operator<ScalarType>             OP;
typedef Tpetra::MultiVector<ScalarType>          MV;
typedef Belos::OperatorTraits<ScalarType,MV,OP> OPT;
typedef Belos::MultiVecTraits<ScalarType,MV>    MVT;

int main(int argc, char *argv[]) {

   Tpetra::ScopeGuard tpetraScope(&argc,&argv);
   Teuchos::BLAS<int,ScalarType> blas;

   int i, j, k, lda, ldq, ldv, ldr, ldt;
   int numrhs = 5, m = -1, n = -1;
   double norma, norma2, tmp = 0.0e+00; 
   MagnitudeType orth, repres, nrmA;

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   //
   // Get the data from the HB file and build the Map,Matrix
   //

//   std::string filename("bcsstk14.hb");
//   std::string filename("shift.hb");
   std::string filename("test.hb");
//   std::string filename("test1.hb");

   RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
   RCP<CrsMatrix<ScalarType> > Amat;
   Tpetra::Utils::readHBMatrix(filename,comm,Amat);
   RCP<const Tpetra::Map<> > map = Amat->getDomainMap();

   RCP<MV> AA;
   RCP<MV> QQ;
   RCP<MV> VV;
   AA = rcp( new MV(map,numrhs) );
   QQ = rcp( new MV(map,numrhs) );
   VV = rcp( new MV(map,numrhs) );

   m = MVT::GetGlobalLength(*AA);
   n = MVT::GetNumberVecs(*AA);

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > TT; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > RR; 

   if (TT == Teuchos::null) {
     TT = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  
   if (RR == Teuchos::null) {
     RR = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  

   MVT::MvRandom( *AA );        // Randomize a MultiVector
   MVT::MvInit( *QQ );          // Initialize Q as zeros
   MVT::Assign( *AA, *VV );     // Copy A into V

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

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
     Q = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n,true) );
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

   // Checks as Serial Dense Matrices
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;

   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(m,n,true) );
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );

   // Generate Random A and Copy Into V
   A->random();
   for( i=0; i<n; i++){
      blas.COPY( m, &(*A)(0,i), 1, &(*V)(0,i), 1);
   }
   nrmA = A->normFrobenius();

   // Copy A into a MultiVec AA
   {
   AA->sync_host();
   auto aa = AA->getLocalViewHost();
   for(i=0;i<m;i++){for(j=0;j<n;j++){ aa(i,j) = (*A)(i,j); }}
   }

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Step 0 -- initialization step FOR MultiVectors
   j = 0;

   std::vector<double> dot(2);
   Teuchos::Range1D index_prev(0,0);

   MV a_j = *(AA->subView(index_prev));
   MV q_j = *(QQ->subView(index_prev));

   MVT::MvDot( a_j, a_j, dot );

   {
   AA->sync_host();
   auto aa = AA->getLocalViewHost();
   tmp = aa(0,0); 
   }

   norma = sqrt( dot[0] );
   norma2 = dot[0] - tmp * tmp;
 
   (*TT)(0,0) = ( tmp > 0 ) ? ( tmp + norma ) : ( tmp - norma );
   MVT::MvScale( a_j, 1.0e+00 / (*TT)(0,0) );
   (*TT)(0,0) = ( 2.0e+00 ) / ( (1.0e+00) + norma2 / ( (*TT)(0,0) * (*TT)(0,0) ) ); 

   (*RR)(0,0) = ( tmp > 0 ) ? ( - norma ) : ( + norma );

   MVT::Assign( a_j, q_j); 
   MVT::MvScale( q_j, - (*TT)(0,0) );

   {
   QQ->sync_host();
   auto qq = QQ->getLocalViewHost();
   qq(0,0) = 1.0e+00 - (*TT)(0,0);
   }

/*
   {
   QQ->sync_host();
   auto qq = QQ->getLocalViewHost();
   // Orthogonality Check
   orth_check->putScalar();
   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, 1, m, 1.0e+00, &(qq)(0,0), m, 0.0e+00, &(*orth_check)(0,0), 1);
   for( i=0; i<1; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   orth = orth_check->normFrobenius();
   printf("%3.2e, ",orth);
   // Representativity check
   for( i = 0; i < 1; i++ ){ blas.COPY( m, &(qq)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, m, 1, 1.0e+00, &(*RR)(0,0), ldv, &(*repres_check)(0,0), m );
   for( k=0; k<1; k++ ){ for( i=0; i<m; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } }
   repres = repres_check->normFrobenius();
   printf("%3.2e\n ",repres);
   }
*/

//   MVT::MvPrint( a_j, std::cout);
//   MVT::MvPrint( *AA, std::cout);
//   printf("\n%3.2f\n", dot[0]);
//   printf("\n%3.2f\n", tmp);


   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Step 0 -- initialization step FOR SerialDenseMatrices

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




