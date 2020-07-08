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
//#include<Kokkos_Core.hpp>
//#include<KokkosBlas.hpp>
//#include<Kokkos_Random.hpp>

typedef double ScalarType;
typedef Teuchos::ScalarTraits<ScalarType> SCT;
typedef typename SCT::magnitudeType MagnitudeType;
typedef Tpetra::Operator<ScalarType>             OP;
typedef Tpetra::MultiVector<ScalarType>          MV;
typedef Belos::OperatorTraits<ScalarType,MV,OP> OPT;
typedef Belos::MultiVecTraits<ScalarType,MV>    MVT;
typedef Tpetra::Import<> import_type;

using Tpetra::Operator;
using Tpetra::CrsMatrix;
using Tpetra::MultiVector;
using Teuchos::RCP;

// I've added
using Teuchos::outArg;
using std::endl;
using map_type = Tpetra::Map<>;
using LO = typename MV::local_ordinal_type;
using GO = typename MV::global_ordinal_type;
using Node = typename MV::node_type;

int main(int argc, char *argv[]) {

   Tpetra::ScopeGuard tpetraScope(&argc,&argv);
   {
   Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

   const int my_rank = comm->getRank();
   const int pool_size = comm->getSize();
   const Tpetra::Details::DefaultTypes::global_ordinal_type indexBase = 0;

   if (my_rank == 0) {
      std::cout << "Total number of processes: " << pool_size << std::endl;
   }

   Teuchos::BLAS<int,ScalarType> blas;

   int i, j, k, lda, ldq, ldv, ldr, ldt;
   int seed, numrhs = 867, m = -1, n = -1;
   int endingp, startingp;
   double norma, norma2, tmp = 0.0e+00; 
   size_t mloc, offset, local_m;
   MagnitudeType orth, repres, nrmA;

   seed = my_rank*m*m; srand(seed);

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   //
   // Get the data from the HB file and build the Map,Matrix
   //

//   std::string filename("bcsstk14.hb");
//   std::string filename("shift.hb");
//   std::string filename("test.hb");
   std::string filename("test1.hb");

   RCP<CrsMatrix<ScalarType> > Amat;
   Tpetra::Utils::readHBMatrix(filename,comm,Amat);
   RCP<const Tpetra::Map<> > map = Amat->getDomainMap();

   RCP<MV> AA;
   RCP<MV> QQ;
   AA = rcp( new MV(map,numrhs) );
   QQ = rcp( new MV(map,numrhs) );

   m = MVT::GetGlobalLength(*AA);
   n = MVT::GetNumberVecs(*AA);
   mloc = AA->getLocalLength();
   const Tpetra::global_size_t numGlobalIndices = m;

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > TT; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > RR; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 

   if (TT == Teuchos::null) {
     TT = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  
   if (RR == Teuchos::null) {
     RR = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType> );
   }  

   std::vector<double> dot(n);

   MVT::MvRandom( *AA );        // Randomize a MultiVector
   MVT::MvInit( *QQ );          // Initialize Q as zeros

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   lda = m, ldt = n;

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > A; 
   if (A == Teuchos::null) {
     A = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n) );
   }  

   // Checks as Serial Dense Matrices
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;

   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n,true) );
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );

   // Copy MultiVec AA into A
   {
   AA->sync_host();
   auto aa = AA->getLocalViewHost();
   for(i=0;i<mloc;i++){for(j=0;j<n;j++){ (*A)(i,j) = aa(i,j); }}
   }
   nrmA = A->normFrobenius(); // This may be wrong, i.e. only taking the norm on 1 process


   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Getting starting point and ending point
   if( my_rank < m - pool_size*mloc ){ startingp = ( m - ( pool_size - my_rank ) * mloc + 1 ); 
   } else { startingp = ( m - ( pool_size - (my_rank) ) * mloc ); }
   endingp = startingp + mloc - 1;  

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Step 0 -- initialization step FOR MultiVectors
   j = 0;

   Teuchos::Range1D index_prev(0,0);

   RCP<MV> a_j = MVT::CloneViewNonConst( *AA, index_prev );
   RCP<MV> q_j = MVT::CloneViewNonConst( *QQ, index_prev );

   MVT::SetBlock( *AA, index_prev, *a_j );
   MVT::SetBlock( *QQ, index_prev, *q_j );

   MVT::MvDot( *a_j, *a_j, dot );

   if (startingp == 0) {       // full
      local_m = ( mloc > j ) ? ( j ) : ( mloc );
       offset = 0;
   } else if (startingp > j) { // empty
       local_m = 0;
       offset = 0;
   } else {                    // part
       if( endingp > j ){
          local_m = j - startingp;
       } else { 
         local_m = mloc;
       }
       offset = 0;
   }  
   RCP<const map_type> submapj = rcp(new map_type (j+1, indexBase, comm, Tpetra::LocalGlobal::LocallyReplicated));
   RCP<const map_type> globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
   import_type importer (globalMap, submapj);
   RCP<MV> TopA_j = MVT::CloneCopy( *AA, index_prev );
   RCP<MV> Broadcast;
   Broadcast = rcp( new MV(submapj,1) );
   Broadcast->doImport (*TopA_j, importer, Tpetra::INSERT);
   {
   auto b = Broadcast->getLocalViewHost();
   (*RR)(0,0) = b(0,0);
   }

   norma = sqrt( dot[0] );
   norma2 = dot[0] - (*RR)(0,0) * (*RR)(0,0);
 
   (*TT)(0,0) = ( (*RR)(0,0) > 0 ) ? ( (*RR)(0,0) + norma ) : ( (*RR)(0,0) - norma );

   MVT::MvScale( *a_j, 1.0e+00 / (*TT)(0,0) );

   (*TT)(0,0) = ( 2.0e+00 ) / ( (1.0e+00) + norma2 / ( (*TT)(0,0) * (*TT)(0,0) ) ); 

   (*RR)(0,0) = ( (*RR)(0,0) > 0 ) ? ( - norma ) : ( + norma );

   MVT::Assign( *a_j, *q_j); 
   MVT::MvScale( *q_j, -(*TT)(0,0) );

   if( my_rank == 0 ){
      {
      auto qq = QQ->getLocalViewHost();
      qq(0,0) = 1.0e+00 - (*TT)(0,0);
      auto aa = AA->getLocalViewHost();
      aa(0,0) = 1.0e+00; 
      }
   }

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////
//   RCP<MV> A_j; 
   for( j=1; j<n; j++){

      Teuchos::Range1D index_prev1(0,j-1);
      Teuchos::Range1D index_prev2(j,j);
      work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(j,1) );

      RCP<MV> A_j = MVT::CloneViewNonConst( *AA, index_prev1 );
      RCP<MV> a_j = MVT::CloneViewNonConst( *AA, index_prev2 );
      RCP<MV> q_j = MVT::CloneViewNonConst( *QQ, index_prev2 );
 
      // Step 1:
      MVT::MvTransMv( (+1.0e+00), *A_j, *a_j, *work );              // One AllReduce
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, j, &(*TT)(0,0), ldt, &(*work)(0,0), 1 ); // SerialDense Ops
      MVT::MvTimesMatAddMv( (-1.0e+00), *A_j, *work, (+1.0e+00), *a_j );      

      // Step 2:
      if (startingp == 0) {       // full
         local_m = ( mloc > j ) ? ( j ) : ( mloc );
          offset = 0;
      } else if (startingp > j) { // empty
          local_m = 0;
          offset = 0;
      } else {                    // part
          if( endingp > j ){
             local_m = j - startingp;
          } else { 
            local_m = mloc;
          }
          offset = 0;
      }  
      RCP<const map_type> submapj = rcp(new map_type (j+1, indexBase, comm, Tpetra::LocalGlobal::LocallyReplicated));
      RCP<const map_type> globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
      import_type importer (globalMap, submapj);
      RCP<MV> TopA_j = MVT::CloneCopy( *AA, index_prev2 );
      RCP<MV> Broadcast;
      Broadcast = rcp( new MV(submapj,1) );
      Broadcast->doImport (*TopA_j, importer, Tpetra::INSERT);
      {
      auto b = Broadcast->getLocalViewHost();
      for(i=0;i<j+1;i++) (*RR)(i,j) = b(i,0);              // One broadcast
//      auto a = AA->getLocalView();
      auto a = AA->getLocalViewHost();
      if( startingp < j ){ // j and below you do nothing to, we want to zero out 1:j-1
         if( endingp < j ){
            for(i=0;i<mloc;i++){ a(i,j) = 0.0e+00; }
         } else {
            for(i=0;i<j-startingp;i++){ a(i,j) = 0.0e+00; }
         }
      }
      }

      MVT::MvDot( *a_j, *a_j, dot );               // Two AllReduce
      norma2 = dot[0] - (*RR)(j,j) * (*RR)(j,j);
      norma = sqrt( dot[0] );
      (*RR)(j,j) = ( (*RR)(j,j) > 0 ) ? ( (*RR)(j,j) + norma ) : ( (*RR)(j,j) - norma );
      (*TT)(j,j) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*RR)(j,j) * (*RR)(j,j) ) );
      MVT::MvScale( *a_j, ( 1 / (*RR)(j,j) ) );
      (*RR)(j,j) = ( (*RR)(j,j) > 0 ) ? ( - norma ) : ( + norma );
      {
      if( startingp <= j && endingp >= j ){
         auto a = AA->getLocalViewHost();
         k = j - startingp;
         a(k,j) = 1.0e+00; 
      }
      }

      // Step 3
      MVT::Assign( *a_j, *q_j); 
      MVT::MvScale( *q_j, -(*TT)(j,j) );
      {
      auto q = QQ->getLocalViewHost();
      if( startingp <= j ){ 
         if( endingp >= j ){
            for(i=0;i<j-startingp;i++){ q(i,j) = 0.0e+00; }
            k = j - startingp;
            q(k,j) = 1.0e+00 - (*TT)(j,j);
         } else { 
            for(i=0;i<mloc;i++){ q(i,j) = 0.0e+00; }
         }
      }
      }
      for(i=0;i<j;i++) (*work)(i,0) = 0.0e+00; 
      MVT::MvTransMv( (+1.0e+00), *A_j, *q_j, *work );              // Three AllReduce
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*TT)(0,0), ldt, &(*work)(0,0), 1 ); // SerialDense Ops
      MVT::MvTimesMatAddMv( (-1.0e+00), *A_j, *work, (+1.0e+00), *q_j );    
//      MVT::MvPrint( *QQ, std::cout ); 
 
      // Step 4
      for(i=0;i<j;i++) (*work)(i,0) = 0.0e+00; 
      MVT::MvTransMv( (+1.0e+00), *A_j, *a_j, *work );               // Four AllReduce
      for(i=0;i<j;i++) (*TT)(i,j) = (*work)(i,0);
      blas.SCAL( j, -(*TT)(j,j), &(*TT)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*TT)(0,0), ldt, &(*TT)(0,j), 1 );

   }
//   if( my_rank == 0 ) for(i=0;i<n;i++){for(j=0;j<n;j++){ printf("%+3.2e, ", (*RR)(i,j) ); } printf("\n"); }

   {
   // Orthogonality Check
   orth_check->putScalar();
//   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, n, mloc, 1.0e+00, &(qq)(0,0), mloc, 0.0e+00, &(*orth_check)(0,0), n);
   MVT::MvTransMv( (+1.0e+00), *QQ, *QQ, *orth_check );
   for(i=0;i<n;i++){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
//   if( my_rank == 0 ) for(i=0;i<n;i++){for(j=0;j<n;j++){ printf("%+3.2e, ", (*orth_check)(i,j) ); } printf("\n"); }
   orth = orth_check->normFrobenius();

   // Representativity check
//   QQ->sync_host();
   auto q = QQ->getLocalViewHost();
   for(i=0;i<n;i++){ blas.COPY( mloc, &(q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, mloc, n, 1.0e+00, &(*RR)(0,0), n, &(*repres_check)(0,0), mloc );
   for( k=0; k<n; k++ ){ for( i=0; i<mloc; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } } 
   repres = repres_check->normFrobenius();
   } 

//   if( my_rank == 0 ){
      printf("m = %3d, n = %3d,  ",m,n);
      printf("|| I - Q'Q || = %3.3e, ", orth);
      printf("|| A - QR || / ||A|| = %3.3e \n", repres/nrmA);
//   }


//   {
//      auto a = AA->getLocalViewHost();
//      printf("**%3d**\n",my_rank);
//      for(i=0;i<mloc;i++){for(j=0;j<n;j++){ printf("%+3.2e, ", (a)(i,j) ); } printf("\n"); }
//   }
//      MVT::MvPrint( *AA, std::cout ); 

   }


   return 0;
}




