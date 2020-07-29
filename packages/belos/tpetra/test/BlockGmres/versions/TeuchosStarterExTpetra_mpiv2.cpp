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
   int seed, numrhs = 2, m = -1, n = -1;
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
   std::string filename("test.hb");
//   std::string filename("test1.hb");

   RCP<CrsMatrix<ScalarType> > Amat;
   Tpetra::Utils::readHBMatrix(filename,comm,Amat);
   RCP<const Tpetra::Map<> > map = Amat->getDomainMap();

   RCP<MV> A;
   RCP<MV> Q;
   A = rcp( new MV(map,numrhs) );
   Q = rcp( new MV(map,numrhs) );

   m = MVT::GetGlobalLength(*A);
   n = MVT::GetNumberVecs(*A);
   mloc = A->getLocalLength();

   const Tpetra::global_size_t numGlobalIndices = m;

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > T; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > R; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 

   if (T == Teuchos::null) {
     T = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  
   if (R == Teuchos::null) {
     R = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType> );
   }  

   std::vector<double> dot(n);

   MVT::MvRandom( *A );        // Randomize a MultiVector
   MVT::MvInit( *Q );          // Initialize Q as zeros

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   lda = m, ldt = n;

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > AA; 
   if (AA == Teuchos::null) {
     AA = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n) );
   }  

   // Checks as Serial Dense Matrices
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;

   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n,true) );
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );

   // Copy MultiVec A into SerialDense AA
   {
//   A->sync_host();
   auto a = A->getLocalViewHost();
   for(i=0;i<mloc;i++){for(j=0;j<n;j++){ (*AA)(i,j) = a(i,j); }}
   }
   nrmA = AA->normFrobenius();

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Find the starting point and ending point globally for each process
   if( my_rank < m - pool_size*mloc ){ startingp = ( m - ( pool_size - my_rank ) * mloc + 1 ); 
   } else { startingp = ( m - ( pool_size - (my_rank) ) * mloc ); }
   endingp = startingp + mloc - 1; 
 
//   printf("**%2d**, ",my_rank);
//   for(i=0;i<mloc;i++){ printf("%2d, ", ( startingp + i ) ); } printf("\n");
 
//   Teuchos::ArrayRCP< const double > testing(mloc);
//   jj = 1;
//   testing = A->getData(jj);

//   This is me offsetting by j on a multivec
//   j = 3;
//   if (startingp >= j) { // full
//       local_m = mloc;
//       offset = 0;
//   } else if (startingp+mloc < j) { // empty
//       local_m = 0;
//       offset = mloc-1;
//   } else { // part
//       local_m = (startingp+mloc)-j;
//       offset = j-startingp;
//   }   
//   RCP<const Tpetra::Map<LO,GO,Node> >  submap;
//   submap = Tpetra::createContigMapWithNode<LO,GO,Node>(Teuchos::Range1D::INVALID, local_m, comm);
//   Teuchos::Range1D index(0,2);
//   RCP<MV> A1 = MVT::CloneCopy( *A, index );
//   RCP<MV> A2 = A1->offsetViewNonConst(submap, offset);
 
//   MVT::MvPrint( *A, std::cout );
//   MVT::MvPrint( *A2, std::cout );

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Step 0 -- initialization step FOR MultiVectors
   j = 0;

   Teuchos::Range1D index_prev(0,0);

   RCP<MV> a_j = MVT::CloneViewNonConst( *A, index_prev );
   RCP<MV> q_j = MVT::CloneViewNonConst( *Q, index_prev );

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
   RCP<MV> TopA_j = MVT::CloneCopy( *A, index_prev );
   RCP<MV> Broadcast;
   Broadcast = rcp( new MV(submapj,1) );
   Broadcast->doImport (*TopA_j, importer, Tpetra::INSERT);
   {
   auto b = Broadcast->getLocalViewHost();
   tmp = b(j,0);
   }

   norma = sqrt( dot[0] );
   norma2 = dot[0] - tmp * tmp;

   (*T)(0,0) = ( tmp > 0 ) ? ( tmp + norma ) : ( tmp - norma );
   MVT::MvScale( *a_j, 1.0e+00 / (*T)(0,0) );
   (*T)(0,0) = ( 2.0e+00 ) / ( (1.0e+00) + norma2 / ( (*T)(0,0) * (*T)(0,0) ) ); 
   (*R)(0,0) = ( tmp > 0 ) ? ( - norma ) : ( + norma );

   MVT::Assign( *a_j, *q_j); 
   MVT::MvScale( *q_j, -(*T)(0,0) );

   {
   if( my_rank == 0 ){
      //Q->sync_host();
      auto q = Q->getLocalViewHost();
      q(0,0) = 1.0e+00 - (*T)(0,0);
   }
   }

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   RCP<MV> BotA_j;
   RCP<MV> BlockA;
   RCP<const Tpetra::Map<LO,GO,Node> > submap;

   for( j=1; j<n; j++){

      Teuchos::Range1D index_prev1(0,j-1); // Grabs columns 0:j-1
      Teuchos::Range1D index_prev2(j,j);   // Grabs column j
      Teuchos::Range1D index_prev3(0,j);   // Grabs columns 0:j
      work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(j,1) ); // Workspace - may not be needed

      // This logic sets up broadcasting the top j by j block of A_{1:m,1:j}
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
      submapj = rcp(new map_type (j+1, indexBase, comm, Tpetra::LocalGlobal::LocallyReplicated));
      globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
      import_type importer (globalMap, submapj);

      TopA_j = MVT::CloneCopy( *A, index_prev3 );
      Broadcast = rcp( new MV(submapj,j+1) );
      Broadcast->doImport (*TopA_j, importer, Tpetra::INSERT);
      {
      auto b = Broadcast->getLocalViewHost();
      for(i=0;i<j;i++){ (*R)(i,j) = b(i,j); } // Column
      for(i=0;i<j;i++){ (*R)(j,i) = b(j,i); } // Row
      (*R)(j,j) = b(j,j);                     // Overlapping element
      }

      // This sets up the logic to grab the lower block A_{j+1:m,1:j-1} used for computation
      if (startingp >= j+1) {            // full
          local_m = mloc;
          offset = 0;
      } else if (startingp+mloc < j+1) { // empty
          local_m = 0;
          offset = mloc-1;
      } else {                           // part
          local_m = (startingp+mloc)-j-1;
          offset = j+1-startingp;
      }   
      submap = Tpetra::createContigMapWithNode<LO,GO,Node>(Teuchos::Range1D::INVALID, local_m, comm);

      // This grabs the bottom block of A_{j+1:m,1:j} 
      RCP<MV> BlockA = MVT::CloneCopy( *A, index_prev1 );
      RCP<MV> BotA_j = BlockA->offsetViewNonConst(submap, offset);

      // This grabs the bottom column of A_{j+1:m,j}
      RCP<MV> ColA = MVT::CloneCopy( *A, index_prev2 );
      RCP<MV> a_j = ColA->offsetViewNonConst(submap, offset);
      
      // Step 1:
      MVT::MvTransMv( (+1.0e+00), *BotA_j, *a_j, *work );
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::TRANS, Teuchos::UNIT_DIAG, j, &(*R)(0,0), n, &(*work)(0,0), 1 ); 
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 ); 
      MVT::MvTimesMatAddMv( (-1.0e+00), *BotA_j, *work, (+1.0e+00), *a_j );   
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::UNIT_DIAG, j, &(*R)(0,0), n, &(*work)(0,0), 1 ); 
      for(i=0;i<j;i++){ (*R)(i,j) = (*R)(i,j) - (*work)(i,0); }
   
      // Step 2:
      MVT::MvDot( *a_j, *a_j, dot ); 
      norma2 = dot[0] - (*R)(j,j) * (*R)(j,j);
      norma = sqrt( dot[0] );
      (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( (*R)(j,j) + norma ) : ( (*R)(j,j) - norma );
      (*T)(j,j) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*R)(j,j) * (*R)(j,j) ) );
      MVT::MvScale( *a_j, ( 1 / (*R)(j,j) ) );
      (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( - norma ) : ( + norma );

      // Step 3
//      MVT::Assign( *a_j, *q_j); 
//      MVT::MvScale( *q_j, -(*T)(j,j) );
      {
      auto q = Q->getLocalViewHost();
      auto a = A->getLocalViewHost();
      if( startingp == 0 ){
         if( endingp >= j){
            for(i=0;i<j;i++) q(i,j) = 0.0e+00;         
            for(i=j+1;i<mloc;i++) q(i,j) = a(i,j) * (-(*T)(j,j));   
            q(j,j) = 1.0e+00 - (*T)(j,j); 
         } else {
            for(i=0;i<mloc;i++) q(i,j) = 0.0e+00;     
         }
      } else if( startingp > j ){      
         for(i=0;i<mloc;i++) q(i,j) = a(i,j) * (-(*T)(j,j));   
      } else {
         for(i=0;i<j;i++) q(i,j) = 0.0e+00;         
         for(i=j+1;i<mloc;i++) q(i,j) = a(i,j) * (-(*T)(j,j));   
         q(j,j) = +1.0e+00 - (*T)(j,j); 
      }
      }
      // This grabs the bottom column of Q_{j+1:m,j}
      RCP<MV> ColQ = MVT::CloneCopy( *Q, index_prev2 );
      RCP<MV> q_j = ColQ->offsetViewNonConst(submap, offset);

      for(i=0;i<j;i++) (*work)(i,0) = 0.0e+00; 
      MVT::MvTransMv( (+1.0e+00), *BotA_j, *q_j, *work );
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::TRANS, Teuchos::UNIT_DIAG, j, &(*R)(0,0), n, &(*work)(0,0), 1 ); 
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 ); 
      MVT::MvTimesMatAddMv( (-1.0e+00), *BotA_j, *work, (+1.0e+00), *q_j );    

      // Step 4
      for(i=0;i<j;i++) (*work)(i,0) = 0.0e+00; 
      MVT::MvTransMv( (+1.0e+00), *BotA_j, *a_j, *work );
      blas.TRMV( Teuchos::LOWER_TRI, Teuchos::TRANS, Teuchos::UNIT_DIAG, j, &(*R)(0,0), n, &(*work)(0,0), 1 ); 
      for(i=0;i<j;i++) (*T)(i,j) = (*work)(i,0);
      blas.SCAL( j, -(*T)(j,j), &(*T)(0,j), 1 );
      blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );

   }

   {
   // Orthogonality Check
   orth_check->putScalar();
//   blas.SYRK( Teuchos::UPPER_TRI, Teuchos::TRANS, n, mloc, 1.0e+00, &(qq)(0,0), mloc, 0.0e+00, &(*orth_check)(0,0), n);
   MVT::MvTransMv( (+1.0e+00), *Q, *Q, *orth_check );
   if( my_rank == 0 ) for( i=0; i<n; i++ ){ (*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i); }
   if( my_rank == 0 )for(i=0;i<n;i++){for(j=0;j<n;j++){ printf("%+3.2e, ", (*orth_check)(i,j) ); } printf("\n"); }
   orth = orth_check->normFrobenius();

   // Representativity check
//   QQ->sync_host();
   auto q = Q->getLocalViewHost();
   for( i = 0; i < n; i++ ){ blas.COPY( mloc, &(q)(0,i), 1, &(*repres_check)(0,i), 1); }
   blas.TRMM( Teuchos::RIGHT_SIDE, Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, mloc, n, 1.0e+00, &(*R)(0,0), n, &(*repres_check)(0,0), mloc );
   for( k=0; k<n; k++ ){ for( i=0; i<mloc; i++ ){  (*repres_check)(i,k) = (*AA)(i,k) - (*repres_check)(i,k); } } 
   repres = repres_check->normFrobenius();
   } 

   if( my_rank == 0 ){
      printf("m = %3d, n = %3d,  ",m,n);
      printf("|| I - Q'Q || = %3.3e, ", orth);
      printf("|| A - QR || / ||A|| = %3.3e \n", repres/nrmA);
   }

   }
   return 0;
}




