#include "Teuchos_BLAS.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "BelosConfigDefs.hpp"
#include "BelosOutputManager.hpp"

//#include <Teuchos_CommandLineProcessor.hpp>
//#include <Teuchos_ParameterList.hpp>
//#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>


// I/O for Harwell-Boeing files
#include <Tpetra_MatrixIO.hpp>
#include "BelosMultiVecTraits.hpp"
#include "BelosTpetraAdapter.hpp"
#include <MatrixMarket_Tpetra.hpp>

// I've added
//#include <Teuchos_Time.hpp>
#include <vector>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_CommHelpers.hpp>
#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
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
   Teuchos::BLAS<int,ScalarType> blas;

   const int my_rank = comm->getRank();
   const int pool_size = comm->getSize();
   const Tpetra::Details::DefaultTypes::global_ordinal_type indexBase = 0;

   if (my_rank == 0) {
      std::cout << "Total number of processes: " << pool_size << std::endl;
   }

   int i, j, k, ldr, ldt;
   int seed, numrhs = 50, m = -1, n = -1;
   int endingp, startingp;
   double norma, norma2; 
   size_t mloc, offset, local_m;
   MagnitudeType orth, repres, nrmA;

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   //
   // Get the data from the HB file and build the Map,Matrix
   //

//   std::string filename("bcsstk14.hb");
//   std::string filename("shift.hb");
//   std::string filename("test.hb");
//   std::string filename("test1.hb");
//   std::string filename("a0nsdsil.hb");

//   RCP<CrsMatrix<ScalarType> > Amat;
//   Tpetra::Utils::readHBMatrix(filename,comm,Amat);
//   RCP<const Tpetra::Map<> > map = Amat->getDomainMap();

///   RCP<CrsMatrix<ST> > A;
   //Tpetra::Utils::readHBMatrix(filename,comm,A);
//   A = Tpetra::MatrixMarket::Reader<CrsMatrix<ST> >::readSparseFile(filename,comm);
//   RCP<const Tpetra::Map<> > map = A->getDomainMap();

   m = 20000; n = 50;
   for( i = 1; i < argc; i++ ) {
      if( strcmp( argv[i], "-m" ) == 0 ) {
         m = atoi(argv[i+1]);
         i++;
      }
      if( strcmp( argv[i], "-n" ) == 0 ) {
         n = atoi(argv[i+1]);
         i++;
      }
   }

   numrhs = n;

   RCP<const map_type> map = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));

   RCP<MV> A = rcp( new MV(map,numrhs) );
   RCP<MV> Q = rcp( new MV(map,numrhs) );

   mloc = A->getLocalLength();
   m = MVT::GetGlobalLength(*A);
   const Tpetra::global_size_t numGlobalIndices = m;
   ldr = n, ldt = n;
   seed = my_rank*m*m; srand(seed);

   MVT::MvRandom( *A ); 
   MVT::MvInit( *Q );
 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > AA; 
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
   if (AA == Teuchos::null) {
     AA = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n) );
   }  

   // Checks as Serial Dense Matrices
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;

   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n,true) );
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );

   std::vector<double> dot(n);

   // Copy MultiVec A into SerialDense AA
   {
   A->sync_host();
   auto a = A->getLocalViewHost();
   for(i=0;i<mloc;i++){for(j=0;j<n;j++){ (*AA)(i,j) = a(i,j); }}
   }
//   nrmA = AA->normFrobenius(); // This may be wrong, i.e. only taking the norm on 1 process
                                 // Not sure if mpi process is built in here
   MVT::MvNorm(*A,dot,Belos::TwoNorm);
   for(i=0;i<n;i++){ dot[i] = dot[i] * dot[i]; if(i!=0){ dot[0] += dot[i]; } } 
   nrmA = sqrt(dot[0]); 

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Set up printers for output: 
   Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
   Teuchos::RCP<Belos::OutputManager<double> > printer_ = Teuchos::rcp( new Belos::OutputManager<double>(Belos::TimingDetails,outputStream) );
   std::string Label ="QR factor time ";
   //(You can create multiple labels to time different kernels, if you like )

   //Initialize timer: (Do once per label, I think)
#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::RCP<Teuchos::Time> timerIRSolve_ = Teuchos::TimeMonitor::getNewCounter(Label);
#endif

   { //scope guard for timer
#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::TimeMonitor slvtimer(*timerIRSolve_);
#endif

   // Getting starting point and ending point for each process
   if( my_rank < m - pool_size*mloc ){ startingp = ( m - ( pool_size - my_rank ) * mloc + 1 ); 
   } else { startingp = ( m - ( pool_size - (my_rank) ) * mloc ); }
   endingp = startingp + mloc - 1;  

   for( j=0; j<n; j++){

      if( j == 0 ){

         Teuchos::Range1D index_prev(0,0);

         RCP<MV> a_j = MVT::CloneViewNonConst( *A, index_prev );
         RCP<MV> q_j = MVT::CloneViewNonConst( *Q, index_prev );

         MVT::SetBlock( *A, index_prev, *a_j );
         MVT::SetBlock( *Q, index_prev, *q_j );
         MVT::MvDot( *a_j, *a_j, dot );

         RCP<const map_type> submapj = rcp(new map_type (j+1, indexBase, comm, Tpetra::LocalGlobal::LocallyReplicated));
         RCP<const map_type> globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
         import_type importer(globalMap, submapj);
         RCP<MV> TopA_j = MVT::CloneCopy( *A, index_prev );
         RCP<MV> Broadcast = rcp( new MV(submapj,1) );
         Broadcast->doImport(*TopA_j, importer, Tpetra::INSERT);
         {
         auto b = Broadcast->getLocalViewHost();
         (*R)(0,0) = b(0,0);
         }

         norma = sqrt( dot[0] );
         norma2 = dot[0] - (*R)(0,0) * (*R)(0,0);
         (*T)(0,0) = ( (*R)(0,0) > 0 ) ? ( (*R)(0,0) + norma ) : ( (*R)(0,0) - norma );
         MVT::MvScale( *a_j, 1.0e+00 / (*T)(0,0) );
         (*T)(0,0) = ( 2.0e+00 ) / ( (1.0e+00) + norma2 / ( (*T)(0,0) * (*T)(0,0) ) ); 
         (*R)(0,0) = ( (*R)(0,0) > 0 ) ? ( - norma ) : ( + norma );

         MVT::Assign( *a_j, *q_j); 
         MVT::MvScale( *q_j, -(*T)(0,0) );
         if( my_rank == 0 ){
            {
            auto q = Q->getLocalViewHost();
            q(0,0) = 1.0e+00 - (*T)(0,0);
            auto a = A->getLocalViewHost();
            a(0,0) = 1.0e+00; 
            }
         }

      } else {

         Teuchos::Range1D index_prev1(0,j-1);
         Teuchos::Range1D index_prev2(j,j);
         work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(j,1) ); // Not sure if I should be doing this within an algorithm. I would prefer to use T_j as the workspace needed..

         RCP<MV> A_j  = MVT::CloneViewNonConst( *A, index_prev1 );
         RCP<MV> a_j  = MVT::CloneViewNonConst( *A, index_prev2 );
         RCP<MV> q_j  = MVT::CloneViewNonConst( *Q, index_prev2 );

         // Step 1:
         MVT::MvTransMv( (+1.0e+00), *A_j, *a_j, *work );              // One AllReduce
         blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 );
         MVT::MvTimesMatAddMv( (-1.0e+00), *A_j, *work, (+1.0e+00), *a_j );      

         // Step 2:
         RCP<const map_type> submapj = rcp(new map_type (j+1, indexBase, comm, Tpetra::LocalGlobal::LocallyReplicated));
         RCP<const map_type> globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
         import_type importer (globalMap, submapj);
         RCP<MV> TopA_j = MVT::CloneCopy( *A, index_prev2 );
         RCP<MV> Broadcast;
         Broadcast = rcp( new MV(submapj,1) );
         Broadcast->doImport (*TopA_j, importer, Tpetra::INSERT);
         {
         auto b = Broadcast->getLocalViewHost();
         for(i=0;i<j+1;i++) (*R)(i,j) = b(i,0);                        // One broadcast
         auto a = A->getLocalViewHost();
         if( startingp < j ){ 
            if( endingp < j ){
               for(i=0;i<mloc;i++){ a(i,j) = 0.0e+00; }
            } else {
               for(i=0;i<j-startingp;i++){ a(i,j) = 0.0e+00; }
            }
         }
         }

         if (startingp >= j+1) { // full
             local_m = mloc;
             offset = 0;
         } else if (startingp+mloc < j+1) { // empty
             local_m = 0;
             offset = mloc-1;
         } else { // part
             local_m = (startingp+mloc)-j-1;
             offset = j+1-startingp;
         }   
         RCP<const Tpetra::Map<LO,GO,Node> >  submapjj;
         submapjj = Tpetra::createContigMapWithNode<LO,GO,Node>(Teuchos::Range1D::INVALID, local_m, comm);
         RCP<MV> A_jj = MVT::CloneCopy( *A, index_prev2 );
         RCP<MV> a_jj = A_jj->offsetViewNonConst(submapjj, offset); 

         MVT::MvDot( *a_jj, *a_jj, dot );                                // Two AllReduce
         norma2 = dot[0];
         norma = sqrt( dot[0]  + (*R)(j,j) * (*R)(j,j) );
         (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( (*R)(j,j) + norma ) : ( (*R)(j,j) - norma );
         (*T)(j,j) = (2.0e+00) / ( (1.0e+00) + norma2 / ( (*R)(j,j) * (*R)(j,j) ) );
         MVT::MvScale( *a_j, ( 1 / (*R)(j,j) ) );
         (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( - norma ) : ( + norma );
         {
         if( startingp <= j && endingp >= j ){
            auto a = A->getLocalViewHost();
            k = j - startingp;
            a(k,j) = 1.0e+00; 
         }
         }

         // Step 3
         MVT::Assign( *a_j, *q_j); 
         MVT::MvScale( *q_j, -(*T)(j,j) );
         {
         auto q = Q->getLocalViewHost();
         if( startingp <= j ){ 
            if( endingp >= j ){
               for(i=0;i<j-startingp;i++){ q(i,j) = 0.0e+00; }
               k = j - startingp;
               q(k,j) = 1.0e+00 - (*T)(j,j);
            } else { 
               for(i=0;i<mloc;i++){ q(i,j) = 0.0e+00; }
            }
         }
         }
         for(i=0;i<j;i++) (*work)(i,0) = 0.0e+00; 
         MVT::MvTransMv( (+1.0e+00), *A_j, *q_j, *work );              // Three AllReduce
         blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 ); 
         MVT::MvTimesMatAddMv( (-1.0e+00), *A_j, *work, (+1.0e+00), *q_j );    
 
         // Step 4
         for(i=0;i<j;i++) (*work)(i,0) = 0.0e+00; 
         MVT::MvTransMv( (+1.0e+00), *A_j, *a_j, *work );               // Four AllReduce
         for(i=0;i<j;i++) (*T)(i,j) = (*work)(i,0);
         blas.SCAL( j, -(*T)(j,j), &(*T)(0,j), 1 );
         blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );

      }

   }

   } //end timer scope guard (i.e. Stop timing.)
   //Print final timing details:
   Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );
   
   {

   // Orthogonality Check
   orth_check->putScalar();
   MVT::MvTransMv( (+1.0e+00), *Q, *Q, *orth_check );
   for(i=0;i<n;i++){(*orth_check)(i,i) =  1.0e+00 - (*orth_check)(i,i);}
   orth = orth_check->normFrobenius();

   // Representativity check
   auto q = Q->getLocalViewHost();
   for(i=0;i<n;i++){ blas.COPY( mloc, &(q)(0,i), 1, &(*repres_check)(0,i), 1); }
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




