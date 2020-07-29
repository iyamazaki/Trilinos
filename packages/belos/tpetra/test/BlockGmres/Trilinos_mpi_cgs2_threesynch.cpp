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
#include <vector>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_CommHelpers.hpp>
//#include <Teuchos_Time.hpp>
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

   RCP<MV> Q = rcp( new MV(map,numrhs) );

   mloc = Q->getLocalLength();
   m = MVT::GetGlobalLength(*Q);
   const Tpetra::global_size_t numGlobalIndices = m;
   ldr = n, ldt = n;
   seed = my_rank*m*m; srand(seed);

   MVT::MvRandom( *Q ); 
 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > A; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > R; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 

   if (R == Teuchos::null) {
     R = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType> );
   }  
   if (A == Teuchos::null) {
     A = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n) );
   }  

   // Checks as Serial Dense Matrices
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > repres_check;

   repres_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(mloc,n,true) );
   orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );

   std::vector<double> dot(n);

   // Copy MultiVec A into SerialDense AA for checks
   {
   Q->sync_host();
   auto q = Q->getLocalViewHost();
   for(i=0;i<mloc;i++){for(j=0;j<n;j++){ (*A)(i,j) = q(i,j); }}
   }
   nrmA = A->normFrobenius();  // This may be wrong, i.e. only taking the norm on 1 process
                               // Not sure if mpi process is built in here

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Set up printers for output: 
   Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
   Teuchos::RCP<Belos::OutputManager<double> > printer_ = Teuchos::rcp( new Belos::OutputManager<double>(Belos::TimingDetails,outputStream) );
   std::string Label ="QR factor time CGS2";
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

         Teuchos::Range1D index_prev2(j,j);
         RCP<MV> q_j = MVT::CloneViewNonConst( *Q, index_prev2 );

         MVT::MvDot( *q_j, *q_j, dot ); 
         (*R)(0,0) = sqrt( dot[0] );                 
         MVT::MvScale( *q_j, 1/(*R)(0,0) );
         

      } else {

         Teuchos::Range1D index_prev1(0,j-1);
         Teuchos::Range1D index_prev2(j,j);
         work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(j,1) ); 

         RCP<MV> Q_j = MVT::CloneViewNonConst( *Q, index_prev1 );
         RCP<MV> q_j = MVT::CloneViewNonConst( *Q, index_prev2 );

         // Step 1:
         MVT::MvTransMv( (+1.0e+00), *Q_j, *q_j, *work );              // One AllReduce
         MVT::MvTimesMatAddMv( (-1.0e+00), *Q_j, *work, (+1.0e+00), *q_j );  
         for(i=0;i<j;i++) (*R)(i,j) = (*work)(i,0);

         MVT::MvTransMv( (+1.0e+00), *Q_j, *q_j, *work );              // Two AllReduce
         MVT::MvTimesMatAddMv( (-1.0e+00), *Q_j, *work, (+1.0e+00), *q_j );  
         for(i=0;i<j;i++) (*R)(i,j) += (*work)(i,0);

         MVT::MvDot( *q_j, *q_j, dot );                                // Three AllReduce
         (*R)(j,j) = sqrt( dot[0] );
         MVT::MvScale( *q_j, ( 1 / (*R)(j,j) ) );

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
   for( k=0; k<n; k++ ){ for( i=0; i<mloc; i++ ){  (*repres_check)(i,k) = (*A)(i,k) - (*repres_check)(i,k); } } 
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




