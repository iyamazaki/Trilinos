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

   int i, j;
   int Testing, seed, numrhs, m, n;
   double tmp; 
   const double one (1.0);
   const double zero (0.0);
   MagnitudeType orth(0.0), repres(0.0), nrmA(0.0);
   
   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   int ntests = 2;
   m = 20000; n = 50; Testing = 0;
   for( i = 1; i < argc; i++ ) {
      if( strcmp( argv[i], "-ntests" ) == 0 ) {
         ntests = atoi(argv[i+1]);
         i++;
      }
      if( strcmp( argv[i], "-m" ) == 0 ) {
         m = atoi(argv[i+1]);
         i++;
      }
      if( strcmp( argv[i], "-n" ) == 0 ) {
         n = atoi(argv[i+1]);
         i++;
      }
      if( strcmp( argv[i], "-testing" ) == 0 ) {
         Testing = atoi(argv[i+1]);
         i++;
      }
   }

   numrhs = n;

for (int test = 0; test < ntests; test++) {
   RCP<const map_type> map = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
   RCP<MV> Q = rcp( new MV(map,numrhs) );
   RCP<MV> A = rcp( new MV(map,numrhs) );
   RCP<MV> Q_j;
   RCP<MV> Q_j2;
   RCP<MV> q_j;
   RCP<MV> q_j2;
   RCP<MV> q_jm1;
   RCP<MV> q_jnew;

   m = MVT::GetGlobalLength(*Q);
   seed = my_rank*m*m; srand(seed);

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

   MVT::MvRandom( *Q ); 
 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > R; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 

   if (R == Teuchos::null) {
     R = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,2) );
   }  

   std::vector<double> dot(n);

   // Compute the Frobenius Norm of A
   if( Testing ) MVT::Assign( *Q, *A ); 
   if( Testing ){
      MVT::MvNorm(*Q,dot,Belos::TwoNorm);
      for(i=0;i<n;i++){ dot[i] = dot[i] * dot[i]; if(i!=0){ dot[0] += dot[i]; } } 
      nrmA = sqrt(dot[0]); 
   }

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Set up printers for output: 
   Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
   Teuchos::RCP<Belos::OutputManager<double> > printer_ = Teuchos::rcp( new Belos::OutputManager<double>(Belos::TimingDetails,outputStream) );
   std::string Label ="QR factor time DCGS2";
   std::string Label4 ="QR factor time MvDot operations in DCGS2";
   std::string Label2 ="QR factor time MvTransMv operations in DCGS2";
   std::string Label3 ="QR factor time MvTimesMatAddMv operations in DCGS2";
   std::string Label1 ="QR factor time MvAddMv operations in DCGS2";
   //(You can create multiple labels to time different kernels, if you like )

   //Initialize timer: (Do once per label, I think)
#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::RCP<Teuchos::Time> timerIRSolve_ = Teuchos::TimeMonitor::getNewCounter(Label);
   Teuchos::RCP<Teuchos::Time> timerIRSolve1_ = Teuchos::TimeMonitor::getNewCounter(Label1);
   Teuchos::RCP<Teuchos::Time> timerIRSolve2_ = Teuchos::TimeMonitor::getNewCounter(Label2);
   Teuchos::RCP<Teuchos::Time> timerIRSolve3_ = Teuchos::TimeMonitor::getNewCounter(Label3);
   Teuchos::RCP<Teuchos::Time> timerIRSolve4_ = Teuchos::TimeMonitor::getNewCounter(Label4);
#endif

   { //scope guard for timer
#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::TimeMonitor slvtimer(*timerIRSolve_);
#endif

   for( j=0; j<n; j++){

      if( j == 0 ){

      }

      if( j == 1 ){

         Teuchos::Range1D index_prev1(0,1);
         Teuchos::Range1D index_prev2(0,0);
         Teuchos::Range1D index_prev3(1,1);

         Teuchos::SerialDenseMatrix<int,ScalarType> work1 (Teuchos::View, *work, 2, 1, 0, 0);

         Q_j  = MVT::CloneViewNonConst( *Q, index_prev1 );
         q_j  = MVT::CloneViewNonConst( *Q, index_prev2 );
         q_j2 = MVT::CloneViewNonConst( *Q, index_prev3 );

	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve2_);
         #endif

         MVT::MvTransMv( one, *Q_j, *q_j, work1 );        // One AllReduce
	 }

         (*R)(0,0) = sqrt( work1(0,0) );                 
         MVT::MvScale( *q_j, 1/(*R)(0,0) );
         (*R)(0,1) = work1(1,0) / (*R)(0,0);

        { //scope guard for timer
        #ifdef BELOS_TEUCHOS_TIME_MONITOR
           Teuchos::TimeMonitor slvtimer(*timerIRSolve1_);
        #endif

	MVT::MvAddMv( one, *q_j2, (-(*R)(0,1)), *q_j, *q_j2 );
	}

      }
    
      if( j >= 2){

         Teuchos::Range1D index_prev (0,j-2);
         Teuchos::Range1D index_prev1(0,j-1);
         Teuchos::Range1D index_prev2(j-1,j);
         Teuchos::Range1D index_prev3(j-1,j-1);
         Teuchos::Range1D index_prev4(j,j);

         Teuchos::SerialDenseMatrix<int,ScalarType> work1 (Teuchos::View, *work, j, 2, 0, 0);

         Q_j2 = MVT::CloneViewNonConst( *Q, index_prev );
         Q_j = MVT::CloneViewNonConst( *Q, index_prev1 );
         q_j = MVT::CloneViewNonConst( *Q, index_prev2 );
         q_jm1 = MVT::CloneViewNonConst( *Q, index_prev3 );
         q_j2 = MVT::CloneViewNonConst( *Q, index_prev4 );

	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve2_);
         #endif

         MVT::MvTransMv( one, *Q_j, *q_j, work1 );        // One AllReduce
	 }

         for(i=0;i<j;i++) (*R)(i,j) = work1(i,1); 
         (*R)(j-1,j-1) = work1(j-1,0);

         tmp = zero; for(i=0;i<j-1;i++) tmp += work1(i,0) * (*R)(i,j);
         (*R)(j-1,j) = (*R)(j-1,j) - tmp;

         tmp = zero; for(i=0;i<j-1;i++) tmp += work1(i,0) * work1(i,0);
         (*R)(j-1,j-1) = sqrt( (*R)(j-1,j-1) - tmp );

         for(i=0;i<j-1;i++) (*R)(i,j-1) = (*R)(i,j-1) + work1(i,0); 
         (*R)(j-1,j) = (*R)(j-1,j) / (*R)(j-1,j-1);

	 // Note: This copies the first column work1 into work2, we need R in column 2
         Teuchos::SerialDenseMatrix<int,ScalarType> work2 (Teuchos::View, work1, j-1, 2, 0, 0);	 

         for(i=0;i<j-1;i++) work2(i,1) = (*R)(i,j);
	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve3_);
         #endif

         //Kokkos::Timer timer;
         //timer.reset();
         MVT::MvTimesMatAddMv( -one, *Q_j2, work2, one, *q_j );  
         //Kokkos::fence();
         //double mvtime = timer.seconds();
         //printf( " > %d: %e seconds\n",j,mvtime );
	 }
         MVT::MvScale( *q_jm1, ( one / (*R)(j-1,j-1) ) );
	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve1_);
         #endif

         MVT::MvAddMv( one, *q_j2, (-(*R)(j-1,j)), *q_jm1, *q_j2 );
	 }

      }

      if( j == n-1 ){

         Teuchos::Range1D index_prev1(0,j-1);
         Teuchos::Range1D index_prev2(j,j);

         Teuchos::SerialDenseMatrix<int,ScalarType> work1 (Teuchos::View, *work, j, 1, 0, 0);

         Q_j = MVT::CloneViewNonConst( *Q, index_prev1 );
         q_j = MVT::CloneViewNonConst( *Q, index_prev2 );
	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve2_);
         #endif

         MVT::MvTransMv( one, *Q_j, *q_j, work1 ); 
	 }
	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve4_);
         #endif
         MVT::MvDot( *q_j, *q_j, dot );
	 }

         tmp = zero; for(i=0;i<j-1;i++) tmp = tmp + ( work1(i,0) * work1(i,0) );
         (*R)(j,j) = sqrt( dot[0] - tmp );
 	 { //scope guard for timer
         #ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor slvtimer(*timerIRSolve3_);
         #endif

         //Kokkos::Timer timer;
         //timer.reset();
         MVT::MvTimesMatAddMv( -one, *Q_j, work1, one, *q_j );  
         //Kokkos::fence();
         //double mvtime = timer.seconds();
         //printf( " > %d: %e seconds\n",j,mvtime );
	 }
         for(i=0;i<j-1;i++) (*R)(i,j) = (*R)(i,j) + work1(i,0);
         MVT::MvScale( *q_j, ( one / (*R)(j,j) ) );

      }

   }

   } //end timer scope guard (i.e. Stop timing.)
   //Print final timing details:
   if (test == 0) {
     Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );
     Teuchos::TimeMonitor::zeroOutTimers ();  
   } else if (test == ntests-1) {
     Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );
   }

   if( Testing ){  
      // Orthogonality Check
      Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
      orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );
      orth_check->putScalar();
      MVT::MvTransMv( one, *Q, *Q, *orth_check );
      for(i=0;i<n;i++){(*orth_check)(i,i) = one - (*orth_check)(i,i);}
      orth = orth_check->normFrobenius();
      // Representativity check
      RCP<MV> repres_check  = rcp( new MV(map,numrhs) );
      MVT::MvTimesMatAddMv( one, *Q, *R, zero, *repres_check );
      MVT::MvAddMv( one, *A, -one, *repres_check, *Q );
      MVT::MvNorm(*Q,dot,Belos::TwoNorm);
      for(i=0;i<n;i++){ dot[i] = dot[i] * dot[i]; if(i!=0){ dot[0] += dot[i]; } } 
      repres = sqrt(dot[0]); 
      if( my_rank == 0 ){
         printf("m = %3d, n = %3d,  ",m,n);
         printf("|| I - Q'Q || = %3.3e, ", orth);
         printf("|| A - QR || / ||A|| = %3.3e \n", repres/nrmA);
      } 
   } else if (test == ntests-1) {
      if( my_rank == 0 ) printf("m = %3d, n = %3d\n",m,n);
   }
 
   }
}
   return 0;

}
