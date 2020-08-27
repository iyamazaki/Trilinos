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
//#include <Teuchos_Time.hpp>
//#include<Kokkos_Random.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MatrixIO.hpp>
#include "BelosMultiVecTraits.hpp"
#include "BelosTpetraAdapter.hpp"
#include <MatrixMarket_Tpetra.hpp>
#include <vector>                  // I've added from here and below
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_CommHelpers.hpp>
#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>

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

   int m, n, i, j;
   int seed;
   size_t mloc, offset, local_m;

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
   seed = my_rank*m*m; srand(seed);
   RCP<const map_type> map = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
   RCP<const map_type> globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   RCP<MV> A = rcp( new MV(map,n) );
   RCP<MV> x = rcp( new MV(map,1) );
   RCP<MV> y = rcp( new MV(map,1) );

   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,1) );
   }
   for(i=0;i<n;i++){
      (*work)(i,0) = (double)rand() / (double)(RAND_MAX) - 0.5e+00;
   }

   // Get local/global size and lengths
   mloc = A->getLocalLength();
   m = MVT::GetGlobalLength(*A);
   const Tpetra::global_size_t numGlobalIndices = m;

   // initialize
   MVT::MvRandom( *x ); 
   MVT::MvRandom( *y ); 
   MVT::MvRandom( *A ); 

   std::vector<double> dot(1);

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Set up printers for output: 
   Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
   Teuchos::RCP<Belos::OutputManager<double> > printer_ = Teuchos::rcp( new Belos::OutputManager<double>(Belos::TimingDetails,outputStream) );
   std::string Label =" Computation time for MvTimeMatAddMv ";

   //Initialize timer: (Do once per label)
#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::RCP<Teuchos::Time> timerIRSolve_ = Teuchos::TimeMonitor::getNewCounter(Label);
#endif

   { //scope guard for timer

#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::TimeMonitor slvtimer(*timerIRSolve_);
#endif

   for( j=0; j<n; j++){
         MVT::MvTimesMatAddMv( (-1.0e+00), *A, *work, (+1.0e+00), *y );      
   }
 
   } // end timer scope guard (i.e. Stop timing.)

   // Print final timing details:
   Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

   if( my_rank == 0 ){    
      printf("m = %3d, n = %3d, num_procs = %3d\n",m,n,pool_size);
   }

   }

   return 0;

}




