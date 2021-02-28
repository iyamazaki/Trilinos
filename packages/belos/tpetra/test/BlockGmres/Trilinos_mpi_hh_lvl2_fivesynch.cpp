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
#include <vector>                  
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
using range_type = Kokkos::pair<int, int>;

int main(int argc, char *argv[]) {

   Tpetra::ScopeGuard tpetraScope(&argc,&argv);
   {
   Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
   Teuchos::BLAS<int,ScalarType> blas;

   const ScalarType zero (0.0);
   const ScalarType one  (1.0);
   const ScalarType two  (2.0);

   const int my_rank = comm->getRank();
   const int pool_size = comm->getSize();
   const Tpetra::Details::DefaultTypes::global_ordinal_type indexBase = 0;

   int Testing, seed, numrhs;
   LO i, j, k, ldt;
   LO mloc;
   GO m, n, endingp, startingp;
   MagnitudeType orth (0.0);
   MagnitudeType repres (0.0);
   MagnitudeType nrmA (0.0);
   MagnitudeType norma (0.0);
   MagnitudeType norma2 (0.0);

   m = 20000; n = 50; Testing = 0;
   for( i = 1; i < argc; i++ ) {
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
   seed = my_rank*m*m; srand(seed);
   numrhs = n;
   RCP<const map_type> map = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));
   RCP<const map_type> globalMap = rcp(new map_type (m, indexBase, comm, Tpetra::GloballyDistributed));

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   //
   // Get the data from the HB file and build the Map,Matrix
   //
   // This block is if you want to use a specific matrix
   // and not a random matrix specified above

//   std::string filename("bcsstk14.hb");
//   std::string filename("shift.hb");
//   std::string filename("test.hb");
//   std::string filename("test1.hb");
//   std::string filename("a0nsdsil.hb");

//   RCP<CrsMatrix<ScalarType> > Amat;
//   Tpetra::Utils::readHBMatrix(filename,comm,Amat);
//   RCP<const Tpetra::Map<> > map = Amat->getDomainMap();

//   RCP<CrsMatrix<ST> > A;
//   Tpetra::Utils::readHBMatrix(filename,comm,A);
//   A = Tpetra::MatrixMarket::Reader<CrsMatrix<ST> >::readSparseFile(filename,comm);
//   RCP<const Tpetra::Map<> > map = A->getDomainMap();

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Define the needed matrices. A gets overwritten with V
   RCP<MV> A    = rcp( new MV(map,numrhs) );
   RCP<MV> Q    = rcp( new MV(map,numrhs) );
   RCP<MV> Acpy = rcp( new MV(map,numrhs) );

   RCP<const map_type> submapj;
   submapj = rcp(new map_type (n, indexBase, comm, Tpetra::LocalGlobal::LocallyReplicated));
   import_type importer(globalMap, submapj);
   RCP<const Tpetra::Map<LO,GO,Node> >  submapjj;
   RCP<MV> A_j;
   RCP<MV> a_j;
   RCP<MV> q_j;
   RCP<MV> TopA_j;
   RCP<MV> Broadcast = rcp( new MV(submapj,1) );

   // Get local/global size and lengths
   mloc = A->getLocalLength();
   m = MVT::GetGlobalLength(*A);
   ldt = n;

   // initialize
   MVT::MvRandom( *A ); 
   MVT::MvInit( *Q );
   if( Testing ) MVT::Assign( *A, *Acpy ); 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > T; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > R; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > work; 

   if (T == Teuchos::null) {
     T = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  
   if (R == Teuchos::null) {
     R = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) );
   }
   if (work == Teuchos::null) {
     work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType> );
   }  
   std::vector<double> dot(n);

   // Compute the Frobenius Norm of A
   if( Testing ){
      MVT::MvNorm(*A,dot,Belos::TwoNorm);
      for(i=0; i<n; i++){ dot[i] = dot[i] * dot[i]; if(i!=0){ dot[0] += dot[i]; } } 
      nrmA = sqrt(dot[0]); 
   }

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Set up printers for output: 
   Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
   Teuchos::RCP<Belos::OutputManager<double> > printer_ = Teuchos::rcp( new Belos::OutputManager<double>(Belos::TimingDetails,outputStream) );
   std::string Label ="QR factor time for HH level 2 5-synch ";

   //Initialize timer: (Do once per label)
#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::RCP<Teuchos::Time> timerIRSolve_ = Teuchos::TimeMonitor::getNewCounter(Label);
#endif

   { //scope guard for timer

#ifdef BELOS_TEUCHOS_TIME_MONITOR
   Teuchos::TimeMonitor slvtimer(*timerIRSolve_);
#endif

   // Getting starting point and ending point for each process
   if( my_rank == 0 ){ 
      startingp = 0; 
   } else if( my_rank < m - pool_size*mloc  ){ 
      startingp = ( m - ( pool_size - my_rank ) * mloc + 1 ); 
   } else { 
      startingp = ( m - ( pool_size - (my_rank) ) * mloc ); 
   } endingp = startingp + mloc - 1;  
   //printf("%3d, %3d, %3d, %3d\n",my_rank,mloc,startingp,endingp);

   // Begin the orthogonalization process
   for( j=0; j<n; j++){

      if( j == 0 ){

         Teuchos::Range1D index_prev(0,0);

         a_j = MVT::CloneViewNonConst( *A, index_prev );
         q_j = MVT::CloneViewNonConst( *Q, index_prev );

         TopA_j = MVT::CloneCopy( *A, index_prev );
         Broadcast->doImport(*TopA_j, importer, Tpetra::INSERT);
         {
         Broadcast->sync_host ();
         auto b = Broadcast->getLocalViewHost();
         (*R)(0,0) = b(0,0);
         if( my_rank == 0 ){
            {
            A->sync_host ();
            A->modify_host ();
            auto a = A->getLocalViewHost();
            a(0,0) = zero; 
            A->sync_device ();
            }
         }
         }

         MVT::MvDot( *a_j, *a_j, dot );
         norma = sqrt( dot[0] + (*R)(0,0) * (*R)(0,0) );
         norma2 = dot[0];
         (*T)(0,0) = ( (*R)(0,0) > 0 ) ? ( (*R)(0,0) + norma ) : ( (*R)(0,0) - norma );
         MVT::MvScale( *a_j, one / (*T)(0,0) );
         (*T)(0,0) = two / ( one + norma2 / ( (*T)(0,0) * (*T)(0,0) ) ); 
         (*R)(0,0) = ( (*R)(0,0) > 0 ) ? ( - norma ) : ( + norma );

         MVT::Assign( *a_j, *q_j); 
         MVT::MvScale( *q_j, -(*T)(0,0) );
         if( my_rank == 0 ){
            {
            Q->sync_host ();
            Q->modify_host ();
            auto q = Q->getLocalViewHost();
            q(0,0) = one - (*T)(0,0);
            Q->sync_device ();

            A->sync_host ();
            A->modify_host ();
            auto a = A->getLocalViewHost();
            a(0,0) = one; 
            A->sync_device ();
            }
         }

      } else {

         // Setting the index for applying V_{j-1} to a_j and constructing q_j
         Teuchos::Range1D index_prev1(0,j-1);
         Teuchos::Range1D index_prev2(j,j);
         work = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(j,1) ); 

         // Setting up matrices for computation
         A_j  = MVT::CloneViewNonConst( *A, index_prev1 );
         a_j  = MVT::CloneViewNonConst( *A, index_prev2 );
         q_j  = MVT::CloneViewNonConst( *Q, index_prev2 );

         // Step 1: (I - V_{j-1} T_{j-1}^T V_{j-1}^T ) a_j
         MVT::MvTransMv( one, *A_j, *a_j, *work );              // One AllReduce
         blas.TRMV( Teuchos::UPPER_TRI, Teuchos::TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 );
         MVT::MvTimesMatAddMv( -one, *A_j, *work, one, *a_j );      

         // Step 2: Broadcast R_{1:j,j}, construct v_j and \tau_j
         TopA_j = MVT::CloneCopy( *A, index_prev2 );
         Broadcast->doImport(*TopA_j, importer, Tpetra::INSERT);
         {
         Broadcast->sync_host ();
         auto b = Broadcast->getLocalViewHost();
         for(i=0;i<j+1;i++) (*R)(i,j) = b(i,0);                        // One broadcast
         // Setting the top j-1 elements in v_j to zero
         A->sync_host ();
         A->modify_host ();
         auto a = A->getLocalViewHost();
         if( startingp <= j ){ 
            if( endingp < j ){
               for(i=0;i<mloc;i++){ a(i,j) = zero; }
            } else {
               for(i=0;i<j-startingp+1;i++){ a(i,j) = zero; } 
            }
         }
         A->modify_device ();
         }

         MVT::MvDot( *a_j, *a_j, dot );                               // Two AllReduce
         norma2 = dot[0];
         norma = sqrt( dot[0]  + (*R)(j,j) * (*R)(j,j) );

         (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( (*R)(j,j) + norma ) : ( (*R)(j,j) - norma );
         (*T)(j,j) = two / ( one + norma2 / ( (*R)(j,j) * (*R)(j,j) ) );
         MVT::MvScale( *a_j, ( 1 / (*R)(j,j) ) );
         (*R)(j,j) = ( (*R)(j,j) > 0 ) ? ( - norma ) : ( + norma );
         {
         if( startingp <= j && endingp >= j ){
            k = j-startingp;
            #if 1
            A->sync_host ();
            A->modify_host ();
            auto a = A->getLocalViewHost();
            a(k,j) = one;
            A->sync_device ();
            #else
            A->sync_device ();
            A->modify_device ();
            auto a = A->getLocalViewDevice();
            auto akj = Kokkos::subview (a, range_type (k, k+1), range_type (j, j+1));
            Kokkos::deep_copy (akj, one);
            #endif
         }
         }

         // Step 3: Construct q_j = ( I - V_{j-1} T_{j-1} V_{j-1}^T )( I - v_j \tau_j v_j^T ) e_j
         MVT::Assign( *a_j, *q_j); 
         MVT::MvScale( *q_j, -(*T)(j,j) );
         {
         if( startingp <= j ){ 
            Q->sync_host ();
            Q->modify_host ();
            auto q = Q->getLocalViewHost();
            if( endingp >= j ){
               for(i=0;i<j-startingp;i++){ q(i,j) = zero; }
               k = j-startingp;
               q(k,j) = one - (*T)(j,j);
            } else { 
               for(i=0;i<mloc;i++){ q(i,j) = zero; }
            }
            Q->sync_device ();
         }
         }

	 for(i=0;i<j;i++) (*work)(i,0) = zero; 
         MVT::MvTransMv( one, *A_j, *q_j, *work );            // Three AllReduce
         blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*work)(0,0), 1 ); 
         MVT::MvTimesMatAddMv( -one, *A_j, *work, one, *q_j );    
 
         // Step 4: Construct T_{1:j-1,j} = -\tau_j T_{1:j-1,1:j-1} V_{1:j-1}^T v_j
         for(i=0;i<j;i++) (*work)(i,0) = zero; 
         MVT::MvTransMv( one, *A_j, *a_j, *work );             // Four AllReduce
         for(i=0;i<j;i++) (*T)(i,j) = (*work)(i,0);
         blas.SCAL( j, -(*T)(j,j), &(*T)(0,j), 1 );
         blas.TRMV( Teuchos::UPPER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, j, &(*T)(0,0), ldt, &(*T)(0,j), 1 );

      } 

   } // end of for loop for orthogonalization method

   } // end timer scope guard (i.e. Stop timing.)

   // Print final timing details:
   Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

   if( Testing ){  
      RCP<MV> repres_check  = rcp( new MV(map,numrhs) );
      Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > orth_check; 
      orth_check = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n,true) ); 
      // Orthogonality Check
      orth_check->putScalar();
      MVT::MvTransMv( one, *Q, *Q, *orth_check );
      for(i=0;i<n;i++){(*orth_check)(i,i) =  one - (*orth_check)(i,i);}
      orth = orth_check->normFrobenius();
      // Representativity check
      MVT::MvTimesMatAddMv( one, *Q, *R, zero, *repres_check );
      MVT::MvAddMv( one, *Acpy, -one, *repres_check, *Q );
      MVT::MvNorm(*Q,dot,Belos::TwoNorm);
      for(i=0;i<n;i++){ dot[i] = dot[i] * dot[i]; if(i!=0){ dot[0] += dot[i]; } } 
      repres = sqrt(dot[0]); 
   }

   if( my_rank == 0 ){    
      std::cout << "m = " << m << ", n = " << n << ", num_procs = " << pool_size << std::endl;
      if( Testing ) printf("|| I - Q'Q ||        = %3.3e\n", orth);
      if( Testing ) printf("|| A - QR || / ||A|| = %3.3e \n", repres/nrmA);
   }

   }

   return 0;

}




