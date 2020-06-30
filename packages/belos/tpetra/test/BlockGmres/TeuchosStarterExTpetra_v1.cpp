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
   int numrhs = 5, m = numrhs, n = numrhs;
   double norma, norma2;
   MagnitudeType orth, repres, nrmA;

   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   //
   // Get the data from the HB file and build the Map,Matrix
   //
//   std::string filename("shift.hb");
   std::string filename("test.hb");
//   std::string filename("test1.hb");
//   std::string filename("bcsstk14.hb");

   RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
   RCP<CrsMatrix<ScalarType> > Amat;
   Tpetra::Utils::readHBMatrix(filename,comm,Amat);
   RCP<const Tpetra::Map<> > map = Amat->getDomainMap();

     // Here we are using the Tpetra::MultiVector class functions:
//   Teuchos::ArrayView<MagnitudeType> norms;
//   std::vector< Teuchos::ScalarTraits<double>::magnitudeType > norms;

   // Create initial vectors
   // This is an RCP to a Tpetra::MultiVector:
   RCP<MV> AA;
   RCP<MV> QQ;
   RCP<MV> VV;
//   RCP<MV> TT;
   AA = rcp( new MV(map,numrhs) );
   QQ = rcp( new MV(map,numrhs) );
   VV = rcp( new MV(map,numrhs) );
//   TT = rcp( new MV(map,1) );
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > TT; 
   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > RR; 
   if (TT == Teuchos::null) {
     TT = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  
   if (RR == Teuchos::null) {
     RR = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(n,n) );
   }  

   // Here we are acting on MultiVec(s) using Belos MultiVectorTraits:
   MVT::MvRandom( *AA );        // Randomize a MultiVector
   MVT::MvInit( *VV );          // Initialize V as zeros
   MVT::MvInit( *QQ, 1.0e+00 ); // Initialize Q as ones
// MVT::MvInit( *QQ );          // Initialize Q as zeros
// MVT::Assign( *AA, *VV );     // Copy A into V

//   k = MVT::GetGlobalLength(*VV);
//   k = MVT::HasConstantStride(*VV);
//   printf("%3.2d\n\n",k);

//   RCP<Tpetra::Vector<ScalarType>> vv;
//   vv = rcp( new Tpetra::Vector<ScalarType>(map,numrhs) );
//   MVT::MvPrint( *AA, std::cout);
//   MVT::MvPrint( *vv, std::cout);
//   MVT::MvPrint( *TT, std::cout);

//   std::vector<double> w(27,5);
//   std::vector<int> w(5,5);
//   std::vector<int> w(5,0);
//   for(i=0;i<5;i++)for(j=0;j<5;j++) w[i,j] = 0;
//   w[0] = 1, w[1] = 0, w[2] = 0, w[3] = 0, w[4] = 0;
//   for(i=0;i<5;i++) printf("%1.1d, ", w[i]);
//   printf("\n\n");
//   MVT::SetBlock( *QQ, w, *VV);


//   TT = MVT::CloneCopy( *AA, w );
//   MVT::CloneView( *AA, w );

//   MVT::MvPrint( *AA, std::cout);
//   MVT::MvPrint( *VV, std::cout);
//   MVT::MvPrint( *QQ, std::cout);

//   std::vector<double> nrms(5,0);
//   for(i=0;i<5;i++) nrms[i] = 0.0e+00;
//   for(i=0;i<5;i++) printf("%1.1f, ", nrms[i]);
//   printf("\n");

//   MVT::MvNorm( *AA, nrms, Belos::TwoNorm ); 
//   for(i=0;i<5;i++) printf("%1.1f, ", nrms[i]);
//   printf("\n");
//   std::cout << "The norm of column 1 of X is: " << nrms[0] << std::endl;

//   QQ->update( 1.0e+00, *AA, 1.0e+00); // QQ + 1 * AA
//   QQ->update( 1.0e+00, *AA, 0.0e+00, *VV, 0.0e+00); //Copy 1 * AA and 0 * VV into 0 * QQ
//   MVT::MvPrint( *QQ, std::cout);

//   std::vector<ScalarType> dots(5,0);
//   MVT::MvDot( *AA, *AA, dots );
//   for(i=0;i<5;i++) printf("%3.2e, ", dots[i]);
//   printf("\n");

//   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > BB; 
//   if (BB == Teuchos::null) {
//     BB = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,double>(n,n) );
//   }  
//   MVT::MvTransMv( (+1.0e+00), *AA, *AA, *BB ); 
//   for(i=0;i<5;i++){for(j=0;j<5;j++){
//   printf("%+3.2f ", (*BB)(i,j));
//   }printf("\n");}
//   printf("\n");

//   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > BB; 
//   if (BB == Teuchos::null) {
//     BB = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,double>(5,1) );
//   }
//   (*BB)(0,0) = +1.0e+00;
//   for(i=0;i<5;i++){for(j=0;j<1;j++){
//   printf("%+3.2f ", (*BB)(i,j));
//   }printf("\n");}
//   printf("\n");

//   MVT::MvPrint( *AA, std::cout);
//   MVT::MvTimesMatAddMv( (+1.0e+00), *AA, *BB, (+0.0e+00), *TT ); 
//   MVT::MvPrint( *TT, std::cout);

//   std::vector<int> w(5,0);
//   for(i=0;i<5;i++) w[i] = 0;
//   MVT::SetBlock( *QQ, w, *VV);

//   MVT::MvPrint( *VV, std::cout);
//   Teuchos::RCP<Teuchos::ArrayView<ScalarType>> WW;
//   WW.subCopy(w);
//   RCP<MV> XX = this->subView (w);

//   MVT::MvPrint( *VV, std::cout);
//   std::vector<ScalarType> dots(5,0);
//   MVT::MvDot( *QQ, *VV, dots );
//   for(i=0;i<5;i++) printf("%3.2e, ", dots[i]);
//   printf("\n");

/*

// The following is taken from an example provided by Mark Hoemenn

   const int myRank = comm->getRank ();

   // Print out the Tpetra software version information.
   if (myRank == 0) {
     std::cout << Tpetra::version() << endl << endl;
   }

   // Type of the Tpetra::Map specialization to use.
   using map_type = Tpetra::Map<>;

   // The type of the Tpetra::Vector specialization to use.  The first
   // template parameter is the Scalar type.  The "Scalar" type is the
   // type of the values stored in the Tpetra::Vector.  You could use
   // Tpetra::Vector<>::scalar_type to get the default Scalar type.  We
   // will assume that it's double.
   //
   // using scalar_type = Tpetra::Vector<>::scalar_type;
   using vector_type = Tpetra::Vector<double>;

   // The "LocalOrdinal" (LO) type is the type of "local" indices.
   // The typedef is commented out to avoid "unused typedef" warnings.
   //
   //using local_ordinal_type = vector_type::local_ordinal_type;
   // The "GlobalOrdinal" (GO) type is the type of "global" indices.
   using global_ordinal_type = vector_type::global_ordinal_type;

   // Create a Tpetra Map
   // The total (global, i.e., over all MPI processes) number of
   // entries in the Map.
   //
   // For this example, we scale the global number of entries in the
   // Map with the number of MPI processes.  That way, you can run this
   // example with any number of MPI processes and every process will
   // still have a positive number of entries.
   const Tpetra::global_size_t numGlobalEntries = comm->getSize () * 5;

   // Index base of the Map.  We choose zero-based (C-style) indexing.
   const global_ordinal_type indexBase = 0;

   // Construct a Map that puts the same number of equations on each
   // MPI process.
   RCP<const map_type> contigMap = rcp (new map_type (numGlobalEntries, indexBase, comm));

   // Create a Tpetra Vector
   // Create a Vector with the Map we created above.
   // This version of the constructor will fill in the vector with zeros.
   vector_type x (contigMap);

   // Fill the Vector with a single number, or with random numbers
   // Set all entries of x to 42.0.
   x.putScalar (42.0);

   // norm2() is a collective, so we need to call it on all processes
   // in the Vector's communicator.
   auto x_norm2 = x.norm2 ();
   if (myRank == 0) {
     std::cout << "Norm of x (all entries are 42.0): " << x_norm2 << endl;
   }

   // Set the entries of x to (pseudo)random numbers.  Please don't
   // consider this a good parallel pseudorandom number generator.
   x.randomize ();
   x_norm2 = x.norm2 ();
   if (myRank == 0) {
     std::cout << "Norm of x (random numbers): " << x_norm2 << endl;
   }

   // Read the entries of the Vector
   {
      // Get a view of the Vector's entries.  The view has type
      // Kokkos::View.  Kokkos::View acts like an array, but is
      // reference-counted like std::shared_ptr or Teuchos::RCP.  This
      // means that it may persist beyond the lifetime of the Vector.  A
      // View is like a shallow copy of the data, so be careful
      // modifying the Vector while a view of it exists.  You may
      // decrement the reference count manually by assigning an empty
      // View to it.  We put this code in an inner scope (in an extra
      // pair of {}) so that the Kokkos::View will fall out of scope
      // before the next example, which modifies the entries of the
      // Vector.
      // We want a _host_ View.  Vector implements "dual view"
      // semantics.  This is really only relevant for architectures with
      // two memory spaces.
      x.sync_host ();
      auto x_2d = x.getLocalViewHost ();
 
      // getLocalView returns a 2-D View by default.  We want a 1-D
      // View, so we take a subview.
      auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);
 
      // x_data.extent (0) may be longer than the number of local
      // rows in the Vector, so be sure to ask the Vector for its
      // dimensions, rather than the ArrayRCP.
      const size_t localLength = x.getLocalLength ();
 
      // Count the local number of entries less than 0.5.
      // Use local indices to access the entries of x_data.
      size_t localCount = 0;
      for (size_t k = 0; k < localLength; ++k) {
        if (x_1d(k) < 0.5) {
          ++localCount;
        }
      }
 
      // "reduceAll" is a type-safe templated version of MPI_Allreduce.
      // "outArg" is like taking the address using &, but makes it more
      // clear that its argument is an output argument of a function.
      size_t globalCount = 0;
      reduceAll<int, size_t> (*comm, REDUCE_SUM, localCount, outArg (globalCount));

      // Find the total number of entries less than 0.5, over all
      // processes in the Vector's communicator.  Note the trick for
      // pluralizing the word "entry" conditionally on globalCount.
      if (myRank == 0) {
        std::cout << "x has " << globalCount << " entr"
            << (globalCount != 1 ? "ies" : "y")
            << " less than 0.5." << endl;
      }
   }

   // Modify the entries of the Vector
   {
      // Get a nonconst persisting view of the entries in the Vector.
      // "Nonconst" means that you may modify the entries.  "Persisting"
      // means that the view persists beyond the lifetime of the Vector.
      // Even after the Vector's destructor is called, the view won't go
      // away.  If you create two nonconst persisting views of the same
      // Vector, and modify the entries of one view during the lifetime
      // of the other view, the entries of the other view are undefined.
      x.sync_host ();
      auto x_2d = x.getLocalViewHost ();
      auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);

      // We're going to modify the data on host.
      x.modify_host ();

      // Use local indices to access the entries of x_data.
      // x_data.extent (0) may be longer than the number of local
      // rows in the Vector, so be sure to ask the Vector for its
      // dimensions.
      const size_t localLength = x.getLocalLength ();
      for (size_t k = 0; k < localLength; ++k) {
        // Add k (the local index) to every entry of x.  Treat 'double'
        // as a function to convert k (an integer) to double.
        x_1d(k) += double (k);
      }
      using memory_space = vector_type::device_type::memory_space;
      x.sync<memory_space> ();

      // Print the norm of x.
      x_norm2 = x.norm2 ();
      if (myRank == 0) {
        std::cout << "Norm of x (modified random numbers): " << x_norm2 << endl;
      }
   }
*/


//   {
//   AA->sync_host ();
//   auto x_2d = AA->getLocalViewHost ();
//   auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);
//   double tmpp;
//   x_2d(0,1) = 15.0;
//   tmpp = x_2d(0,1);
//   AA->modify_host ();
//   printf("\n %3.2e \n\n",tmpp);
//   }
//   MVT::MvPrint( *AA, std::cout);



   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   m = MVT::GetGlobalLength(*AA);
   n = MVT::GetNumberVecs(*AA);
//   printf("%3.2d, %3.2d\n",ll,pp);

   lda = m, ldq = m, ldv = m, ldr = n, ldt = n;

   // Main Serial Dense Matrices
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
//   A->print(std::cout);
//   MVT::MvPrint( *AA, std::cout);


   ////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////

   // Step 0 -- initialization step FOR MultiVectors
   double tmp = 0.0e+00; 
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
 
   {
   AA->sync_host();
   auto aa = AA->getLocalViewHost();
   QQ->sync_host();
   auto qq = QQ->getLocalViewHost();
   qq(0,0) = 1.0e+00 - (*TT)(0,0);
   for( i=1; i<m; i++ ){ qq(i,0) = - ( aa(i,0) ) * ( (*T)(0,0) ); }   
   }


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




