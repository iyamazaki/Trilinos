#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestRepository.hpp"
#include "BelosSolverManager.hpp"
#include "BelosSolverFactory.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosGmresPolySolMgr.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Details_CooMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "Ifpack2_Factory.hpp"
#include "KokkosBlas1_mult.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Zoltan2_PartitioningProblem.hpp"
#include "Zoltan2_XpetraCrsMatrixAdapter.hpp"
#include "Zoltan2_XpetraMultiVectorAdapter.hpp"
#define HAVE_GALERI
#if defined(HAVE_GALERI)
 #include "Galeri_XpetraMaps.hpp"
 #include "Galeri_XpetraProblemFactory.hpp"
#endif
#include <iostream>

//#define Tpetra_INST_INT_LONG_LONG

namespace { // (anonymous)

struct CommandLineOptions {
  std::string preproc_type{"cholqr"};
  int numRows {100000};
  int numColsStart {1};
  int numColsEnd   {100};
  bool benchQR {false};
  bool verbose {false};
};
CommandLineOptions commandLineOptions;

TEUCHOS_STATIC_SETUP()
{
  Teuchos::CommandLineProcessor& clp = Teuchos::UnitTestRepository::getCLP();
  clp.addOutputSetupOptions (true);
  clp.setOption ("preproc", &commandLineOptions.preproc_type,
                 "Preprocess type");
  clp.setOption ("numRows", &commandLineOptions.numRows,
                 "Number of rows");
  clp.setOption ("numColsStart", &commandLineOptions.numColsStart,
                 "Number of columns");
  clp.setOption ("numColsEnd", &commandLineOptions.numColsEnd,
                 "Number of columns");
  clp.setOption ("benchQR", "noBenchQR", &commandLineOptions.benchQR,
                 "Benchmark QR process");
  clp.setOption ("verbose", "quiet", &commandLineOptions.verbose,
                 "Verbosity");
}

void
testSolver (Teuchos::FancyOStream& out,
            bool& success, const bool verbose)
{
  using Teuchos::FancyOStream;
  using Teuchos::getFancyOStream;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcpFromRef;
  using std::endl;
  using map_type = Tpetra::Map<>;
  using MV = Tpetra::MultiVector<>;
  using OP = Tpetra::Operator<>;
  using SC = MV::scalar_type;
  using GO = map_type::global_ordinal_type;
  using LO = map_type::local_ordinal_type;
  using mag_type = MV::mag_type;
  using STS = Teuchos::ScalarTraits<SC>;
  using STM = Teuchos::ScalarTraits<mag_type>;
  using LOTS = Teuchos::OrdinalTraits<LO>;
  using GOTS = Teuchos::OrdinalTraits<GO>;

  using dense_vector_type = Teuchos::SerialDenseVector<LO, SC>;

  typedef Teuchos::TimeMonitor time_monitor_type;
  typedef Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<SC>> reader_type;
  typedef Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<SC>> writer_type;
  typedef MV::dual_view_type::t_dev local_vectors_type;
  // The Teuchos unit test framework likes to capture output to 'out',
  // and not print anything until the test is done.  This can hinder
  // debugging.  If the test crashes without useful output, try
  // setting this to 'true'.  That will change 'myOut' from an alias
  // to 'out', into a wrapper for std::cerr.
  constexpr bool debug = true;

  const SC ZERO = STS::zero ();
  const SC ONE = STS::one ();

  RCP<FancyOStream> myOutPtr =
    debug ? getFancyOStream (rcpFromRef (std::cerr)) : rcpFromRef (out);
  Teuchos::FancyOStream& myOut = *myOutPtr;

  auto comm = Tpetra::getDefaultComm ();
  const int myRank = comm->getRank ();
  const int numProcs = comm->getSize ();
  std::string preproc_type = commandLineOptions.preproc_type;

  int numWarms = (verbose ? 0 : 50);
  int numCalls = (verbose ? 1 : 5000);
  int numRows = commandLineOptions.numRows;
  int numColsStart = commandLineOptions.numColsStart;
  int numColsEnd = commandLineOptions.numColsEnd;
  RCP<map_type> map = rcp( new Tpetra::Map<LO,GO>(numRows, 0, comm) );
  if (myRank == 0) {
    myOut << endl << " " << preproc_type << endl;
    myOut << (STS::isComplex ? "> in complex type" : "> in real type") << endl;
    myOut << "  LO type       : " << LOTS::name()  << endl;
    myOut << "  GO type       : " << GOTS::name()  << endl;
    myOut << "  Global rows   : " << map->getGlobalNumElements() << endl;
    myOut << "  # Warmup      : " << numWarms << endl;
    myOut << "  # Calls       : " << numCalls << endl;
  }
  for (int numCols = numColsStart; numCols <= numColsEnd; numCols++) {
    int sketchSize = (preproc_type == "cholqr" ? numCols : ((preproc_type == "count" || preproc_type == "count-gauss") ? 2*(numCols*numCols) : 2*numCols));
    int sketchSize2 = 2*numCols;

    Teuchos::RCP<MV> B = rcp(new MV(map, numCols));
    MV G  = impl::makeStaticLocalMultiVector (*B, sketchSize,  numCols);
    MV G2 = impl::makeStaticLocalMultiVector (*B, sketchSize2, numCols);
    MV R  = impl::makeStaticLocalMultiVector (*B, numCols,     numCols);

    B->randomize ();
    /*{
      R.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *B, *B, ZERO);
      {
        auto R_loc = R.getLocalViewHost (Tpetra::Access::ReadWrite);
        lapack.POTRF ('U', numCols, R_loc.data(), R_loc.extent(0), &info);
      }
      {
        auto R_loc = R.getLocalViewDevice (Tpetra::Access::ReadOnly);
        auto Q_loc = B->getLocalViewDevice (Tpetra::Access::ReadWrite);
        KokkosBlas::trsm ("R", "U", "N", "N",
                          ONE, R_loc, Q_loc);
      }
    }*/
    if (preproc_type == "cholqr") {
      if (myRank == 0) myOut << " ------- cholQR numCols = " << numCols << " -------" << std::endl;
      #define USE_KK_GEMM
      #ifdef USE_KK_GEMM
      auto B_loc = B->getLocalViewDevice (Tpetra::Access::ReadOnly);
      auto R_loc = R.getLocalViewDevice (Tpetra::Access::OverwriteAll);
      #endif
      for (int k=0; k<numWarms; k++) {
        #ifdef USE_KK_GEMM
        KokkosBlas::gemm("T", "N", ONE, B_loc, B_loc, ZERO, R_loc);
        #else
        R.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *B, *B, ZERO);
        #endif
      }
      Kokkos::fence();
      {
        RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Matrix Sketch("+std::to_string(numCols)+","+std::to_string(numCalls)+"), CholQR");
        time_monitor_type LocalTimer (*randTimer);
        for (int k=0; k<numCalls; k++) {
          #ifdef USE_KK_GEMM
          KokkosBlas::gemm("T", "N", ONE, B_loc, B_loc, ZERO, R_loc);
          #else
          R.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *B, *B, ZERO);
          #endif
        }
        Kokkos::fence();
      }
    } else if (preproc_type == "gauss") {
      // Dense Gauss Sketch
      if (myRank == 0) myOut << " ------- GaussSketch numCols = " << numCols << "sketchSize = " << sketchSize << " -------" << std::endl;
      Teuchos::RCP<MV> Omega = rcp(new MV(map, sketchSize));
      Omega->randomize ();
      {
        #ifdef USE_KK_GEMM
        auto O_loc = Omega->getLocalViewDevice (Tpetra::Access::ReadOnly);
        auto B_loc = B->getLocalViewDevice (Tpetra::Access::ReadOnly);
        auto G_loc = G.getLocalViewDevice (Tpetra::Access::OverwriteAll);
        #endif
        for (int k=0; k<numWarms; k++) {
          #ifdef USE_KK_GEMM
          KokkosBlas::gemm("T", "N", ONE, O_loc, B_loc, ZERO, G_loc);
          #else
          G.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *Omega, *B, ZERO);
          #endif
        }
        Kokkos::fence();
        {
          RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Matrix Sketch("+std::to_string(numCols)+","+std::to_string(numCalls)+"), Gauss");
          time_monitor_type LocalTimer (*randTimer);
          for (int k=0; k<numCalls; k++) {
            #ifdef USE_KK_GEMM
            KokkosBlas::gemm("T", "N", ONE, O_loc, B_loc, ZERO, G_loc);
            #else
            G.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *Omega, *B, ZERO);
            #endif
          }
          Kokkos::fence();
        }
      }
    } else if (preproc_type == "count" || preproc_type == "count-gauss") {
      using execution_space = Kokkos::DefaultExecutionSpace;
      using host_execution_space = Kokkos::DefaultHostExecutionSpace;
      using host_memory_space    = typename host_execution_space::memory_space;

      using crsmat_t = KokkosSparse::CrsMatrix<double, int, execution_space, void, int>;
      using graph_t  = crsmat_t::StaticCrsGraphType;
      using rowmap_t = graph_t::row_map_type::non_const_type;
      using colind_t = graph_t::entries_type::non_const_type;
      using nzvals_t = crsmat_t::values_type::non_const_type;

      int locNumRows = B->getLocalLength();
      rowmap_t rowmap("rowmap", locNumRows+1);
      colind_t colind("colind", locNumRows);
      nzvals_t nzvals("nzvals", locNumRows);
      auto rowmap_h = Kokkos::create_mirror_view(rowmap);
      auto colind_h = Kokkos::create_mirror_view(colind);
      auto nzvals_h = Kokkos::create_mirror_view(nzvals);

      int seed_ind = 123;
      int seed_val = 456;
      std::default_random_engine generator_ind; generator_ind.seed(seed_ind);
      std::default_random_engine generator_val; generator_val.seed(seed_val);
      std::uniform_int_distribution<int> pick_ind(0, sketchSize-1);
      std::uniform_int_distribution<int> pick_val(0, 1);
      for (int i=0; i<locNumRows; i++) {
         colind_h(i) = pick_ind(generator_ind);
         nzvals_h(i) = int(2*pick_val(generator_val) - 1);
         rowmap_h(i) = i;
      }
      rowmap_h(locNumRows) = locNumRows;
      Kokkos::deep_copy(rowmap, rowmap_h);
      Kokkos::deep_copy(colind, colind_h);
      Kokkos::deep_copy(nzvals, nzvals_h);

      graph_t static_graph(colind, rowmap);
      crsmat_t Omega("CrsMatrix", sketchSize, nzvals, static_graph);
      /*printf( "\nA=[\n" );
      for (int i=0; i<numRows; i++) {
        for (int k=rowmap_h(i); k<rowmap_h(i+1); k++) printf("%d %d %.2f\n",i,colind_h(k),nzvals_h(k));
      }
      printf( "];\n" );*/

      bool tranSpMV = true;
      if (tranSpMV) {
        if (myRank == 0) myOut << " ------- countSketch(trans) numCols = " << numCols << "sketchSize = " << sketchSize << " -------" << std::endl;
        auto B_loc = B->getLocalViewDevice(Tpetra::Access::ReadOnly);
        auto G_loc = G.getLocalViewDevice(Tpetra::Access::OverwriteAll);

        for (int k=0; k<numWarms; k++) {
          KokkosSparse::spmv("T", ONE, Omega, B_loc, ZERO, G_loc);
        }
        Kokkos::fence();
        {
          RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Matrix Sketch("+std::to_string(numCols)+","+std::to_string(numCalls)+"), Count");
          time_monitor_type LocalTimer (*randTimer);
          for (int k=0; k<numCalls; k++) {
            KokkosSparse::spmv("T", ONE, Omega, B_loc, ZERO, G_loc);
          }
          Kokkos::fence();
        }
      } else {
        // Transpose
        if (myRank == 0) myOut << " ------- countSketch(non-trans) numCols = " << numCols << " -------" << std::endl;
        rowmap_t rowmapT("rowmap", sketchSize+1);
        colind_t colindT("colind", locNumRows);
        nzvals_t nzvalsT("nzvals", locNumRows);
        auto rowmapT_h = Kokkos::create_mirror_view(rowmapT);
        auto colindT_h = Kokkos::create_mirror_view(colindT);
        auto nzvalsT_h = Kokkos::create_mirror_view(nzvalsT);
        for (int i = 0; i <= sketchSize; i++) rowmapT_h(i) = 0;
        for (int i = 0; i < locNumRows; i++) rowmapT_h(colind_h(i)+1) ++;
        for (int i=0; i<sketchSize; i++) rowmapT_h(i+1) += rowmapT_h(i);
        for (int i = 0; i < locNumRows; i++) {
          nzvalsT_h(rowmapT_h(colind_h(i))) = nzvals_h(i);
          colindT_h(rowmapT_h(colind_h(i))) = i;
          rowmapT_h(colind_h(i)) ++;
        }
        for (int i=sketchSize; i>0; i--) rowmapT_h(i) = rowmapT_h(i-1);
        rowmapT_h(0) = 0;
        Kokkos::deep_copy(rowmapT, rowmapT_h);
        Kokkos::deep_copy(colindT, colindT_h);
        Kokkos::deep_copy(nzvalsT, nzvalsT_h);

        /*printf( "\nA=[\n" );
        for (int i=0; i<sketchSize; i++) {
          for (int k=rowmapT_h(i); k<rowmapT_h(i+1); k++) printf("%d %d %.2f\n",i,colindT_h(k),nzvalsT_h(k));
          }
        printf( "];\n" );*/
        graph_t static_graphT(colindT, rowmapT);
        Omega = crsmat_t("CrsMatrix", locNumRows, nzvalsT, static_graphT);
        {
          auto B_loc = B->getLocalViewDevice(Tpetra::Access::ReadOnly);
          auto G_loc = G.getLocalViewDevice(Tpetra::Access::OverwriteAll);

          for (int k=0; k<numWarms; k++) {
            KokkosSparse::spmv("N", ONE, Omega, B_loc, ZERO, G_loc);
          }
          Kokkos::fence();
          {
            RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Matrix Sketch("+std::to_string(numCols)+")");
            time_monitor_type LocalTimer (*randTimer);
            for (int k=0; k<numCalls; k++) {
              KokkosSparse::spmv("N", ONE, Omega, B_loc, ZERO, G_loc);
            }
            Kokkos::fence();
          }
        }
      }
      if (preproc_type == "count-gauss") {
        if (myRank == 0) myOut << " ------- CountGaussSketch numCols = " << numCols << " -------" << std::endl;
        local_vectors_type Omega("omega", sketchSize, sketchSize2);
        Kokkos::Random_XorShift64_Pool<execution_space> random(13718);
        Kokkos::fill_random(Omega, random, SC(1));
	{
          auto B_loc = G.getLocalViewDevice(Tpetra::Access::ReadOnly);
          auto G_loc = G2.getLocalViewDevice(Tpetra::Access::OverwriteAll);
	  printf( " (%d,%d) x (%d,%d) = (%d,%d)\n",Omega.extent(0),Omega.extent(1), B_loc.extent(0),B_loc.extent(1), G_loc.extent(0),G_loc.extent(1)); fflush(stdout);
          for (int k=0; k<numWarms; k++) {
            KokkosBlas::gemm("T", "N", ONE, Omega, B_loc, ZERO, G_loc);
          }
          Kokkos::fence();
          {
            RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Matrix Sketch("+std::to_string(numCols)+","+std::to_string(numCalls)+"), Gauss");
            time_monitor_type LocalTimer (*randTimer);
            for (int k=0; k<numCalls; k++) {
              KokkosBlas::gemm("T", "N", ONE, Omega, B_loc, ZERO, G_loc);
            }
            Kokkos::fence();
          }
        }
      }
    }
    // Generate R to preprocess
    int info;
    bool benchmarkQR = commandLineOptions.benchQR;
    int numWarms_ = (benchmarkQR ? numWarms : 1);
    int numCalls_ = (benchmarkQR ? numCalls : 0);
    Teuchos::LAPACK<LO, SC> lapack;
    if (preproc_type == "cholqr") {
      if (numProcs > 1) {
        auto R_loc = R.getLocalViewHost (Tpetra::Access::ReadWrite);
        //auto R_loc = R.getLocalViewDevice (Tpetra::Access::ReadWrite);
        int numSend = R_loc.extent(0)*numCols;
        MPI_Allreduce(MPI_IN_PLACE, R_loc.data(), numSend, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (numCalls > 0) {
          RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("MPI_Allreduce("+std::to_string(numCols)+")");
          time_monitor_type LocalTimer (*randTimer);
	  //printf( " MPI_Allreduce(%d x %d) %d\n",R_loc.extent(0),numCols,numCalls );
          for (int k=0; k<numCalls_; k++) {
            MPI_Allreduce(MPI_IN_PLACE, R_loc.data(), numSend, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          }
        }
      }
      {
        auto R_loc = R.getLocalViewHost (Tpetra::Access::ReadWrite);
        for (int k=0; k<numWarms_; k++) {
          lapack.POTRF ('U', numCols, R_loc.data(), R_loc.extent(0), &info);
        }
        // benchmark
        if (numCalls_ > 0) {
          RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Local POTRF("+std::to_string(numCols)+","+std::to_string(numCalls_)+")");
          time_monitor_type LocalTimer (*randTimer);
          for (int k=0; k<numCalls_; k++) {
            lapack.POTRF ('U', numCols, R_loc.data(), R_loc.extent(0), &info);
          }
        }
      }
    } else {
      // get workspace size
      int lwork = -1;
      int sketchSize_ = (preproc_type == "count-gauss" ? sketchSize2 : sketchSize);
      dense_vector_type tau (numCols, true);
      {
        SC TEMP;
        lapack.GEQRF (sketchSize_, numCols, &TEMP, sketchSize_,
                      tau.values (), &TEMP, lwork, &info);
        int lwork_geqrf = Teuchos::as<LO> (STS::real (TEMP));
        //lapack.ORGQR (sketchSize, numCols, numCols, &TEMP, sketchSize,
        //              tau.values (), &TEMP, lwork, &info);
        //int lwork_orgqr = Teuchos::as<LO> (STS::real (TEMP));
        //lwork = (lwork_geqrf > lwork_orgqr ? lwork_geqrf : lwork_orgqr);
        lwork = lwork_geqrf;
      }
      // workspace 
      dense_vector_type WORK (lwork, true);
      {
        auto G_loc = (preproc_type == "count-gauss" ? G2.getLocalViewHost (Tpetra::Access::ReadWrite) : G.getLocalViewHost (Tpetra::Access::ReadWrite));
        if (numProcs > 1) {
          int numSend = G_loc.extent(0)*numCols;
	  printf( " MPI_Allreduce(%d x %d) %d\n",G_loc.extent(0),numCols,numCalls );
          MPI_Allreduce(MPI_IN_PLACE, G_loc.data(), numSend, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          if (numCalls > 0) {
            RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("MPI_Allreduce("+std::to_string(numCols)+")");
            time_monitor_type LocalTimer (*randTimer);
            for (int k=0; k<numCalls_; k++) {
              MPI_Allreduce(MPI_IN_PLACE, G_loc.data(), numSend, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
          }
        }
        // warmup
        for (int k=0; k<numWarms_; k++) {
          // compute QR
          lapack.GEQRF (sketchSize_, numCols, G_loc.data(), G_loc.extent(0),
                        tau.values (), WORK.values (), lwork, &info);
        }
        // benchmark
        if (numCalls_ > 0) {
          RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Local HHQR("+std::to_string(sketchSize_)+"x"+std::to_string(numCols)+")");
          time_monitor_type LocalTimer (*randTimer);
          for (int k=0; k<numCalls_; k++) {
            // compute QR
            lapack.GEQRF (sketchSize_, numCols, G_loc.data(), G_loc.extent(0),
                          tau.values (), WORK.values (), lwork, &info);
            // extract R
            for (int i=0; i<numCols; i++) {
              // make sure positive diagonals
              if (STS::real (G_loc(i, i)) < STM::zero ()) {
                for (int j=i; j<numCols; j++) {
                  G_loc(i, j) = -G_loc(i, j);
                }
              }
            }
          }
        }
      }
    } // end of generate R

    // Preprocess B
    {
      auto R_loc = (preproc_type == "cholqr"      ?  R.getLocalViewDevice (Tpetra::Access::ReadOnly) :
                   (preproc_type == "count-gauss" ? G2.getLocalViewDevice (Tpetra::Access::ReadOnly) :
                                                     G.getLocalViewDevice (Tpetra::Access::ReadOnly)));
      if (preproc_type != "cholqr") {
        using range_type = Kokkos::pair<int, int>;
        R_loc = Kokkos::subview(R_loc, range_type(0, numCols), Kokkos::ALL());
      }
      auto Q_loc = B->getLocalViewDevice (Tpetra::Access::ReadWrite);
      // warmup
      for (int k=0; k<numWarms_; k++) {
        KokkosBlas::trsm ("R", "U", "N", "N",
                          ONE, R_loc, Q_loc);
      }
      Kokkos::fence();
      // benchmark
      if (numCalls_ > 0) {
        RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("Trsm("+std::to_string(numCols)+","+std::to_string(numCalls_)+")");
        time_monitor_type LocalTimer (*randTimer);
        for (int k=0; k<numCalls_; k++) {
          KokkosBlas::trsm ("R", "U", "N", "N",
                            ONE, R_loc, Q_loc);
        }
        Kokkos::fence();
      }
    }

    // CholQR to orthogonalize
    {
      // warmup
      for (int k=0; k<numWarms_; k++) {
        R.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *B, *B, ZERO);
        {
          auto R_loc = R.getLocalViewHost (Tpetra::Access::ReadWrite);
          lapack.POTRF ('U', numCols, R_loc.data(), R_loc.extent(0), &info);
        }
        {
          auto R_loc = R.getLocalViewDevice (Tpetra::Access::ReadOnly);
          auto Q_loc = B->getLocalViewDevice (Tpetra::Access::ReadWrite);
          KokkosBlas::trsm ("R", "U", "N", "N",
                            ONE, R_loc, Q_loc);
        }
      }
      Kokkos::fence();
      // benchmark
      if (numCalls_ > 0) {
        RCP< Teuchos::Time > randTimer = time_monitor_type::getNewCounter ("CholQR("+std::to_string(numCols)+")");
        time_monitor_type LocalTimer (*randTimer);
        for (int k=0; k<numCalls_; k++) {
          R.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *B, *B, ZERO);
          {
            auto R_loc = R.getLocalViewHost (Tpetra::Access::ReadWrite);
            lapack.POTRF ('U', numCols, R_loc.data(), R_loc.extent(0), &info);
          }
          {
            auto R_loc = R.getLocalViewDevice (Tpetra::Access::ReadOnly);
            auto Q_loc = B->getLocalViewDevice (Tpetra::Access::ReadWrite);
            KokkosBlas::trsm ("R", "U", "N", "N",
                              ONE, R_loc, Q_loc);
          }
        }
        Kokkos::fence();
      }
    }

    // Check
    if (verbose) {
      R.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, ONE, *B, *B, ZERO);
      {
        auto R_loc = R.getLocalViewHost (Tpetra::Access::ReadOnly);
        double orthoErr = 0.0;
        for (int i=0; i<numCols; i++) {
          for (int j=0; j<numCols; j++) {
            if (i == j) {
              orthoErr += (R_loc(i,j) - ONE) * (R_loc(i,j) - ONE);
            } else {
              orthoErr += R_loc(i,j) * R_loc(i,j);
            }
          }
        }
        if (myRank == 0) myOut << " Ortho Err : " << std::sqrt(orthoErr) << std::endl;
      }
    }
  }

  time_monitor_type::summarize();
  time_monitor_type::zeroOutTimers ();
}

TEUCHOS_UNIT_TEST( TpetraNativeSolvers, Diagonal )
{
  testSolver (out, success, commandLineOptions.verbose);
}

} // namespace (anonymous)

int main (int argc, char* argv[])
{
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  // warmup?
  //Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
}
