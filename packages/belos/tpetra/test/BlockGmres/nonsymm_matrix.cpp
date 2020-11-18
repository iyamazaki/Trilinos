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
  std::string matrixFilename {""};
  std::string rhsFilename    {""};
  //
  int exaWindNumFiles {0};
  std::string exaWindFilename {""};
  //
  std::string preconditioner {""};
  std::string relaxation {"SGS MT"};
  std::string solverName {"TPETRA GMRES"};
  int maxOrthoSteps {0};
  std::string orthoType {"ICGS"};
  bool reortho {false};
  //
  int  matrixSize {0};
  int  nx2 {-1};
  int  nx3 {-1};
  bool useDiagMat {false};
  double diagAlpha {1e-8}; // first diagonal entry of Simonici matrix
  double diagShift {0.0};  // diagonal shift of Simonici matrix
  double diagPower {1.0};  // diagonal power of Simonici matrix
  double offDiagDiff {1.0 / 8.0};
  // mfh 14 Aug 2018: GMRES takes 20 iterations on this problem (with
  // offDiagDiff = 1/8).  We add 10 iterations to allow for rounding
  // error and differences in the algorithm.
  double tol {1e-8};
  int maxAllowedNumIters {30};
  int maxNumIters {100};
  int restartLength {30};
  int stepSize {1};
  bool useZoltan2  {false};
  bool useParMETIS {false};
  bool useSuperLU  {false};
  bool computeRitzValues {true};
  bool computeRitzValuesOnFly {false};
  bool useCholQR {false};
  bool useCholQR2 {false};
  bool delayHGeneration {false};
  bool doReortho2 {false};
  bool zeroInitialGuess {true};
  bool verbose {true};
  // RHS
  int  nrhs {1};
  int  blockSize {0};
  bool randomRHS {true};
  bool onesRHS {false};
  // GCRODR
  int numRecycledBlocks {5};
  // Schwarz
  int numRelaxationSweeps {1};
  int numInnerRelaxationSweeps {1};
  bool doBackwardSweep {false};
  bool useInnerSptrsv {false};
  bool useSolutionBasedRec {true};
  int clusterSize {16};
  // GmresPoly
  int polyDegrees {2};
};
CommandLineOptions commandLineOptions;

TEUCHOS_STATIC_SETUP()
{
  Teuchos::CommandLineProcessor& clp = Teuchos::UnitTestRepository::getCLP();
  clp.addOutputSetupOptions (true);
  clp.setOption ("solver", &commandLineOptions.solverName,
                 "Name of the solver to test.  Belos::SolverFactory::create "
                 "must accept this string.  Protect with double quotes if it "
                 "has spaces: e.g., \"TPETRA CG PIPELINE\".");
  //
  clp.setOption ("convergenceTol", &commandLineOptions.tol,
                 "Convergence tolerance");
  clp.setOption ("maxOrthoSteps", &commandLineOptions.maxOrthoSteps,
                 "Number of the max orthogonalization steps");
  clp.setOption ("ortho", &commandLineOptions.orthoType,
                 "Name of the orthogonalization procedure");
  clp.setOption ("reortho", "noReortho", &commandLineOptions.reortho,
                 "Whether to reorthogonalize");
  clp.setOption ("maxNumAllowedIters",
                 &commandLineOptions.maxAllowedNumIters,
                 "Maximum number of iterations that the solver is "
                 "allowed to take before converging, in order for "
                 "the test to pass.");
  clp.setOption ("maxNumIters", &commandLineOptions.maxNumIters,
                 "Maximum number of iterations that the solver is "
                 "allowed to take, over all restart cycles.  This has "
                 "nothing to do with the test passing.");
  clp.setOption ("restartLength", &commandLineOptions.restartLength,
                 "Maximum number of iterations per restart cycle.  "
                 "This corresponds to the standard Belos parameter "
                 "\"Num Blocks\".");
  clp.setOption ("preconditioner", &commandLineOptions.preconditioner, "Name of preconditioner ");
  clp.setOption ("relaxation", &commandLineOptions.relaxation, "Type of relaxation ");
  clp.setOption ("verbose", "quiet", &commandLineOptions.verbose,
                 "Whether to print verbose output");
  //
  clp.setOption ("stepSize", &commandLineOptions.stepSize,
                 "Step size; only applies to algorithms that take it.");
  clp.setOption ("useCholQR", "noCholQR", &commandLineOptions.useCholQR,
                 "Whether to use CholQR");
  clp.setOption ("useCholQR2", "noCholQR2", &commandLineOptions.useCholQR2,
                 "Whether to use CholQR2");
  clp.setOption ("delayHGeneration", "noDelayHGeneration", &commandLineOptions.delayHGeneration,
                 "Whether to delay H Generation");
  clp.setOption ("doReortho2", "noReortho2", &commandLineOptions.doReortho2,
                 "Whether to use one-synch reortho");
  clp.setOption ("computeRitzValues", "noRitzValues", &commandLineOptions.computeRitzValues,
                 "Whether to compute Ritz values");
  clp.setOption ("computeRitzValuesOnFly", "noRitzValuesOnFly", &commandLineOptions.computeRitzValuesOnFly,
                 "Whether to compute Ritz values on Fly");
  //
  clp.setOption ("matrixSize", &commandLineOptions.matrixSize,
                 "Global matrix size for the tridiagonal or diagonal matrix.");
  clp.setOption ("offDiagDiff", &commandLineOptions.offDiagDiff,
                 "Value of the term that makes the matrix nonsymmetric");
  clp.setOption ("diagAlpha", &commandLineOptions.diagAlpha,
                 "the first diagonal entry of the Simonici matrix");
  clp.setOption ("diagShift", &commandLineOptions.diagShift,
                 "the diagonal shift of the Simonici matrix");
  clp.setOption ("diagPower", &commandLineOptions.diagPower,
                 "the diagonal power of the Simonici matrix");
  clp.setOption ("useDiagMatrix", "notUseDiagMatrix", &commandLineOptions.useDiagMat,
                 "Whether to use diagonal matrix");
  clp.setOption ("nx2", &commandLineOptions.nx2, "Grid size for 2D Laplace");
  clp.setOption ("nx3", &commandLineOptions.nx3, "Grid size for 3D Laplace");
  clp.setOption ("matrixFilename", &commandLineOptions.matrixFilename, "Name of Matrix "
                 "Market file with the sparse matrix A");
  clp.setOption ("rhsFilename", &commandLineOptions.rhsFilename, "Name of Matrix "
                 "Market file with the right-hand side B");
  clp.setOption ("exaWindNumFiles", &commandLineOptions.exaWindNumFiles,
                 "Number of exaWind files");
  clp.setOption ("exaWindFilename", &commandLineOptions.exaWindFilename, "Name of ExaWind Matrix "
                 "Market file with the sparse matrix A");
  //
  clp.setOption ("useZoltan2", "noZoltan2", &commandLineOptions.useZoltan2,
                 "Whether to use Zoltan2");
  clp.setOption ("useParMETIS", "noParMETIS", &commandLineOptions.useParMETIS,
                 "Whether to use ParMETIS");
  clp.setOption ("useSuperLU", "noSuperLU", &commandLineOptions.useSuperLU,
                 "Whether to use SuperLU");
  //
  clp.setOption ("zeroInitialGuess", "nonzeroInitialGuess",
                 &commandLineOptions.zeroInitialGuess, "Whether to test "
                 "with a zero, or a nonzero, initial guess vector");
  clp.setOption ("randomRHS", "nonRandomRHS",
                 &commandLineOptions.randomRHS, "Whether to test "
                 "with a random or non random (b=A*ones) RHS vector");
  clp.setOption ("onesRHS", "nonOnesRHS",
                 &commandLineOptions.onesRHS, "Whether to test "
                 "with RHS vector with all ones");
  clp.setOption ("nrhs", &commandLineOptions.nrhs, "Number of right-hand-sides");
  clp.setOption ("blockSize", &commandLineOptions.blockSize, "Block size for block GMRES");
  //
  clp.setOption ("numRelaxationSweeps", &commandLineOptions.numRelaxationSweeps,
                 "Num relaxation sweeps");
  clp.setOption ("numInnerRelaxationSweeps", &commandLineOptions.numInnerRelaxationSweeps,
                 "Num inner relaxation sweeps");
  clp.setOption ("doBackwardSweep", "noBackwardSweep", &commandLineOptions.doBackwardSweep,
                 "Whether to do backward or forward sweeps");
  clp.setOption ("useInnerSptrsv", "noInnerSptrsv", &commandLineOptions.useInnerSptrsv,
                 "Whether to use Sptrsv or iteration");
  clp.setOption ("useSolutionBasedRec", "noSolutionBasedRec", &commandLineOptions.useSolutionBasedRec,
                 "Whether to use solution-based GS recurrence");
  clp.setOption ("clusterSize", &commandLineOptions.clusterSize,
                 "Size of cluster for clustered GS");
  //
  clp.setOption ("numRecycledBlocks", &commandLineOptions.numRecycledBlocks,
                 "Num Recycled Blocks for GCRODR");
  //
  clp.setOption ("polyDegrees", &commandLineOptions.polyDegrees,
                 "Num of degrees for GmresPoly");
}

#if 0
//
// The point is to have the matrix be diagonally dominant, but still
// nonsymmetric.
Teuchos::RCP<Tpetra::CrsMatrix<double> >
readExaWind (std::string matfile, int nfiles, Teuchos::FancyOStream& out)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using SC = double;
  using crs_matrix_type = Tpetra::CrsMatrix<SC>;
  //using coo_matrix_type = Tpetra::Details::CooMatrix<SC>;
  using Tpetra::Details::CooMatrix;
  using map_type = Tpetra::Map<>;
  using device_type = typename map_type::device_type;
  using LO = typename map_type::local_ordinal_type;
  using GO = typename map_type::global_ordinal_type;

  using STS = Teuchos::ScalarTraits<SC>;
  const SC ZERO = STS::zero ();

  int ret;
  GO ilower, iupper, jlower, jupper;
  GO irow, icol;
  SC value;

  auto comm = Tpetra::getDefaultComm ();
  const int commRank = comm->getRank ();
  const int commSize = comm->getSize ();

  // local cooMatrix
  CooMatrix<SC, LO, GO> cooMatrix;

  // each process reads a subset of files and insert nonzeros into local cooMatrix
  LO localNumNnzs = 0;
  for (LO ii=commRank; ii < nfiles; ii+=commSize) {
    FILE* fh;
    std::ostringstream suffix;
    suffix << matfile << "." << std::setw(5) << std::setfill('0') << ii;

    // open file
    if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
      throw std::runtime_error("Cannot open matrix file: " + suffix.str());
    } 
    else if (commRank == 0) {
      std::cout << commRank << ": reading:" << suffix.str() << std::endl;
    }
    // get first/last row and column indexes
    #if defined(Tpetra_INST_INT_LONG_LONG)
    //static_assert (std::is_same<GO, long long>::value, "Need to fix fscanf calls; use d or ld instead of lld");
    fscanf(fh, "%lld %lld %lld %lld\n", &ilower, &iupper, &jlower, &jupper);
    #elif defined(Tpetra_INST_INT_LONG)
    //static_assert (std::is_same<GO, long>::value, "Need to fix fscanf calls; use d or lld instead of ld");
    fscanf(fh, "%ld %ld %ld %ld\n", &ilower, &iupper, &jlower, &jupper);
    #else
    //static_assert (std::is_same<GO, int>::value, "Need to fix fscanf calls; use ld or lld instead of d");
    fscanf(fh, "%d %d %d %d\n", &ilower, &iupper, &jlower, &jupper);
    #endif

    // read the first entry
    #if defined(Tpetra_INST_INT_LONG_LONG)
    ret = fscanf(fh, "%lld %lld%*[ \t]%le\n", &irow, &icol, &value);
    #elif defined(Tpetra_INST_INT_LONG)
    ret = fscanf(fh, "%ld %ld%*[ \t]%le\n", &irow, &icol, &value);
    #else
    ret = fscanf(fh, "%d %d%*[ \t]%le\n", &irow, &icol, &value);
    #endif

    // push the entry to local cooMatrix, and read the next entry
    while (ret != EOF) {
      if (value != ZERO) {
        localNumNnzs ++;
        cooMatrix.sumIntoGlobalValue (irow, icol, value);
      }
      //std::cout << "Current entry: " << irow << ", " << icol << ", " << value << std::endl;
      #if defined(Tpetra_INST_INT_LONG_LONG)
      ret = fscanf(fh, "%lld %lld%*[ \t]%le\n", &irow, &icol, &value);
      #elif defined(Tpetra_INST_INT_LONG)
      ret = fscanf(fh, "%ld %ld%*[ \t]%le\n", &irow, &icol, &value);
      #else
      ret = fscanf(fh, "%d %d%*[ \t]%le\n", &irow, &icol, &value);
      #endif
    }
  }

  cooMatrix.fillComplete (comm);
  RCP<const map_type> origRowMap = cooMatrix.getMap ();

  // Convert cooMatrix to CrsMatrix.
  Kokkos::View<size_t*, device_type> rowOffsets;
  Kokkos::View<GO*, device_type> gblColInds;
  Kokkos::View<SC*, device_type> vals;
  cooMatrix.buildCrs(rowOffsets, gblColInds, vals);

  // Count number of nonzeros per row
  const LO origLclNumRows = origRowMap->getNodeNumElements ();
  //std::cout << "rowOffsets.size=" << rowOffsets.size() << " vs. " << origLclNumRows << std::endl;
  //Teuchos::ArrayRCP<size_t> rowCounts (origLclNumRows);
  Teuchos::ArrayView<size_t> rowCounts (origLclNumRows);
  for (LO lclRow = 0; lclRow < origLclNumRows; ++lclRow) {
    rowCounts[lclRow] = rowOffsets[lclRow+1] - rowOffsets[lclRow];
    //std::cout << "rowCounts[" << lclRow << "]=" << rowCounts[lclRow] << std::endl;
  }

  // Converting to global row indexes?
  auto origCrsMatrix = rcp (new crs_matrix_type (origRowMap, rowCounts, Tpetra::StaticProfile));
  for (LO lclRow = 0; lclRow < origLclNumRows; ++lclRow) {
    const GO gblRow = origRowMap->getGlobalElement (lclRow);
    const size_t beg = rowOffsets[lclRow];
    const size_t end = rowOffsets[lclRow+1];
    const LO rowCount = LO (end - beg);
    //std::cout << "In global row " << gblRow << ", I am inserting " << rowCount << " entries" << std::endl;
    origCrsMatrix->insertGlobalValues (gblRow, rowCount, vals.data () + beg, gblColInds.data () + beg);
  }
  origCrsMatrix->fillComplete ();

  // Is this Map a permutation of a contiguous integer range, plus possible overlap?
  RCP<const map_type> oneToOneRowMap = Tpetra::createOneToOne (origRowMap);
  const GO globalIndexDiff = oneToOneRowMap->getMaxAllGlobalIndex () - oneToOneRowMap->getMinAllGlobalIndex () + 1;
  if (globalIndexDiff != GO (oneToOneRowMap->getGlobalNumElements ())) {
    std::cout << oneToOneRowMap->getMaxAllGlobalIndex () << ", " << oneToOneRowMap->getMinAllGlobalIndex () << ", " << oneToOneRowMap->getGlobalNumElements () << std::endl;
    throw std::runtime_error ("Map is not a permutation of a contiguous integer range!");
  }

  // Redistribute the data to the desired output Map.
  RCP<const map_type> rowMap = rcp (new map_type (globalIndexDiff, oneToOneRowMap->getMinAllGlobalIndex (), 
                                                  comm, Tpetra::GloballyDistributed));
  auto crsMatrix_dist = rcp (new crs_matrix_type (rowMap, 0));
  Tpetra::Export<LO, GO> exporter (origRowMap, rowMap);
  crsMatrix_dist->doExport (*origCrsMatrix, exporter, Tpetra::ADD);  
  crsMatrix_dist->fillComplete ();

  //return origCrsMatrix;
  return crsMatrix_dist;
}
#endif

// Create a nonsymmetric tridiagonal matrix representing a
// discretization of a 1-D convection-diffusion operator.
// Stencil looks like this:
//
// [1/4 - offDiagDiff, 1, 1/4 + offDiagDiff]
//
// The point is to have the matrix be diagonally dominant, but still
// nonsymmetric.
template<class SC>
Teuchos::RCP<Tpetra::CrsMatrix<SC> >
createNonsymmTridiagMatrix (const Teuchos::RCP<const Tpetra::Map<> >& rowMap,
                            const SC offDiagDiff)
{
  using Teuchos::rcp;
  using crs_matrix_type = Tpetra::CrsMatrix<SC>;
  using map_type = Tpetra::Map<>;
  using LO = typename map_type::local_ordinal_type;
  using GO = typename map_type::global_ordinal_type;
  using STS = Teuchos::ScalarTraits<SC>;

  const LO lclNumRows = rowMap.is_null () ? LO (0) :
    LO (rowMap->getNodeNumElements ());
  const GO gblMinGblInd = rowMap->getMinAllGlobalIndex ();
  const GO gblMaxGblInd = rowMap->getMaxAllGlobalIndex ();
  auto A = rcp (new crs_matrix_type (rowMap, 3, Tpetra::StaticProfile));

  const SC ONE = STS::one ();
  const SC TWO = ONE + ONE;
  const SC FOUR = TWO + TWO;
  const SC baseOffDiagEnt = ONE / FOUR;
  const SC subDiagEnt = baseOffDiagEnt - offDiagDiff;
  const SC diagEnt = ONE;
  const SC superDiagEnt = baseOffDiagEnt + offDiagDiff;

  Teuchos::Array<GO> gblColIndsBuf (3);
  Teuchos::Array<SC> valsBuf (3);
  for (LO lclRow = 0; lclRow < lclNumRows; ++lclRow) {
    const GO gblRow = rowMap->getGlobalElement (lclRow);
    const GO gblCol = gblRow;
    LO numEnt = 0; // to be set below
    if (gblRow == gblMinGblInd && gblRow == gblMaxGblInd) {
      numEnt = 1;
      valsBuf[0] = diagEnt;
      gblColIndsBuf[0] = gblCol;
    }
    else if (gblRow == gblMinGblInd) {
      numEnt = 2;
      valsBuf[0] = diagEnt;
      valsBuf[1] = superDiagEnt;
      gblColIndsBuf[0] = gblCol;
      gblColIndsBuf[1] = gblCol + GO (1);
    }
    else if (gblRow == gblMaxGblInd) {
      numEnt = 2;
      valsBuf[0] = subDiagEnt;
      valsBuf[1] = diagEnt;
      gblColIndsBuf[0] = gblCol - GO (1);
      gblColIndsBuf[1] = gblCol;
    }
    else {
      numEnt = 3;
      valsBuf[0] = subDiagEnt;
      valsBuf[1] = diagEnt;
      valsBuf[2] = superDiagEnt;
      gblColIndsBuf[0] = gblCol - GO (1);
      gblColIndsBuf[1] = gblCol;
      gblColIndsBuf[2] = gblCol + GO (1);
    }
    Teuchos::ArrayView<GO> gblColInds = gblColIndsBuf.view (0, numEnt);
    Teuchos::ArrayView<SC> vals = valsBuf.view (0, numEnt);
    A->insertGlobalValues (gblRow, gblColInds, vals);
  }
  A->fillComplete ();
  return A;
}

//
// Simonici diagonal matrix 
Teuchos::RCP<Tpetra::CrsMatrix<double> >
createDiagMatrix (const double diagAlpha, const double diagShift, const double diagPower, const Teuchos::RCP<const Tpetra::Map<> >& rowMap)
{
  using Teuchos::rcp;
  using SC = double;
  using crs_matrix_type = Tpetra::CrsMatrix<SC>;
  using map_type = Tpetra::Map<>;
  using LO = typename map_type::local_ordinal_type;
  using GO = typename map_type::global_ordinal_type;

  const LO lclNumRows = rowMap.is_null () ? LO (0) :
    LO (rowMap->getNodeNumElements ());
  auto A = rcp (new crs_matrix_type (rowMap, 1, Tpetra::StaticProfile));

  Teuchos::Array<GO> gblColIndsBuf (1);
  Teuchos::Array<SC> valsBuf (1);
  for (LO lclRow = 0; lclRow < lclNumRows; ++lclRow) {
    const GO gblRow = rowMap->getGlobalElement (lclRow);
    const GO gblCol = gblRow;
    LO numEnt = 0; // to be set below
    if (gblRow == 0) {
      numEnt = 1;
      valsBuf[0] = diagAlpha + diagShift;
      gblColIndsBuf[0] = gblCol;
    }
    else {
      numEnt = 1;
      valsBuf[0] = gblCol + diagShift;
      gblColIndsBuf[0] = gblCol;
    }
    valsBuf[0] = pow(valsBuf[0], diagPower);
    Teuchos::ArrayView<GO> gblColInds = gblColIndsBuf.view (0, numEnt);
    Teuchos::ArrayView<SC> vals = valsBuf.view (0, numEnt);
    A->insertGlobalValues (gblRow, gblColInds, vals);
  }
  A->fillComplete ();
  return A;
}

void
testSolver (Teuchos::FancyOStream& out,
            bool& success,
            const std::string& solverName,
            const int maxAllowedNumIters,
            const bool verbose)
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

  typedef Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<SC>> reader_type;
  typedef Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<SC>> writer_type;
  typedef Teuchos::TimeMonitor time_monitor_type;
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

  // convert solverName to upper-case
  std::string SolverName = "";
  for (size_t i = 0; i < solverName.length (); i++){
    char upper_case = std::toupper(solverName[i]);
    SolverName.insert (i, 1, upper_case);
  }

  auto comm = Tpetra::getDefaultComm ();
  const int myRank = comm->getRank ();
  if (myRank == 0) {
    if (solverName.find("TPETRA") != std::string::npos) {
      myOut << "Test \"native\" Tpetra version of solver \"" << solverName << "\""
            << endl;
    } else {
      myOut << "Test Belos version of solver \"" << solverName << "\""
            << endl;
    }
    myOut << "  preconditioner: " << commandLineOptions.preconditioner << endl;
    myOut << "  orthoType     : " << commandLineOptions.orthoType << endl;
    myOut << (STS::isComplex ? "> in complex type" : "> in real type") << endl;
    myOut << "  LO type       : " << LOTS::name()  << endl;
    myOut << "  GO type       : " << GOTS::name()  << endl;
  }
  Teuchos::OSTab tab1 (out);

  if (myRank == 0) {
    myOut << "Create the linear system to solve" << endl;
  }

  // set up matrix A
  Teuchos::RCP<Tpetra::CrsMatrix<SC>> A;
  RCP< Teuchos::Time > ioTimer = time_monitor_type::getNewCounter ("Matrix Create");
  {
    time_monitor_type LocalTimer (*ioTimer);
    if (commandLineOptions.matrixFilename != "") {
      // read matrix from a file
      if (myRank == 0)
        myOut << "matrix read from " << commandLineOptions.matrixFilename << endl;
      A = reader_type::readSparseFile (commandLineOptions.matrixFilename, comm);
    }
#if 0
    else if (commandLineOptions.exaWindFilename != "") {
      if (myRank == 0)
        myOut << "exaWind matrix read from " << commandLineOptions.exaWindFilename << endl;
      A = readExaWind (commandLineOptions.exaWindFilename, commandLineOptions.exaWindNumFiles, out);
    }
#endif
    else if (commandLineOptions.nx3 > 0) {
      Teuchos::ParameterList galeriList;
      GO nx = commandLineOptions.nx3;
      Tpetra::global_size_t nGlobalElements = nx * nx * nx;
      galeriList.set("nx", nx);
      galeriList.set("ny", nx);
      galeriList.set("nz", nx);

      RCP<const map_type> map = rcp(new map_type(nGlobalElements, 0, comm));

#if defined(HAVE_GALERI)
      typedef Galeri::Xpetra::Problem<map_type, Tpetra::CrsMatrix<SC>, MV> Galeri_t;
      RCP<Galeri_t> galeriProblem =
                    Galeri::Xpetra::BuildProblem<SC, LO, GO,
                                        map_type, Tpetra::CrsMatrix<SC>, MV>
                                       ("Laplace3D", map, galeriList);
      A = galeriProblem->BuildMatrix();
#endif
      if (myRank == 0) {
        std::cout << "3D Laplace matrix with nx=" << commandLineOptions.nx3 << endl;
      }
    }
    else if (commandLineOptions.nx2 > 0) {
      Teuchos::ParameterList galeriList;
      GO nx = commandLineOptions.nx2;
      Tpetra::global_size_t nGlobalElements = nx * nx;
      galeriList.set("nx", nx);
      galeriList.set("ny", nx);

      RCP<const map_type> map = rcp(new map_type(nGlobalElements, 0, comm));

#if defined(HAVE_GALERI)
      typedef Galeri::Xpetra::Problem<map_type, Tpetra::CrsMatrix<SC>, MV> Galeri_t;
      RCP<Galeri_t> galeriProblem =
                    Galeri::Xpetra::BuildProblem<SC, LO, GO,
                                        map_type, Tpetra::CrsMatrix<SC>, MV>
                                       ("Laplace2D", map, galeriList);
      A = galeriProblem->BuildMatrix();
#endif
      if (myRank == 0) {
        std::cout << "2D Laplace matrix with nx=" << commandLineOptions.nx2 << endl;
      }
    }
    else if (commandLineOptions.useDiagMat) {
      const GO gblNumRows = (commandLineOptions.matrixSize > 0 ? commandLineOptions.matrixSize : 100);
      const GO indexBase = 0;
      RCP<const map_type> map (new map_type (gblNumRows, indexBase, comm));
      A = createDiagMatrix (commandLineOptions.diagAlpha, commandLineOptions.diagShift, commandLineOptions.diagPower, map);
    }
    else {
      const GO gblNumRows = (commandLineOptions.matrixSize > 0 ? commandLineOptions.matrixSize : 10000);
      const GO indexBase = 0;
      RCP<const map_type> map (new map_type (gblNumRows, indexBase, comm));
      A = createNonsymmTridiagMatrix (map, commandLineOptions.offDiagDiff);
    }
  }
  A->describe (out);

  // Read RHS from a file
  Teuchos::RCP<MV> B; // (A->getRangeMap (), 1);
  if (commandLineOptions.rhsFilename != "") {
    if (myRank == 0)
      myOut << "RHS read from " << commandLineOptions.rhsFilename << endl;
    RCP<const map_type> rhsMap = A->getRangeMap ();
    B = reader_type::readDenseFile (
            commandLineOptions.rhsFilename, comm, rhsMap, false, false);
    commandLineOptions.nrhs = B->getNumVectors ();
  } else {
    //B = rcp (new MV(A->getRangeMap (), commandLineOptions.nrhs));
    B = rcp (new MV(A->getDomainMap (), commandLineOptions.nrhs));
  }

  // matrix partition
  MPI_Barrier(MPI_COMM_WORLD);
  RCP< Teuchos::Time > partTimer = time_monitor_type::getNewCounter ("Matrix Partition");
  if (commandLineOptions.useZoltan2 || commandLineOptions.useParMETIS) {
    time_monitor_type LocalTimer (*partTimer);
    if (myRank == 0) {
      myOut << " Use Zoltan2/ParMETIS for redistributing matrix" << endl;
    }
    // Create an input adapter for the Tpetra matrix.
    Zoltan2::XpetraCrsMatrixAdapter<Tpetra::CrsMatrix<SC, LO, GO>>
      zoltan_matrix(A);

    // Specify partitioning parameters
    Teuchos::ParameterList zoltan_params;
    zoltan_params.set("partitioning_approach", "partition");
    //
    if (commandLineOptions.useParMETIS) {
      zoltan_params.set("algorithm", "parmetis");
    }
    //
    zoltan_params.set("symmetrize_input", "transpose");
    zoltan_params.set("partitioning_objective", "minimize_cut_edge_weight");
    if (myRank == 0) {
      myOut << zoltan_params.currentParametersString() << endl;
    }

    // Create and solve partitioning problem
    Zoltan2::PartitioningProblem<Zoltan2::XpetraCrsMatrixAdapter<Tpetra::CrsMatrix<SC, LO, GO>>> 
      problem(&zoltan_matrix, &zoltan_params);
    problem.solve();

    // Redistribute matrix
    RCP<Tpetra::CrsMatrix<SC, LO, GO>> zoltan_A_;
    zoltan_matrix.applyPartitioningSolution (*A, zoltan_A_, problem.getSolution());
    // Set it as coefficient matrix
    A = zoltan_A_;

    // Redistribute RHS
    RCP<MV> zoltan_B_;
    Zoltan2::XpetraMultiVectorAdapter<Tpetra::MultiVector<>> adapterVector(rcpFromRef (*B));
    adapterVector.applyPartitioningSolution (*B, zoltan_B_, problem.getSolution());
    // Set it as RHS
    B = zoltan_B_;
  }

printf( " myRank = %d\n",myRank );
  // Set up RHS (if not read from file)
  MV X_initial (A->getDomainMap (), commandLineOptions.nrhs);
  if (commandLineOptions.rhsFilename == "") {
    if (commandLineOptions.onesRHS) {
      B->putScalar (ONE);
      if (myRank == 0) {
        myOut << "rhs=ones" << endl;
      }
printf( " %d: rhs = one\n",myRank );
    }
    else if (commandLineOptions.randomRHS) {
      B->randomize ();
      if (myRank == 0) {
        myOut << "rhs=random" << endl;
      }
printf( " %d: rhs = randomm\n",myRank );
    }
    else {
      X_initial.putScalar (ONE);
      A->apply (X_initial, *B);
      if (myRank == 0) {
        myOut << "rhs=A*ones" << endl;
      }
printf( " %d: rhs = A*ones\n",myRank );
    }
  }

  // Set up initial guess
  if (commandLineOptions.zeroInitialGuess) {
    X_initial.putScalar (ZERO); // (re)set initial guess to zero
  }
  else {
    X_initial.putScalar (ONE); // just something nonzero, to test
  }
  MV X (X_initial, Teuchos::Copy);

#if 0
  {
    char filename[200];
    sprintf(filename,"A%d_%d.dat",comm->getSize (), myRank);
    writer_type::writeSparseFile (filename, A);
  }
#endif

  if (myRank == 0) {
    myOut << "Set up the linear system to solve" << endl;
  }
  auto lp = rcp (new Belos::LinearProblem<SC, MV, OP> (A, rcpFromRef (X), B));
  lp->setProblem ();

  // set up preconditioner
  MPI_Barrier(MPI_COMM_WORLD);
  RCP< Teuchos::Time > Ifpack2InitTimer = time_monitor_type::getNewCounter ("Ifpack2: initialize");
  RCP< Teuchos::Time > Ifpack2CompTimer = time_monitor_type::getNewCounter ("Ifpack2: compute");
  if (commandLineOptions.preconditioner != "") {
    Teuchos::RCP<Ifpack2::Preconditioner<SC, LO, GO>> M
      = Ifpack2::Factory::create (commandLineOptions.preconditioner, 
                                  Teuchos::rcp_const_cast<const Tpetra::CrsMatrix<SC>>(A), 0);

    RCP<ParameterList> PrecParams = parameterList ("Preconditioner");
    if (commandLineOptions.preconditioner == "RELAXATION") {
      if (commandLineOptions.relaxation == "Richardson") {
        PrecParams->set ("relaxation: type", "Richardson");
      }
      //
      else if (commandLineOptions.relaxation == "GS MT") {
        PrecParams->set ("relaxation: type", "MT Gauss-Seidel");
      }
      else if (commandLineOptions.relaxation == "GS CL") {
        PrecParams->set ("relaxation: type", "MT Gauss-Seidel");
        PrecParams->set ("relaxation: mtgs cluster size", commandLineOptions.clusterSize);
      }
      else if (commandLineOptions.relaxation == "GS") {
        PrecParams->set ("relaxation: type", "Gauss-Seidel");
      }
      else if (commandLineOptions.relaxation == "GS2") {
        PrecParams->set ("relaxation: type", "Two-stage Gauss-Seidel");
        PrecParams->set ("relaxation: inner sweeps", commandLineOptions.numInnerRelaxationSweeps);
        PrecParams->set ("relaxation: inner sparse-triangular solve", commandLineOptions.useInnerSptrsv);
        PrecParams->set ("relaxation: solution based recurrence", commandLineOptions.useSolutionBasedRec);
        PrecParams->set ("relaxation: backward mode", commandLineOptions.doBackwardSweep);
      }
      //
      else if (commandLineOptions.relaxation == "SGS MT") {
        PrecParams->set ("relaxation: type", "MT Symmetric Gauss-Seidel");
      }
      else if (commandLineOptions.relaxation == "SGS CL") {
        PrecParams->set ("relaxation: type", "MT Symmetric Gauss-Seidel");
        PrecParams->set ("relaxation: mtgs cluster size", commandLineOptions.clusterSize);
      }
      else if (commandLineOptions.relaxation == "SGS2") {
        PrecParams->set ("relaxation: type", "Two-stage Symmetric Gauss-Seidel");
        PrecParams->set ("relaxation: inner sweeps", commandLineOptions.numInnerRelaxationSweeps);
        PrecParams->set ("relaxation: inner sparse-triangular solve", commandLineOptions.useInnerSptrsv);
        PrecParams->set ("relaxation: solution based recurrence", commandLineOptions.useSolutionBasedRec);
      }
      else if (commandLineOptions.relaxation == "SGS") {
        PrecParams->set ("relaxation: type", "Symmetric Gauss-Seidel");
      } else {
        std::cout << " Invalide relaxation type " << std::endl;
        exit(0);
      }
      PrecParams->set ("relaxation: sweeps", commandLineOptions.numRelaxationSweeps);
    }
    if (commandLineOptions.preconditioner == "SCHWARZ") {
      int overlap = 0;
      PrecParams->set ("schwarz: overlap level", overlap);

      bool direct_subdomain = true;
      if (direct_subdomain) {
        PrecParams->set ("relaxation: sweeps", commandLineOptions.numRelaxationSweeps);

        PrecParams->set ("subdomain solver name", "AMESOS2");
        if (commandLineOptions.useSuperLU) {
          ParameterList &SubdomainParams = PrecParams->sublist("subdomain solver parameters");
          SubdomainParams.set ("Amesos2 solver name", "Superlu");

          ParameterList &Amesos2Params = SubdomainParams.sublist("Amesos2");
          ParameterList &SuperluParams = Amesos2Params.sublist("SuperLU");
          //SuperLUParams->set ("Equil", true);
        }
      }
      else {
        PrecParams->set ("subdomain solver name",  "RELAXATION");
        PrecParams->set ("relaxation: type",       "Symmetric Gauss-Seidel");
        PrecParams->set ("relaxation: sweeps",     commandLineOptions.numRelaxationSweeps);
      }
    }
    M->setParameters (*PrecParams);
    {
      time_monitor_type LocalTimer (*Ifpack2InitTimer);
      M->initialize ();
    }
    {
      time_monitor_type LocalTimer (*Ifpack2CompTimer);
      M->compute ();
    }
    if (myRank == 0) {
      myOut << "Preconditioner:" << endl;
      myOut << PrecParams->currentParametersString() << endl;
    }

    lp->setRightPrec(M);
  }

  // Set parameters
  if (myRank == 0) {
    myOut << "Set parameters" << endl;
  }
  RCP<ParameterList> params = parameterList ("Belos");

  int frequency = 1;
  int verbosity = Belos::Errors + Belos::Warnings;
  if (SolverName.find("TPETRA") != std::string::npos) {
    params->set ("Verbosity", verbose ? 1 : 0);
  }
  else if (commandLineOptions.verbose) {
    params->set ("Output Frequency", frequency);
    verbosity += Belos::StatusTestDetails;
    verbosity += Belos::IterationDetails + Belos::FinalSummary;
    params->set ("Verbosity", verbosity);
  }
  //
  if (SolverName != "BICGSTAB" && SolverName != "LSQR") {
    params->set ("Orthogonalization",  commandLineOptions.orthoType);
    if (commandLineOptions.maxOrthoSteps > 0) {
      params->set ("Max Orthogonalization Passes", commandLineOptions.maxOrthoSteps);
    }
    params->set ("Num Blocks", commandLineOptions.restartLength);
    int restartLength = commandLineOptions.restartLength;
    int maxRestarts = (commandLineOptions.maxNumIters+restartLength-1)/restartLength;
    params->set ("Maximum Restarts", maxRestarts);
  }
  //
  params->set ("Convergence Tolerance",  commandLineOptions.tol );
  params->set ("Maximum Iterations", commandLineOptions.maxNumIters);

  if (SolverName == "TPETRA GMRES" ||
      SolverName == "TPETRA GMRES S-STEP" ||
      SolverName == "TPETRA GMRES SINGLE REDUCE" ||
      SolverName == "TPETRA BLOCK GMRES S-STEP") {
    params->set ("Step Size", commandLineOptions.stepSize);
    if (myRank == 0) {
      if (commandLineOptions.computeRitzValues) {
        myOut << " computeRitzValues\n" << endl;
      } else {
        myOut << " noRitzValues\n" << endl;
      }
    }
    params->set ("Reorthogonalize Blocks", commandLineOptions.reortho);
    params->set ("Compute Ritz Values", commandLineOptions.computeRitzValues);
    params->set ("Compute Ritz Values on Fly", commandLineOptions.computeRitzValuesOnFly);
    params->set ("CholeskyQR",  commandLineOptions.useCholQR);
    params->set ("CholeskyQR2", commandLineOptions.useCholQR2);
    params->set ("Delay H generation", commandLineOptions.delayHGeneration);
    params->set ("Perform low-synch reortho", commandLineOptions.doReortho2);
  } else if (SolverName == "TPETRA GMRES PIPELINE") {
    params->set ("Compute Ritz Values", commandLineOptions.computeRitzValues);
  }
  if (SolverName == "GCRODR") {
    params->set ("Num Recycled Blocks", commandLineOptions.numRecycledBlocks);
  }
  if (SolverName == "BLOCK GMRES") {
    if (commandLineOptions.blockSize > 0) {
      params->set ("Block Size", commandLineOptions.blockSize);
    } else {
      params->set ("Block Size", commandLineOptions.nrhs);
    }
  }

  if (myRank == 0) {
    myOut << "Create solver instance using Belos::SolverFactory" << endl;
    myOut << params->currentParametersString() << endl;
  }
  // gmres calls BelosPseudoBlockGmresSolMgr that calls BelosPseudoBlockGmresIter
  RCP<Belos::SolverManager<SC, MV, OP> > solver;
  if (SolverName == "GMRESPOLY") {
    ParameterList polyList;
    if (commandLineOptions.verbose) {
      polyList.set ("Output Frequency", frequency);
      polyList.set ("Verbosity", verbosity);          // Verbosity for GmresPoly
    }
    int maxdegree = commandLineOptions.polyDegrees;
    polyList.set ("Maximum Degree", maxdegree );      // Maximum degree of the GMRES polynomial
    polyList.set ("Outer Solver", "Gmres");           // Solver that uses the poly prec
    polyList.set ("Outer Solver Params", *params);    // Parameters for the outer solver

    solver = rcp (new Belos::GmresPolySolMgr<SC, MV, OP>(lp, rcp(&polyList,false)));
  }
  else {
    try {
      Belos::SolverFactory<SC, MV, OP> factory;
      //auto names = factory.supportedSolverNames();
      //if (myRank == 0) myOut << names << endl;
      solver = factory.create (solverName, Teuchos::null);
    }
    catch (std::exception& e) {
      myOut << "*** FAILED: Belos::SolverFactory::create threw an exception: "
          << e.what () << endl;
      success = false;
      return;
    }
    TEST_ASSERT( solver.get () != nullptr );
    if (solver.get () == nullptr) {
      myOut << "Belos::SolverFactory returned a null solver." << endl;
      return;
    }

    try {
      solver->setParameters (params);
    }
    catch (std::exception& e) {
      myOut << "*** FAILED: setParameters threw an exception: "
            << e.what () << endl;
      success = false;
      return;
    }
    catch (...) {
      myOut << "*** FAILED: setParameters threw an exception "
        "not a subclass of std::exception." << endl;
      success = false;
      return;
    }
    solver->setProblem (lp);
  }

  // compute initial residual (mainly to wramup SpMV)
  MV R_initial (B->getMap (), B->getNumVectors ());
  A->apply (X, R_initial);
  R_initial.update (ONE, *B, -ONE);
  Teuchos::Array<mag_type> R_initial_norms (R_initial.getNumVectors ());
  R_initial.norm2 (R_initial_norms ());

  // !! solve !!
  if (myRank == 0) {
    myOut << "Solve the linear system" << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  RCP< Teuchos::Time > BelosSolverTimer = time_monitor_type::getNewCounter ("Belos  : solve");
  Belos::ReturnType belosResult;
  {
    time_monitor_type LocalTimer (*BelosSolverTimer);
    belosResult = solver->solve ();
  }

  if (myRank == 0) {
    myOut << "Belos solver wrapper result: "
          << (belosResult == Belos::Converged ? "Converged" : "Unconverged")
          << endl
          << "Number of iterations: " << solver->getNumIters ()
          << endl;
    myOut << "DoF                 : " << X.getGlobalLength() << ", "
                                      << X.getLocalLength() << endl;
  }
  //TEST_ASSERT( belosResult == Belos::Converged );

  if (myRank == 0) {
    myOut << "Check the explicit residual norm(s)" << endl;
  }

  // Get the tolerance that the solver actually used.
  const mag_type tol = [&] () {
      const char tolParamName[] = "Convergence Tolerance";
      auto pl = solver->getCurrentParameters ();
      if (! pl->isType<mag_type> (tolParamName)) {
        pl = solver->getValidParameters ();
      }
      if (SolverName == "GMRESPOLY") {
        return commandLineOptions.tol;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION
          (! pl->isType<mag_type> (tolParamName), std::logic_error,
         "Solver lacks \"" << tolParamName << "\" parameter, in either "
         "getCurrentParameters() or getValidParameters().");
        return pl->get<mag_type> (tolParamName);
      }
    } ();

  //MV R_initial (B->getMap (), B->getNumVectors ());
  //A->apply (X_initial, R_initial);
  //R_initial.update (ONE, *B, -ONE);
  //Teuchos::Array<mag_type> R_initial_norms (R_initial.getNumVectors ());
  //R_initial.norm2 (R_initial_norms ());

  MV R_final (B->getMap (), B->getNumVectors ());
  A->apply (X, R_final);
  R_final.update (ONE, *B, -ONE);
  Teuchos::Array<mag_type> R_final_norms (R_final.getNumVectors ());
  R_final.norm2 (R_final_norms ());

  Teuchos::Array<mag_type> B_norms (B->getNumVectors ());
  Teuchos::Array<mag_type> X_initial_norms (X_initial.getNumVectors ());
  B->norm2 (B_norms ());
  X_initial.norm2 (X_initial_norms ());

  for (size_t j = 0; j < R_final.getNumVectors (); ++j) {
    const mag_type relResNorm = (R_initial_norms[j] == STM::zero ()) ?
      R_final_norms[j] :
      R_final_norms[j] / R_initial_norms[j];
    if (myRank == 0) {
      myOut << "Column " << (j+1) << " of " << R_final.getNumVectors ()
            << endl;
      myOut << " >> Right-hand-side norm: " << B_norms[j] 
            << ": Initial solution norm: "  << X_initial_norms[j]
            << endl;
      myOut << " >> Absolute residual norm: " << R_final_norms[j]
            << ", Relative residual norm: " << relResNorm
            << ", Tolerance: " << tol
            << endl;
      if (relResNorm > tol) {
        myOut << " *** explicit residual norm check failed *** " << endl;
      }
    }
    //TEST_ASSERT( relResNorm <= tol );
  }
  if (myRank == 0) myOut << endl;

  //TEST_ASSERT( solver->getNumIters () <= maxAllowedNumIters );

  time_monitor_type::summarize();
  time_monitor_type::zeroOutTimers ();
}

TEUCHOS_UNIT_TEST( TpetraNativeSolvers, Diagonal )
{
  testSolver (out, success, commandLineOptions.solverName,
              commandLineOptions.maxAllowedNumIters,
              commandLineOptions.verbose);
}

} // namespace (anonymous)

namespace BelosTpetra {
namespace Impl {
  // extern void register_Cg (const bool verbose);
  // extern void register_CgPipeline (const bool verbose);
  // extern void register_CgSingleReduce (const bool verbose);
  extern void register_Gmres (const bool verbose);
  extern void register_GmresPipeline (const bool verbose);
  extern void register_GmresS (const bool verbose);
  extern void register_GmresSingleReduce (const bool verbose);
  extern void register_GmresSstep (const bool verbose);
  extern void register_BlockGmresSstep (const bool verbose);
} // namespace Impl
} // namespace BelosTpetra

int main (int argc, char* argv[])
{
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);

  constexpr bool verbose = false;
  // BelosTpetra::Impl::register_Cg (verbose);
  // BelosTpetra::Impl::register_CgPipeline (verbose);
  // BelosTpetra::Impl::register_CgSingleReduce (verbose);
  BelosTpetra::Impl::register_Gmres (verbose);
  BelosTpetra::Impl::register_GmresPipeline (verbose);
  BelosTpetra::Impl::register_GmresS (verbose);
  BelosTpetra::Impl::register_GmresSingleReduce (verbose);
  BelosTpetra::Impl::register_GmresSstep (verbose);
  //BelosTpetra::Impl::register_BlockGmresSstep (verbose);

  // warmup?
  Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
}
