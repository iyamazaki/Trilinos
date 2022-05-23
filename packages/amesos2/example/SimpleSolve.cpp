// @HEADER
//
// ***********************************************************************
//
//           Amesos2: Templated Direct Sparse Solver Package
//                  Copyright 2011 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

/**
   \file   SimpleSolve.cpp
   \author Eric Bavier <etbavie@sandia.gov>
   \date   Sat Jul 17 10:35:39 2010

   \brief  Simple example of Amesos2 usage.

   This example solves a simple sparse system of linear equations using the
   Amesos2 interface to the Superlu solver.
*/

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include "Amesos2.hpp"
#include "Amesos2_Version.hpp"


int main(int argc, char *argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc,&argv);

#if 0
  typedef std::complex<double> SC;
#else
  typedef Tpetra::CrsMatrix<>::scalar_type SC;
#endif
  typedef Tpetra::Map<>::local_ordinal_type LO;
  typedef Tpetra::Map<>::global_ordinal_type GO;
  typedef Teuchos::ScalarTraits<SC> STS;
  typedef typename STS::magnitudeType RL;

  typedef Tpetra::CrsMatrix<SC,LO,GO> MAT;
  typedef Tpetra::MultiVector<SC,LO,GO> MV;

  using Tpetra::global_size_t;
  using Teuchos::tuple;
  using Teuchos::RCP;
  using Teuchos::rcp;

  bool SuperNodal = false;   // for Cholmod
  bool getDiagonals = false; // for Cholmod
  std::string matrixFilename = "";
  std::string solverType = "SuperLU";
  Teuchos::CommandLineProcessor cmdp(false, false);
  cmdp.setOption("matrixFilename",                 &matrixFilename, "Matrix file name");
  cmdp.setOption("solverType",                     &solverType,     "Amesos2 solver type");
  cmdp.setOption("superNodal",     "noSuperNodal", &SuperNodal,     "Cholmod SuperNodal type");
  cmdp.setOption("getDiagonals",   "noDiagonals",  &getDiagonals,   "get diagonal entries of factors");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }
  solverType = Amesos2::tolower (solverType);

  std::ostream &out = std::cout;
  out << std::endl;
  Teuchos::RCP<const Teuchos::Comm<int> > comm =
    Tpetra::getDefaultComm();

  const SC ONE = STS::one ();
  const size_t numVectors = 1;
  size_t myRank = comm->getRank();

  RCP<MAT> A;
  // Construct matrix
  if (matrixFilename != "") {
     out << " reading : " << matrixFilename << std::endl;
     typedef Tpetra::MatrixMarket::Reader<MAT> reader_type;
     A = reader_type::readSparseFile (matrixFilename, comm);
  } else {
    /*
     * We will solve a system with a known solution, for which we will be using
     * the following matrix:
     *
     * [ [ 7,  0,  -3, 0,  -1, 0 ]
     *   [ 2,  8,  0,  0,  0,  0 ]
     *   [ 0,  0,  1,  0,  0,  0 ]
     *   [ -3, 0,  0,  5,  0,  0 ]
     *   [ 0,  -1, 0,  0,  4,  0 ]
     *   [ 0,  0,  0,  -2, 0,  6 ] ]
     *
     */
    // create a Map
    global_size_t nrows = 6;
    RCP<Tpetra::Map<LO,GO> > map
      = rcp( new Tpetra::Map<LO,GO>(nrows,0,comm) );
    A = rcp( new MAT(map,3) ); // max of three entries in a row
    if( myRank == 0 ){
      A->insertGlobalValues(0,tuple<GO>(0,2,4),tuple<SC>(7,-3,-1));
      A->insertGlobalValues(1,tuple<GO>(0,1),tuple<SC>(2,8));
      A->insertGlobalValues(2,tuple<GO>(2),tuple<SC>(1));
      A->insertGlobalValues(3,tuple<GO>(0,3),tuple<SC>(-3,5));
      A->insertGlobalValues(4,tuple<GO>(1,4),tuple<SC>(-1,4));
      A->insertGlobalValues(5,tuple<GO>(3,5),tuple<SC>(-2,6));
    }
    A->fillComplete();
  }

  // Create random X
  auto rowmap = A->getRangeMap ();
  RCP<MV> X = rcp(new MV(rowmap,numVectors));
  X->randomize();

  /* Create B
   */
  RCP<MV> B = rcp(new MV(rowmap,numVectors));
  A->apply (*X, *B);

  /* Create solver interface to Superlu with Amesos2 factory method
   */ 
  out << " " << Amesos2::version() << std::endl;
  out << " Amesos2 solver type : " << solverType << std::endl;
  // Before we do anything, check that solverType is enabled
  if( !Amesos2::query(solverType) ){
    std::cerr << solverType << " not enabled.  Exiting..." << std::endl;
    return EXIT_SUCCESS;        // Otherwise CTest will pick it up as
                                // failure, which it isn't really
  }
  RCP<MV> Xhat = rcp(new MV(rowmap,numVectors));
  RCP<Amesos2::Solver<MAT,MV> > solver = Amesos2::create<MAT,MV>(solverType, A, Xhat, B);

  Teuchos::ParameterList amesos2_params("Amesos2");
  if (solverType == "cholmod") {
    Teuchos::ParameterList &solver_params = amesos2_params.sublist ("Cholmod");
    solver_params.set ("SuperNodal", SuperNodal);
  }
  solver->setParameters (rcpFromRef(amesos2_params));

  /* Solve
   */  
  solver->symbolicFactorization().numericFactorization().solve();


  /* Compute Residual
   */
  RCP<MV> R = rcp(new MV(rowmap, numVectors));
  RCP<MV> E = rcp(new MV(rowmap, numVectors));
  A->apply (*Xhat, *R);
  R->update (ONE, *B, -ONE);

  /* Compute Error
   */
  Tpetra::deep_copy(*E, *X);
  E->update (ONE, *Xhat, -ONE);

  Teuchos::Array<RL> Rnorm (numVectors);
  Teuchos::Array<RL> Enorm (numVectors);
  Teuchos::Array<RL> Bnorm (numVectors);
  Teuchos::Array<RL> Xnorm (numVectors);
  R->norm2 (Rnorm ());
  E->norm2 (Enorm ());
  B->norm2 (Bnorm ());
  X->norm2 (Xnorm ());
  if( myRank == 0 ){
    out << std::endl;
    out << " Exact solution norm : " << std::endl;
    for (size_t j = 0; j < numVectors; j++) {
      out << " > " << Xnorm[j] << std::endl;
    }
    out << std::endl;
    out << " Right-hand-side norm : " << std::endl;
    for (size_t j = 0; j < numVectors; j++) {
      out << " > " << Bnorm[j] << std::endl;
    }
    out << std::endl;
    out << " Error norm : " << std::endl;
    for (size_t j = 0; j < numVectors; j++) {
      out << " > " << Enorm[j] << std::endl;
    }
    out << std::endl;
    out << " Residual norm : " << std::endl;
    for (size_t j = 0; j < numVectors; j++) {
      out << " > " << Rnorm[j] << std::endl;
    }
    out << std::endl;
  }

  if (getDiagonals) {
    RCP<MV> D = rcp(new MV(rowmap,1));
    solver->getDiagonals(D);

    RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
    D->describe(*fos,Teuchos::VERB_EXTREME);
  }
  // We are done.
  return 0;
}
