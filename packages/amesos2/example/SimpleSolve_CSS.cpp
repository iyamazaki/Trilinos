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

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterXMLFileReader.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_CrsGraphTransposer.hpp>

// I/O for Matrix-Market files
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Import.hpp>

#include <Amesos2.hpp>
#include <Amesos2_Version.hpp>

#if defined(HAVE_AMESOS2_XPETRA) && defined(HAVE_AMESOS2_ZOLTAN2)
# include <Zoltan2_OrderingProblem.hpp>
# include <Zoltan2_PartitioningProblem.hpp>
# include <Zoltan2_TpetraRowGraphAdapter.hpp>
# include <Zoltan2_TpetraCrsMatrixAdapter.hpp>
# include <Zoltan2_XpetraMultiVectorAdapter.hpp>
#include "parmetis.h"
#endif


int main(int argc, char *argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc,&argv);

  typedef Tpetra::CrsMatrix<>::scalar_type Scalar;
  typedef Tpetra::Map<>::local_ordinal_type LO;
  typedef Tpetra::Map<>::global_ordinal_type GO;
  typedef Tpetra::Map<>::node_type NO;

  typedef Tpetra::RowGraph<LO, GO, NO> Graph;
  typedef Tpetra::CrsGraph<LO, GO, NO> CrsGraph;
  typedef Tpetra::CrsMatrix<Scalar,LO,GO> MAT;
  typedef Tpetra::MultiVector<Scalar,LO,GO> MV;

  using HostExecSpaceType =  Kokkos::DefaultHostExecutionSpace;
  using Tpetra::global_size_t;
  using Tpetra::Map;
  using Tpetra::Import;
  using Teuchos::RCP;
  using Teuchos::rcp;


  //
  // Get the default communicator
  //
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int myRank = comm->getRank();

  Teuchos::oblackholestream blackhole;

  bool printMatrix     = false;
  bool printSolution   = false;
  bool checkSolution   = false;
  bool printTiming     = false;
  bool useStackedTimer = false;
  bool allprint        = false;
  bool verbose = false;
  bool symmetrize = true;
  bool useZoltan2 = false;
  bool useParMETIS = false;
  std::string mat_filename("arc130.mtx");
  std::string rhs_filename("");
  std::string solvername("KLU");
  std::string xml_filename("");
  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("filename",&mat_filename,"Filename for Matrix-Market test matrix.");
  cmdp.setOption("rhs_filename",&rhs_filename,"Filename for Matrix-Market right-hand-side.");
  cmdp.setOption("solvername",&solvername,"Name of solver.");
  cmdp.setOption("xml_filename",&xml_filename,"XML Filename for Solver parameters.");
  cmdp.setOption("print-matrix","no-print-matrix",&printMatrix,"Print the full matrix after reading it.");
  cmdp.setOption("print-solution","no-print-solution",&printSolution,"Print solution vector after solve.");
  cmdp.setOption("check-solution","no-check-solution",&checkSolution,"Check solution vector after solve.");
  cmdp.setOption("symmetrize","no-symmetrize",&symmetrize,"Symmetrize for Zoltan2");
  cmdp.setOption("use-zoltan2","no-zoltan2",&useZoltan2,"Use Zoltan2 (Hypergraph) for repartitioning");
  cmdp.setOption("use-parmetis","no-parmetis",&useParMETIS,"Use ParMETIS for repartitioning");
  cmdp.setOption("print-timing","no-print-timing",&printTiming,"Print solver timing statistics");
  cmdp.setOption("use-stacked-timer","no-stacked-timer",&useStackedTimer,"Use StackedTimer to print solver timing statistics");
  cmdp.setOption("all-print","root-print",&allprint,"All processors print to out");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  std::ostream& out = ( (allprint || (myRank == 0)) ? std::cout : blackhole );
  RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));

  // Say hello
  out << myRank << " : " << Amesos2::version() << std::endl << std::endl;

  const size_t numVectors = 1;

  // Read matrix
  if (myRank == 0) {
    out << std::endl << " Reading " << mat_filename << std::endl << std::endl;
  }
  RCP<const MAT> A = Tpetra::MatrixMarket::Reader<MAT>::readSparseFile(mat_filename, comm);

  // get the map (Range Map used for both X & B)
  RCP<const Map<LO,GO> > rngmap = A->getRangeMap();
  RCP<const Map<LO,GO> > dmnmap = A->getDomainMap();
  GO nrows = A->getGlobalNumRows();

  // Create random X
  RCP<MV> X = rcp(new MV(dmnmap,numVectors));
  X->randomize();

  // Create B
  RCP<MV> B = rcp(new MV(rngmap,numVectors));
  if (rhs_filename == "") {
    /*
     * Use RHS:
     *
     *  [[10]
     *   [10]
     *   [10]
     *   [10]
     *   [10]
     *   [10]]
     */
    B->putScalar(10);
  } else {
    B = Tpetra::MatrixMarket::Reader<MAT>::readDenseFile (rhs_filename, comm, rngmap);
  }

  RCP<Teuchos::StackedTimer> stackedTimer;
  RCP<Amesos2::Solver<MAT,MV> > solver;
  if (useZoltan2 || useParMETIS) {
#if defined(HAVE_AMESOS2_XPETRA) && defined(HAVE_AMESOS2_ZOLTAN2)
    if(useStackedTimer) {
      stackedTimer = rcp(new Teuchos::StackedTimer("Amesos2 SimpleSolve-File"));
      Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    }
    // ===================================================
    // Symmetrize the matrix
    RCP<const CrsGraph> G;
    if (symmetrize) {
      if (myRank == 0) printf( "\n== symmetrize ==\n" );
      Tpetra::CrsGraphTransposer<LO,GO,NO> transposer(A->getCrsGraph());
      G = transposer.symmetrize();
    } else {
      if (myRank == 0) printf( "\n == original (skip symmetrize) ==\n" );
      G = A->getCrsGraph();
    }
    if (verbose) {
      RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      G->describe(*fancy,Teuchos::VERB_EXTREME);
      Tpetra::MatrixMarket::Writer<MAT>::writeSparseGraphFile("G.mtx", *G);
    }

    // ===================================================
    // Find ND partition
    int Nprocs = comm->getSize ();
    GO Nrows = G->getGlobalNumRows ();
    LO localNrows = G->getLocalNumRows ();
    int   *metis_perm    = (idx_t*)malloc( Nrows     * sizeof(int)); //
    int   *metis_iperm   = (idx_t*)malloc( Nrows     * sizeof(int)); // iperm[i] is new ith row
    idx_t *metis_sizes   = (idx_t*)malloc((2*Nprocs) * sizeof(idx_t));
    idx_t *metis_isizes  = (idx_t*)malloc((2*Nprocs) * sizeof(idx_t));
    idx_t *metis_vtxdist = (idx_t*)malloc((1+Nprocs) * sizeof(idx_t));
    {
      // No Zoltan2 interface to ParMETIS ND
      // So just call it directly for now
      // TODO : add the interface to Zoltan2
      idx_t Nrows = G->getGlobalNumRows ();
      auto rowmap = G->getLocalGraphHost ().row_map;
      auto colind = G->getLocalGraphHost ().entries;
      if (verbose) {
        char filename[200];
        sprintf(filename,"Gloc_%d.dat",myRank);
        FILE *fp = fopen(filename, "w");
        for (idx_t i = 0; i < localNrows; i++) {
          idx_t row = G->getRowMap()->getGlobalElement(i);
          for (idx_t k = rowmap[i]; k < rowmap[i+1]; k++) {
            idx_t col = G->getColMap()->getGlobalElement(colind(k));
            fprintf(fp,"%d %d, %d %d\n",i,colind[k], row,col);
          }
        }
        fclose(fp);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // Get row-distribuion
      int *vtxdist = (int*)malloc(Nprocs * sizeof(int));
      MPI_Allgather(&localNrows, 1, MPI_INT, vtxdist, 1, MPI_INT, MPI_COMM_WORLD);
      metis_vtxdist[0] = 0;
      for (int p=0; p<Nprocs; p++) {
        metis_vtxdist[p+1] = metis_vtxdist[p] + vtxdist[p];
      }
      //if (myRank == 0) for (int p=0; p<=Nprocs; p++) printf( " vtxdist[%d]=%d\n",p,metis_vtxdist[p] );
      // Remove diagonals
      idx_t *metis_rowmap = (idx_t*)malloc(rowmap.extent(0) * sizeof(idx_t));
      idx_t *metis_colind = (idx_t*)malloc(colind.extent(0) * sizeof(idx_t));
      idx_t nnz = 0;
      metis_rowmap[0] = nnz;
      for (int i=0; i<localNrows; i++) {
        idx_t row = G->getRowMap()->getGlobalElement(i);
        //if (myRank == 1) printf( " row = %d -> %d\n",i,row );
        for (int k=rowmap(i); k<rowmap(i+1); k++) {
          idx_t col = G->getColMap()->getGlobalElement(colind(k));
          //if (myRank == 1) printf( " - col = %d -> %d\n",colind(k),col );
          if (col != row) {
            //printf(" (%d)\n",nnz );
            metis_colind[nnz] = col;
            nnz ++;
          }
        }
        metis_rowmap[i+1] = nnz;
      }
      if (verbose) {
        char filename[200];
        sprintf(filename,"G_%d.dat",myRank);
        FILE *fp = fopen(filename, "w");
        for (idx_t i = 0; i < localNrows; i++) {
          for (idx_t k = metis_rowmap[i]; k < metis_rowmap[i+1]; k++) {
            fprintf(fp,"%d %d\n",metis_vtxdist[myRank]+i,metis_colind[k]);
          }
        }
        fclose(fp);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      MPI_Barrier(MPI_COMM_WORLD); if (myRank == 0) printf( " calling NodeND\n" ); fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);
      MPI_Comm metis_comm = MPI_COMM_WORLD;
      idx_t options[4]={0,0,0,1}, numflag = 0;
      idx_t *local_perm = (idx_t*)malloc(localNrows * sizeof(idx_t));
      ParMETIS_V3_NodeND(metis_vtxdist, metis_rowmap, metis_colind,
                         &numflag, options,
                         local_perm, metis_sizes, &metis_comm);
      MPI_Barrier(MPI_COMM_WORLD); if (myRank == 0) printf( " calling NodeND done\n" ); fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);

      // form global perm
      {
        int *rcount = (int*)malloc(Nprocs * sizeof(int));
        int *displs = (int*)malloc(Nprocs * sizeof(int));
        for (int i=0; i<Nprocs; i++) {
          rcount[i] = vtxdist[i];
          displs[i] = metis_vtxdist[i];
        }

        int *lperm = (int*)malloc(localNrows * sizeof(int));
        for (int i=0; i<localNrows; i++) lperm[i] = local_perm[i];
        MPI_Allgatherv(lperm, localNrows, MPI_INT, metis_perm, rcount, displs, MPI_INT, MPI_COMM_WORLD);
        for (int i=0; i<Nrows; i++) metis_iperm[metis_perm[i]] = i;

        if (verbose && myRank == 0) {
          printf("\n");
          printf("perm=[\n");
          for (int i=0; i<Nrows; i++) printf( " %d %d %d\n",i,metis_perm[i],metis_iperm[i] );
          printf("];\n");
	}
        if (/*verbose &&*/ myRank == 0) {
          printf("sizes=[\n");
          for (int i=0; i<2*Nprocs-1; i++) printf( " %d %d\n",metis_sizes[i],metis_vtxdist[i] );
          printf("];\n");
        }
      }
    }

    // ===================================================
    // Form min-overlap decomposition into two subdomains
    {
      // ---------------------------------
      // Map from Breadth-fast to Post-order
      int num_queued = 0;
      int num_levels = 1+(log(Nprocs) / log(2));
      // id of the first leaf node (BF order, post_order maps from top-down BF to post-order ND that matrix is ordered into)
      int leaves_id = pow(2.0, (double)(num_levels-1)) - 1;
      int *post_queue = (int*)malloc((2*Nprocs) * sizeof(int));
      int *post_check = (int*)malloc((2*Nprocs) * sizeof(int));
      int *post_order = (int*)malloc((2*Nprocs) * sizeof(int));
      for (int i = 0; i < (2*Nprocs); i++) post_check[i] = 0;
      if (verbose && myRank == 0) printf( " num levels = %d, leaves ID = %d\n",num_levels,leaves_id);

      // push first leaf to queue
      post_queue[num_queued] = 0;
      num_queued ++;

      int num_doms = 0;
      while (num_queued > 0) {
        // pop a node from queue
        int dom_id = post_queue[num_queued-1];
        if (verbose && myRank == 0) printf( " %d: > check (dom_id = %d) = %d\n",myRank,dom_id,post_check[dom_id] );
        if (dom_id >= leaves_id ||     // leaf
            post_check[dom_id] == 2)   // both children processed
        {
          post_order[num_doms] = dom_id;
          if (verbose && myRank == 0) printf( " %d: pop queue(%d) = %d -> post(%d) = %d\n\n",myRank,num_queued-1,dom_id, num_doms,dom_id );
          num_doms ++;

          if (dom_id != 0) {
            // if not root, let the parent node know one of its children has been processed
            int parent_id = (dom_id - 1)/2;
            post_check[parent_id] ++;
          }
          num_queued --;
        } else {
          // LIFO (so push right before left)
          // push right child
          if (verbose && myRank == 0) printf( " %d: push queue(%d) = %d\n",myRank,num_queued,2*dom_id+2 );
          post_queue[num_queued] = 2*dom_id + 2;
          num_queued ++;
          // push left child
          if (verbose && myRank == 0) printf( " %d: push queue(%d) = %d\n\n",myRank,num_queued,2*dom_id+1 );
          post_queue[num_queued] = 2*dom_id + 1;
          num_queued ++;
        }
      }
      int *post_iorder = (int*)malloc((2*Nprocs-1) * sizeof(int)); // Map post-order to top-down BF
      for (int i = 0; i < (2*Nprocs-1); i++) post_iorder[post_order[i]] = i;
      if (verbose && myRank == 0) {
        printf("\ntop-down bf to/from post-order\n" );
        printf(" post=[\n");
        for (int i = 0; i < (2*Nprocs)-1; i++) {
          printf(" %d %d\n",post_order[i],post_iorder[i]);
        }
        printf(" ];\n");
      }

      // ---------------------------------
      // Map Bottom-up BF to Top-down BF
      int offset = 0; // row offset
      int * bf_iorder = (int*)malloc((2*Nprocs) * sizeof(int));
      for (int i = num_levels-1; i >= 0; i--) {
        int first_id  = pow(2.0, i)-1;
        int num_nodes = pow(2.0, i);
        for (int k = 0; k < num_nodes; k++) {
          bf_iorder[offset+k] = first_id+k;
        }
        offset += num_nodes;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (verbose && myRank == 0) {
        printf("\ntop-down to bottom-up bf:\n" );
        printf( " bf_iorder=[\n" );
        for (int i=0; i < 2*Nprocs-1; i++ ) printf( " %d\n",bf_iorder[i] );
        printf( " ];\n\n" ); fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // sizes in post-order (perm is in post-order)
      int * sizes_bf = (int*)malloc((2*Nprocs-1) * sizeof(int));
      int * sizes_po = (int*)malloc((2*Nprocs-1) * sizeof(int));
      for (int i = 0; i < (2*Nprocs-1); i++) sizes_bf[bf_iorder[i]] = metis_sizes[i]; // size in Top-down BF
      for (int i = 0; i < (2*Nprocs-1); i++) sizes_po[i] = sizes_bf[post_order[i]];   // size in post-order
      int * disps_po = (int*)malloc((2*Nprocs) * sizeof(int));
      disps_po[0] = 0;
      for (int i = 0; i < (2*Nprocs-1); i++) disps_po[i+1] = disps_po[i] + sizes_po[i];
      MPI_Barrier(MPI_COMM_WORLD);
      if (verbose && myRank == 0) {
        printf("\ntop-down to bottom-up bf:\n" );
        printf(" post_size=[\n");
        for (int i = 0; i < (2*Nprocs)-1; i++) {
          printf(" %d %d\n",sizes_po[i],disps_po[i]);
        }
        printf(" ];\n"); fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // ---------------------------------
      // Construct Maps to construct min-overlapp DD
      // my subdom size
      GO nRows = 0;
      int skip = 1;
      for (int level = 0; level < (num_levels-1); level++) { // not root-separator
        //if (myRank%skip == 0)   // first MPI in the group
        if ((1+myRank)%skip == 0) // last  MPI in the group
	{
          int p0 = pow(2, (num_levels-1-level))-1;
          int p1 = myRank / skip;
          int p_bf = p0+p1; // top-down BF
          int p_po = post_iorder[p_bf]; // post_order
          if (verbose && myRank == 0) {
            printf(" === level = %d (p = %d+%d = %d -> %d) ==\n",level,p0,p1,p_bf,p_po);
            printf( " %d: nRows = %d + %d -> %d\n",myRank,nRows,sizes_po[p_po], nRows+sizes_po[p_po]); fflush(stdout);
          }
          nRows += sizes_po[p_po];
        }
        skip *= 2;
      }
      //if (myRank % (Nprocs/2) == 0)   // first MPI on each subdomain
      if ((myRank+1) % (Nprocs/2) == 0) // last MPI on each subdomain
      {
        int p_po = post_iorder[0];
        if (verbose) {
          printf( " -> %d: nRows = %d + %d -> %d\n",myRank,nRows,metis_sizes[p_po], nRows+metis_sizes[p_po]); fflush(stdout);
        }
        nRows += metis_sizes[p_po];
      } else if (verbose) {
        printf( " -> %d: nRows = %d\n",myRank,nRows); fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      // > GIDs for interior
      // > Original Row Map
      auto rowMap = A->getRowMap();
      Kokkos::View<GO*, HostExecSpaceType> rowIndexView ("indexList", nRows);
      skip = 1;
      nRows = 0;
      for (int level = 0; level < (num_levels-1); level++) { // not root-separator
        if (myRank%skip == 0) {
          int p0 = pow(2, (num_levels-1-level))-1;
          int p1 = myRank / skip;
          int p_bf = p0+p1; // top-down BF
          int p_po = post_iorder[p_bf]; // post_order
          //if (myRank == 2)printf("\n === level = %d (p = %d+%d = %d -> %d) ==\n",level,p0,p1,p_bf,p_po);
          for (int i=0; i < sizes_po[p_po]; i++) {
            GO old_i = metis_iperm[disps_po[p_po]+i];
            rowIndexView(nRows+i) = old_i;
            //if (myRank == 2) printf( " + rowRndexView(%d) = iperm[%d] = %d\n",nRows+i,disps_po[myRank]+i, old_i );
          }
          nRows += sizes_po[p_po];
        }
        skip *= 2;
      }
      if (myRank % (Nprocs/2) == 0) {
        // first MPI on each subdomain
        // > GIDs for top separator
        int p_po = post_iorder[0];
        for (int i=0; i < metis_sizes[p_po]; i++) {
          GO old_i = metis_iperm[disps_po[p_po]+i];
          rowIndexView(nRows+i) = old_i;
          //if (myRank == 2) printf( " * rowIndexView(%d) = metis_iperm[%d] = %d\n",nRows+i,disps_po[p_po]+i, old_i );
        }
        nRows += metis_sizes[p_po];
      }
      // > Communicator for subdomain
      int color1 = (myRank < Nprocs/2 ? 0 : 1);
      int color2 = (myRank < Nprocs/2 ? 1 : 0);
      RCP<const Teuchos::Comm<int> > subComm1 = comm->split(color1, comm->getRank());
      RCP<const Teuchos::Comm<int> > subComm2 = comm->split(color2, comm->getRank());
      int color = (myRank < Nprocs/2 ? 0 : 1);
      RCP<const Teuchos::Comm<int> > subComm = (myRank < Nprocs/2 ? subComm1 : subComm2);
      //printf( " %d:%d: color(%d,%d) -> size=%d\n",color,myRank,color1,color2,subComm->getSize() ); fflush(stdout);
      // > Original Col Map
      auto colMap = A->getColMap();
      int np = color * (Nprocs-1);
      int nCols = sizes_po[2*Nprocs-2] + (disps_po[np+Nprocs-1]-disps_po[np]);
      Kokkos::View<GO*, HostExecSpaceType> colIndexView ("indexList", nCols);
      nCols = 0;
      for (int i=disps_po[np]; i < disps_po[np+Nprocs-1]; i++) {
        colIndexView(nCols) = metis_iperm[i];
        //if (myRank == 2) printf( "> colIndexView(%d) = metis_iperm[%d] = %d\n",nCols,i,metis_iperm[i] );
	nCols ++;
      }
      for (int i=disps_po[2*Nprocs-2]; i < disps_po[2*Nprocs-1]; i++) {
        colIndexView(nCols) = metis_iperm[i];
        //if (myRank == 2) printf( " * colIndexView(%d) = metis_iperm[%d] = %d\n",nCols,i,metis_iperm[i] );
	nCols ++;
      }
      if (verbose) {
        //printf( " Print row index (%d)\n",myRank ); fflush(stdout);
        char filename[200];
        sprintf(filename,"indRows_%d.dat",myRank);
        FILE *fp = fopen(filename, "w");
        for (idx_t i = 0; i < nRows; i++) {
          fprintf(fp,"%d %d\n",i,rowIndexView(i));
        }
        fclose(fp);
        sprintf(filename,"indCols_%d.dat",myRank);
        fp = fopen(filename, "w");
        for (idx_t i = 0; i < nCols; i++) {
          fprintf(fp,"%d %d\n",i,colIndexView(i));
        }
        fclose(fp);
        MPI_Barrier(MPI_COMM_WORLD);
        //printf( " Done print row index (%d)\n",myRank ); fflush(stdout);
      }
      // > New Row/Col Maps 
      auto rowComm = rowMap->getComm();
      auto colComm = colMap->getComm();
      GO indexBase = rowMap->getIndexBase();
      Teuchos::ArrayView<const GO> rowIndexList (rowIndexView.data(), rowIndexView.size());
      Teuchos::ArrayView<const GO> colIndexList (colIndexView.data(), colIndexView.size());
      Teuchos::RCP<const Map<LO,GO>> newRowMap = Teuchos::rcp (new Map<LO,GO> (nCols, rowIndexList, indexBase, subComm));
      Teuchos::RCP<const Map<LO,GO>> newColMap = Teuchos::rcp (new Map<LO,GO> (nCols, colIndexList, indexBase, subComm));
      int prind_color = 1;
      if (verbose) {
        printf( " Print row maps (%d)\n",myRank ); fflush(stdout);
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        if (myRank==0) std::cout << std::endl << " == Original RowMap ==" << std::endl << std::flush;
        rowMap->describe(*fancy, Teuchos::VERB_EXTREME);
        if (myRank==0) std::cout << std::endl << " == Original ColMap ==" << std::endl << std::flush;
        colMap->describe(*fancy, Teuchos::VERB_EXTREME);

        if (myRank==0) std::cout << std::endl << " == New RowMap ==" << std::endl << std::flush; MPI_Barrier(MPI_COMM_WORLD);
        if (color ==prind_color) newRowMap->describe(*fancy, Teuchos::VERB_EXTREME);             MPI_Barrier(MPI_COMM_WORLD);
        if (myRank==0) std::cout << std::endl << " == New ColMap ==" << std::endl << std::flush; MPI_Barrier(MPI_COMM_WORLD);
        if (color ==prind_color) newColMap->describe(*fancy, Teuchos::VERB_EXTREME);
      }
      fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);

      // do Import subdomain matrix
      GO maxEntries = A->getGlobalMaxNumRowEntries();
      Tpetra::Import<LO, GO, NO> rowImporter(rowMap, newRowMap);
      Teuchos::RCP<MAT> Kii = Teuchos::rcp( new MAT(newRowMap, newColMap, indexBase) );
      Kii->doImport(*A, rowImporter, Tpetra::INSERT);
      Kii->fillComplete();
      if (verbose) {
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        if (myRank==0) std::cout << std::endl << " == New Matrix ==" << std::endl;
        if (color ==prind_color) Kii->describe(*fancy, Teuchos::VERB_EXTREME);
        if (myRank==0) std::cout << std::endl << " == New Matrix done ==" << std::endl;
        char filename[200];
        sprintf(filename,"Kii_%d.dat", color);
	printf( " %d: %s\n",myRank,filename ); fflush(stdout);
        Tpetra::MatrixMarket::Writer<MAT>::writeSparseFile (filename, Kii);
      }
      fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);

      // Create Subdomain Amesos2 solver
      if (myRank==0) std::cout << std::endl << " Build Subdomain solver (" << solvername << ") ==" << std::endl;
      solver = Amesos2::create<MAT,MV>(solvername, Kii);
      if (xml_filename != "") {
        Teuchos::ParameterList test_params =
          Teuchos::ParameterXMLFileReader(xml_filename).getParameters();
        Teuchos::ParameterList& amesos2_params = test_params.sublist("Amesos2");
        //*fos << amesos2_params.currentParametersString() << std::endl;
        solver->setParameters( Teuchos::rcpFromRef(amesos2_params) );
      } else {
        Teuchos::ParameterList amesos2_params("Amesos2");
        Teuchos::ParameterList& solver_params = amesos2_params.sublist(solvername);
        solver_params.set<bool>("IsContiguous", false);
        //*fos << amesos2_params.currentParametersString() << std::endl;
        solver->setParameters( Teuchos::rcpFromRef(amesos2_params) );
      }
      {
        // symbolic/numeric factor subdomain matrix
        //if (color == 1)
        {
          auto SymbolicTimer = Teuchos::TimeMonitor::getNewTimer("Amesos2 SimpleSolve-File : Symbolic Factorization");
          Teuchos::TimeMonitor symbolicTimer( *SymbolicTimer );
          solver->symbolicFactorization();
        }
	MPI_Barrier(MPI_COMM_WORLD);
        //if (color == 1)
        {
          auto NumericTimer = Teuchos::TimeMonitor::getNewTimer("Amesos2 SimpleSolve-File : Numeric Factorization");
          Teuchos::TimeMonitor numericTimer( *NumericTimer );
          solver->numericFactorization();
        }
      }
      MPI_Barrier(MPI_COMM_WORLD); if (myRank == 0) printf( "\n == done with subdomain solve == \n\n" ); 
    }
    if(useStackedTimer) {
      stackedTimer->stopBaseTimer();
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
      useZoltan2, std::invalid_argument,
      "Both Xpetra and Zoltan2 are needed to use Zoltan2.");
#endif
  } else {
    // Stadard Amesos2
    if( printMatrix ){
      A->describe(*fos, Teuchos::VERB_EXTREME);
    }
    else if( verbose ){
      std::cout << std::endl << A->description() << std::endl << std::endl;
    }

    // Constructor from Factory
    if( !Amesos2::query(solvername) ){
      *fos << solvername << " solver not enabled.  Exiting..." << std::endl;
      return EXIT_SUCCESS;
    }

    solver = Amesos2::create<MAT,MV>(solvername, A, X, B);
    if (xml_filename != "") {
      Teuchos::ParameterList test_params =
        Teuchos::ParameterXMLFileReader(xml_filename).getParameters();
      Teuchos::ParameterList& amesos2_params = test_params.sublist("Amesos2");
      *fos << amesos2_params.currentParametersString() << std::endl;
      solver->setParameters( Teuchos::rcpFromRef(amesos2_params) );
    }

    if(useStackedTimer) {
      stackedTimer = rcp(new Teuchos::StackedTimer("Amesos2 SimpleSolve-File"));
      Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    }
    solver->symbolicFactorization().numericFactorization().solve();
    if(useStackedTimer) {
      stackedTimer->stopBaseTimer();
    }
  }

  if( printSolution ){
    // Print the solution
    RCP<Map<LO,GO> > root_map
      = rcp( new Map<LO,GO>(nrows,myRank == 0 ? nrows : 0,0,comm) );
    RCP<MV> Xhat = rcp( new MV(root_map,numVectors) );
    RCP<Import<LO,GO> > importer = rcp( new Import<LO,GO>(rngmap,root_map) );
    if( allprint ){
      if( myRank == 0 ) *fos << "Solution :" << std::endl;
      Xhat->describe(*fos,Teuchos::VERB_EXTREME);
      *fos << std::endl;
    } else {
      Xhat->doImport(*X,*importer,Tpetra::REPLACE);
      if( myRank == 0 ){
        *fos << "Solution :" << std::endl;
        Xhat->describe(*fos,Teuchos::VERB_EXTREME);
        *fos << std::endl;
      }
    }
  }

  if( checkSolution ){
    const Scalar one = Teuchos::ScalarTraits<Scalar>::one ();
    RCP<MV> R = rcp(new MV(rngmap,numVectors));
    A->apply(*X, *R);
    R->update(one, *B, -one);
    for (size_t j = 0; j < numVectors; ++j) {
      auto Rj = R->getVector(j);
      auto Bj = B->getVector(j);
      auto r_norm = Rj->norm2();
      auto b_norm = Bj->norm2();
      if (myRank == 0) {
        *fos << "Relative Residual norm = " << r_norm << " / " << b_norm << " = "
             << r_norm / b_norm << std::endl;
      }
    }
    if (myRank == 0) *fos << std::endl;
  }

  if(useStackedTimer) {
    Teuchos::StackedTimer::OutputOptions options;
    options.num_histogram=3;
    options.print_warnings = false;
    options.output_histogram = true;
    options.output_fraction=true;
    options.output_minmax = true;
    stackedTimer->report(std::cout, comm, options);
  } else if( printTiming ){
    // Print some timing statistics
    solver->printTiming(*fos);
  } else {
    Teuchos::TimeMonitor::summarize();
  }

  // We are done.
  return 0;
}
