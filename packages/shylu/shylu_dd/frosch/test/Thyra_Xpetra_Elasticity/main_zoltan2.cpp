//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#include <ShyLU_DDFROSch_config.h>

#include <mpi.h>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_StackedTimer.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

// Thyra includes
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>

// Stratimikos includes
#include <Stratimikos_FROSch_def.hpp>

#include <Tpetra_Core.hpp>

// Xpetra include
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>

// FROSCH thyra includes
#include "Thyra_FROSchLinearOp_def.hpp"
#include "Thyra_FROSchFactory_def.hpp"
#include <FROSch_Tools_def.hpp>

// Zoltan2 includes
#include "Zoltan2_PartitioningProblem.hpp"
#include "Zoltan2_XpetraCrsMatrixAdapter.hpp"
#include "Zoltan2_XpetraMultiVectorAdapter.hpp"


using UN    = unsigned;
using SC    = double;
using LO    = int;
using GO    = FROSch::DefaultGlobalOrdinal;
using NO    = KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;
using namespace Thyra;

int main(int argc, char *argv[])
{
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();

    CommandLineProcessor My_CLP;

    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

    int M = 3;
    My_CLP.setOption("M",&M,"H / h.");
    int Dimension = 2;
    My_CLP.setOption("DIM",&Dimension,"Dimension.");
    int Overlap = 0;
    My_CLP.setOption("O",&Overlap,"Overlap.");
    string xmlFile = "ParameterList.xml";
    My_CLP.setOption("PLIST",&xmlFile,"File name of the parameter list.");
    bool useepetra = false;
    My_CLP.setOption("USEEPETRA","USETPETRA",&useepetra,"Use Epetra infrastructure for the linear algebra.");
    bool useGeoMap = false;
    My_CLP.setOption("useGeoMap","useAlgMap",&useGeoMap,"Use Geometric Map");
    bool useZoltan2 = false;
    My_CLP.setOption("useZoltan2","noZoltan2",&useZoltan2,"Use Zoltan2");
    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc,argv);
    if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }

    CommWorld->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Thyra Elasticity Test"));
    TimeMonitor::setStackedTimer(stackedTimer);

    int N = 0;
    int color=1;
    if (Dimension == 2) {
        N = (int) (pow(CommWorld->getSize(),1/2.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N || useZoltan2) {
            color=0;
        }
    } else if (Dimension == 3) {
        N = (int) (pow(CommWorld->getSize(),1/3.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N*N || useZoltan2) {
            color=0;
        }
    } else {
        assert(false);
    }

    UnderlyingLib xpetraLib = UseTpetra;
    if (useepetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }

    RCP<const Comm<int> > comm = CommWorld->split(color,CommWorld->getRank());

    if (color==0) {

        RCP<ParameterList> parameterList = getParametersFromXmlFile(xmlFile);

        comm->barrier();
        if (comm->getRank()==0) {
            cout << "##################\n# Parameter List #\n##################" << endl;
            parameterList->print(cout);
            cout << endl;
        }

        comm->barrier(); if (comm->getRank()==0) cout << "##############################\n# Assembly Laplacian #\n##############################\n" << endl;

        ParameterList GaleriList;

        RCP<const Map<LO,GO,NO> > UniqueMap;
        RCP<MultiVector<SC,LO,GO,NO> > Coordinates;

        RCP<Matrix<SC,LO,GO,NO> > K;

        #if 1
        // Re-partition using Zoltan2
        if (useZoltan2) {
          GO nGlobalElements = 3*M*M*M;

          // Create a map just for Rank-0
          int rank0 = (comm->getRank() == 0 ? 0 : 1);
          RCP<const Comm<int> > comm0 = CommWorld->split(rank0, CommWorld->getRank());
comm->barrier();
std::cout << " done 0" << std::endl << std::flush;
comm->barrier();
          #if 0
          using tpetra_map_type = Tpetra::Map<LO,GO,NO>;
          RCP<const tpetra_map_type> Tmap0 = rcp(new tpetra_map_type(nGlobalElements, 0, comm0));
          RCP<const Map<LO,GO,NO> >UniqueMap0 = rcp (new const TpetraMap<LO,GO,NO>(Tmap0));
          #else
          RCP<const Map<LO,GO,NO> > UniqueMap0 = MapFactory<LO,GO,NO>::Build(xpetraLib, nGlobalElements, 0, comm0);
          #endif

comm->barrier();
std::cout << " done 1" << std::endl << std::flush;
comm->barrier();
          // build K0 on Rank-0
          auto out = getFancyOStream (rcpFromRef (std::cout));
          RCP<Matrix<SC,LO,GO,NO> > K0;
comm->barrier();
std::cout << " done 2" << std::endl << std::flush;
comm->barrier();
          if (comm->getRank() == 0) 
          {
            GaleriList.set("nx", GO(M));
            GaleriList.set("ny", GO(M));
            GaleriList.set("nz", GO(M));
            GaleriList.set("mx", GO(1));
            GaleriList.set("my", GO(1));
            GaleriList.set("mz", GO(1));
            std::cout << " M = " << M << " nGlobalElements = " << nGlobalElements << std::endl << std::flush;

            RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > >
              Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Elasticity3D", UniqueMap0, GaleriList);
            K0 = Problem->BuildMatrix();
            K0->describe(*out);
          } else {
            K0 = MatrixFactory<SC,LO,GO,NO>::Build (UniqueMap0);
          }
          comm->barrier();
comm->barrier();
std::cout << " done 3" << std::endl;
comm->barrier();

          // Export K0 to K in uniform 1D block row
          #if 0
          RCP<const tpetra_map_type> Tmap  = rcp(new tpetra_map_type(nGlobalElements, 0, comm));
          UniqueMap = rcp (new const TpetraMap<LO,GO,NO>(Tmap));
          #else
          UniqueMap = MapFactory<LO,GO,NO>::Build(xpetraLib, nGlobalElements, 0, comm);
          #endif
          K = MatrixFactory<SC,LO,GO,NO>::Build (UniqueMap);
          #if 1
          auto exporter = Xpetra::ExportFactory<LO, GO>::Build(UniqueMap0, UniqueMap);
          K->doExport (*K0, *exporter, Xpetra::INSERT);
          #else
          auto importer = Xpetra::ImportFactory<LO, GO>::Build(UniqueMap0, UniqueMap);
          K->doImport (*K0, *importer, Xpetra::INSERT);
          #endif
          K->fillComplete();
          K->describe(*out);

          // wrap K into Zoltan2 matrix
          CrsMatrixWrap<SC,LO,GO,NO>& crsWrapK = dynamic_cast<CrsMatrixWrap<SC,LO,GO,NO>&>(*K);
          Zoltan2::XpetraCrsMatrixAdapter<Xpetra::CrsMatrix<SC, LO, GO>>
            zoltan_matrix(crsWrapK.getCrsMatrix());

          // Specify partitioning parameters
          Teuchos::ParameterList zoltan_params;
          zoltan_params.set("partitioning_approach", "partition");
          zoltan_params.set("symmetrize_input", "transpose");
          zoltan_params.set("partitioning_objective", "minimize_cut_edge_weight");

          // Create and solve partitioning problem
          Zoltan2::PartitioningProblem<Zoltan2::XpetraCrsMatrixAdapter<Xpetra::CrsMatrix<SC, LO, GO>>> 
          problem(&zoltan_matrix, &zoltan_params);
          problem.solve();

          // Redistribute matrix
          RCP<Xpetra::CrsMatrix<SC, LO, GO>> zoltan_K;
          zoltan_matrix.applyPartitioningSolution (*(crsWrapK.getCrsMatrix()), zoltan_K, problem.getSolution());

          // Set it as coefficient matrix
          K = rcp (new CrsMatrixWrap<SC,LO,GO,NO>(zoltan_K));
          UniqueMap = K->getMap();
        } else
        #endif
        {
          GaleriList.set("nx", GO(N*M));
          GaleriList.set("ny", GO(N*M));
          GaleriList.set("nz", GO(N*M));
          GaleriList.set("mx", GO(N));
          GaleriList.set("my", GO(N));
          GaleriList.set("mz", GO(N));

          RCP<const Map<LO,GO,NO> > UniqueNodeMap;
          if (Dimension==2) {
            UniqueNodeMap = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian2D",comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
            UniqueMap = Xpetra::MapFactory<LO,GO,NO>::Build(UniqueNodeMap,2);
            Coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("2D",UniqueMap,GaleriList);
            RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Elasticity2D",UniqueMap,GaleriList);
            K = Problem->BuildMatrix();
          } else if (Dimension==3) {
            UniqueNodeMap = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian3D",comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
            UniqueMap = Xpetra::MapFactory<LO,GO,NO>::Build(UniqueNodeMap,3);
            Coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("3D",UniqueMap,GaleriList);
            RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Elasticity3D",UniqueMap,GaleriList);
            K = Problem->BuildMatrix();
          }
        }

        RCP<Map<LO,GO,NO> > FullRepeatedMap;
        RCP<Map<LO,GO,NO> > RepeatedMap;
        RCP<const Map<LO,GO,NO> > FullRepeatedMapNode;
        if (useGeoMap) {
            if (Dimension == 2) {
                FullRepeatedMap = BuildRepeatedMapGaleriStruct2D<SC,LO,GO,NO>(K,M,Dimension);
                RepeatedMap = FullRepeatedMap;
            } else if (Dimension == 3) {
                FullRepeatedMapNode = BuildRepeatedMapGaleriStruct3D<SC,LO,GO,NO>(K->getMap(),M,Dimension);
                FullRepeatedMap = BuildMapFromNodeMap(FullRepeatedMapNode,Dimension,NodeWise);
                //FullRepeatedMapNode->describe(*fancy,Teuchos::VERB_EXTREME);
                RepeatedMap = FullRepeatedMap;
            }
        } else {
            RepeatedMap = BuildRepeatedMapNonConst<LO,GO,NO>(K->getCrsGraph());
        }

        RCP<MultiVector<SC,LO,GO,NO> > xSolution = MultiVectorFactory<SC,LO,GO,NO>::Build(UniqueMap,1);
        RCP<MultiVector<SC,LO,GO,NO> > xRightHandSide = MultiVectorFactory<SC,LO,GO,NO>::Build(UniqueMap,1);

        xSolution->putScalar(ScalarTraits<SC>::zero());
        xRightHandSide->putScalar(ScalarTraits<SC>::one());

        CrsMatrixWrap<SC,LO,GO,NO>& crsWrapK = dynamic_cast<CrsMatrixWrap<SC,LO,GO,NO>&>(*K);
        RCP<const LinearOpBase<SC> > K_thyra = ThyraUtils<SC,LO,GO,NO>::toThyra(crsWrapK.getCrsMatrix());
        RCP<MultiVectorBase<SC> >thyraX = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xSolution));
        RCP<const MultiVectorBase<SC> >thyraB = ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xRightHandSide);

        //-----------Set Coordinates and RepMap in ParameterList--------------------------
        RCP<ParameterList> plList = sublist(parameterList,"Preconditioner Types");
        sublist(plList,"FROSch")->set("Dimension",Dimension);
        sublist(plList,"FROSch")->set("Overlap",Overlap);
        sublist(plList,"FROSch")->set("DofOrdering","NodeWise");
        sublist(plList,"FROSch")->set("Repeated Map",RepeatedMap);
        if (!useZoltan2) {
          sublist(plList,"FROSch")->set("DofsPerNode",Dimension);
          sublist(plList,"FROSch")->set("Coordinates List",Coordinates);
        }
        comm->barrier();
        if (comm->getRank()==0) {
            cout << "##################\n# Parameter List #\n##################" << endl;
            parameterList->print(cout);
            cout << endl;
        }

        comm->barrier(); if (comm->getRank()==0) cout << "###################################\n# Stratimikos LinearSolverBuilder #\n###################################\n" << endl;
        Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
        Stratimikos::enableFROSch<LO,GO,NO>(linearSolverBuilder);
        linearSolverBuilder.setParameterList(parameterList);

        comm->barrier(); if (comm->getRank()==0) cout << "######################\n# Thyra PrepForSolve #\n######################\n" << endl;

        RCP<LinearOpWithSolveFactoryBase<SC> > lowsFactory =
        linearSolverBuilder.createLinearSolveStrategy("");

        lowsFactory->setOStream(out);
        lowsFactory->setVerbLevel(VERB_HIGH);

        comm->barrier(); if (comm->getRank()==0) cout << "###########################\n# Thyra LinearOpWithSolve #\n###########################" << endl;

        RCP<LinearOpWithSolveBase<SC> > lows =
        linearOpWithSolve(*lowsFactory, K_thyra);

        comm->barrier(); if (comm->getRank()==0) cout << "\n#########\n# Solve #\n#########" << endl;
        SolveStatus<double> status =
        solve<double>(*lows, Thyra::NOTRANS, *thyraB, thyraX.ptr());

        comm->barrier(); if (comm->getRank()==0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }

    CommWorld->barrier();
    stackedTimer->stop("Thyra Elasticity Test");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report(*out,CommWorld,options);

    return(EXIT_SUCCESS);

}
