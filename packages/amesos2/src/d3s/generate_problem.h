#pragma once

#include <vector>
#include <tuple>
#include <mpi.h>
#include "throwAssert.h"

class GenerateProblem
{
public:
  GenerateProblem(MPI_Comm comm_in,
                  const std::string & input_file);

  void scaleDiagonal();
  
  int getMyPID();
  
  int getNumRowsAll() const;

  int getNumProcSolver() const {return numProcSolver;}

  int getNumThreads() const {return num_threads;}

  int getAddNonsymmetry() const {return add_nonsymmetry;}

  int getRemoveSomeEntries() const {return remove_some_entries;}

  int getMessageLevel() const {return msg_level;}

  int getReorderOption() const {return reorder_option;}

  int getDebugLevel() const {return debugLevel;}

  int getNumFactorizations() const {return num_factorizations;}

  void getLinearSystem(std::vector<int> & rowBeginOut,
                       std::vector<int> & columnsOut,
                       std::vector<double> & valuesOut,
                       std::vector<double> & rhsOut,
                       int & startGIDOut) const
  {
    rowBeginOut = rowBegin;
    columnsOut = columns;
    valuesOut = values;
    rhsOut = rhs;
    startGIDOut = startGID;
  }
  
 private:
  
  void getDispls(const std::vector<int> & numEntriesProc,
                 std::vector<int> & displs) const;
  
  void scatterMatrix(const std::vector<std::tuple<int,int,double>> & A,
                     std::vector<int> & numRowsProc);
  
  void scatterRhs(const std::vector<int> & numRowsProc,
                  const std::vector<double> & rhs_mm);
  
  void makeAdjustments();
  
  void readProcMatrices();
  
  void readMatrixMarket(const std::string & matrixFile,
                        int & num_rows,
                        std::vector<std::tuple<int,int,double>> & A) const;
  
  void readMatrixRhs(const std::string & rhsFile,
                     const int numRows,
                     std::vector<double> & rhs_mm);
  
  void readParameters(const std::string & inputFile);
  
  void scaleUpperTriangle();

  void removeSomeMatrixEntries(const int maxRemove);

  void zeroSomeMatrixEntries(const int maxZero);
  
  std::vector<double> getElemMatrix(const int dim) const;

  int getNumNodeElem(const int dim) const;
  
  void generateNodalCoordinates(const std::vector<int> & numElemDir,
                                std::vector<double> & x,
                                std::vector<double> & y,
                                std::vector<double> & z) const;
  
  void getNumTermsProc(const std::vector<std::tuple<int,int,double>> & A,
                       const std::vector<int> & numRowsProc,
                       std::vector<int> & numTermsProc) const;
  
  void extractRowBegin(const std::vector<int> & Arows,
                       const int numRows);
  
  void outputMatrix(const std::vector<std::tuple<int,int,double>> & A) const;
  
  void getLengthAndDirectionCosines(const int node1,
                                    const int node2,
                                    const int dim,
                                    const std::vector<double> & xElem,
                                    const std::vector<double> & yElem,
                                    const std::vector<double> & zElem,
                                    double & length,
                                    double* axial_dcs) const;
  
  void generateElemConnectivity(const std::vector<int> & numElemDir,
                                std::vector<std::vector<int>> & elemConn) const;
  
  void outputElemMatrix(const std::vector<double> & elemMatrix) const;

  void generate_rhs(const int numRows);
  
  void generateMesh(const std::vector<int> & numElemDir,
                    std::vector<std::vector<int>> & elemConn,
                    std::vector<double> & x,
                    std::vector<double> & y,
                    std::vector<double> & z) const;

  void generateMatrix(const std::vector<std::vector<int>> & elemConn,
                      const std::vector<double> & elemMatrix,
                      std::vector<std::tuple<int,int,double>> & A) const;


  void scatterMatrix(const std::vector<std::tuple<int,int,double>> & A);

  void generateMatrix(std::vector<std::tuple<int,int,double>> & A);
  
  void generateProcMatrices();
  
  
  MPI_Comm comm;

  std::vector<int> rowBegin, columns, numRowsProc;
  std::vector<double> values, rhs;
  int myPID=-1, numProc=-1, root=0, startGID=-1;

  int matrix_option, numProcSolver, msg_level, num_threads, reorder_option, debugLevel, add_nonsymmetry,
    remove_some_entries, zero_some_entries, num_factorizations, output_elem_matrix, output_matrix;
  std::string filenameBase, filenameBaseRhs, problem_type;
  std::vector<int> numElemDir;

};

