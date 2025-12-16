#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string>
#include <algorithm>
#include <map>
#include <cstdlib>
#include "generate_problem.h"

GenerateProblem::GenerateProblem(MPI_Comm comm_in,
                                 const std::string & input_file,
                                 const bool sort_colinds) :
  comm(comm_in)
{
  MPI_Comm_rank(comm, &myPID);
  MPI_Comm_size(comm, &numProc);
  readParameters(input_file);

  std::vector<int> numRowsProc;
  std::vector<std::tuple<int,int,double>> A;
  std::vector<double> rhs_mm;
  if (matrix_option == 1) {
    if (myPID == 0 && msg_level > 0) std::cout << "reading matrix from distributed files" << std::endl;
    readProcMatrices();
    if (myPID == 0 && msg_level > 0) std::cout << "done reading matrix from distributed files" << std::endl;
  }
  else if (matrix_option == 2) {  
    if (myPID == 0 && msg_level > 0) std::cout << "generating matrix on-the-fly" << std::endl;
    generateModelMatrix(A);
    scatterMatrix(A, numRowsProc, sort_colinds);

    const int numRows = rowBegin.size() - 1;
    generate_rhs(numRows);
    if (myPID == 0 && msg_level > 0) std::cout << "done generating matrix" << std::endl;
  }
  else if (matrix_option == 3) {
    if (myPID == 0) {
      if (msg_level > 0) std::cout << "reading matrix from matrix market file" << std::endl;
      int num_rows_mm;
      readMatrixMarket(filenameBase, num_rows_mm, A);
      readMatrixRhs(filenameBaseRhs, num_rows_mm, rhs_mm);
      if (msg_level > 0) std::cout << "done reading Matrix Market file" << std::endl;
    }
    scatterMatrix(A, numRowsProc, sort_colinds);
    scatterRhs(numRowsProc, rhs_mm);    
  }
  makeAdjustments();
}

void GenerateProblem::getDispls(const std::vector<int> & numEntriesProc,
                                std::vector<int> & displs) const
{
  displs.assign(numProc, 0);
  displs.resize(numProc, 0);
  for (int i=1; i<numProc; i++) {
    displs[i] = displs[i-1] + numEntriesProc[i-1];
  }
}

int GenerateProblem::getNumRowsAll() const
{
  const int numRows = rowBegin.size() - 1;
  int numRowsAll;
  MPI_Allreduce(&numRows, &numRowsAll, 1, MPI_INT, MPI_SUM, comm);
  return numRowsAll;
}

void GenerateProblem::readProcMatrices()
{
  std::ifstream fin;
  std::string filename = filenameBase + "_mat_" + std::to_string(myPID)
                         + ".dat";
  fin.open(filename);
  if (fin.is_open() == false) {
    std::cout << "could not open file " << filename << std::endl;
    ThrowAssert(0, "error opening file");
  }
  int numRows, numCols, numTerms;
  fin >> numRows >> numCols >> numTerms;
  ThrowAssert(numRows == numCols, "number of rows/cols must be equal");
  rowBegin.resize(numRows+1, 0);
  columns.resize(numTerms);
  values.resize(numTerms);
  std::cout << "reading " << filename << std::endl;
  int numRowsCurr(0);
  int previousRow = -1;
  for (int i=0; i<numTerms; i++) {
    int row, col;
    double value;
    fin >> row >> col >> value;
    row--; col--;
    columns[i] = col;
    values[i] = value;
    if (row != previousRow) {
      rowBegin[numRowsCurr] = i;
      numRowsCurr++;
      previousRow = row;
    }
  }
  rowBegin[numRows] = numTerms;
  fin.close();
  // Note: we sort columns prior to calling MKL/Pardiso in d3_solver.C
  //  sortMatrixColumns(rowBegin, columns, values, myPID);
  std::vector<int> numRowsProc(numProc);
  MPI_Allgather(&numRows, 1, MPI_INT, numRowsProc.data(), 1, MPI_INT, comm);
  startGID = 0;
  for (int i=0; i<myPID; i++) startGID += numRowsProc[i];

  if (filenameBaseRhs == "none") {
    generate_rhs(numRows);
    return;
  }

  filename = filenameBaseRhs + "_rhs_" + std::to_string(myPID)
             + ".dat";
  fin.open(filename);
  if (fin.is_open() == false) {
    std::cout << "could not open file " << filename << std::endl;
    ThrowAssert(0, "error opening file");
  }
  int numRowsRhs;
  fin >> numRowsRhs;
  ThrowAssert(numRowsRhs == numRows, "number of rows not consistent");
  rhs.resize(numRows);
  std::cout << "reading " << filename << std::endl;
  for (int i=0; i<numRows; i++) {
    fin >> rhs[i];
  }
  fin.close();
}

void GenerateProblem::scatterMatrix(const std::vector<std::tuple<int,int,double>> & A,
                                    std::vector<int> & numRowsProc, const bool sort_colinds)
{
  numRowsProc.resize(numProc);
  std::vector<int> startGIDsProc, numTermsProc, ArowsRoot, AcolsRoot;
  std::vector<double> AvalsRoot;
  if (myPID == root) {
    ArowsRoot.resize(A.size());
    AcolsRoot.resize(A.size());
    AvalsRoot.resize(A.size());
    for (size_t i=0; i<A.size(); i++) {
      ArowsRoot[i] = std::get<0>(A[i]);
      AcolsRoot[i] = std::get<1>(A[i]);
      AvalsRoot[i] = std::get<2>(A[i]);
    }
    const int index = A.size() - 1;
    const int numRows = std::get<0>(A[index]) + 1; // 0-based indices
    const int numRowsPerProc = numRows/numProc;
    numRowsProc.resize(numProc);
    startGIDsProc.resize(numProc);
    int numRowsSum = 0;
    for (int i=0; i<numProc; i++) {
      startGIDsProc[i] = numRowsSum;
      numRowsProc[i] = numRowsPerProc;
      if (i == numProc-1) {
        numRowsProc[i] = numRows - numRowsSum;
      }
      numRowsSum += numRowsProc[i];
    }
    getNumTermsProc(A, numRowsProc, numTermsProc);
  }
  int numRows, numTermsA;
  MPI_Scatter(numRowsProc.data(), 1, MPI_INT, &numRows, 1, MPI_INT, root, comm);
  MPI_Scatter(startGIDsProc.data(), 1, MPI_INT, &startGID, 1, MPI_INT, root, comm);
  MPI_Scatter(numTermsProc.data(), 1, MPI_INT, &numTermsA, 1, MPI_INT, root, comm);
  columns.resize(numTermsA);
  values.resize(numTermsA);
  std::vector<int> Arows(numTermsA);
  std::vector<int> displs(numProc, 0);
  if (myPID == root) {
    for (int i=1; i<numProc; i++) {
      displs[i] = displs[i-1] + numTermsProc[i-1];
    }
  }
  MPI_Scatterv(ArowsRoot.data(), numTermsProc.data(), displs.data(), MPI_INT,
               Arows.data(), numTermsA, MPI_INT, root, comm);
  MPI_Scatterv(AcolsRoot.data(), numTermsProc.data(), displs.data(), MPI_INT,
               columns.data(), numTermsA, MPI_INT, root, comm);
  MPI_Scatterv(AvalsRoot.data(), numTermsProc.data(), displs.data(), MPI_DOUBLE,
               values.data(), numTermsA, MPI_DOUBLE, root, comm);
  extractRowBegin(Arows, numRows);
  if (sort_colinds) {
    printf( " ** Sorting column indexes for checking **\n" );
    std::vector<int>    columns_in(columns.size());
    std::vector<double> values_in(values.size());
    for (int i=0; i<columns.size(); i++) columns_in[i] = columns[i];
    for (int i=0; i<values.size(); i++) values_in[i] = values[i];

    std::vector<int> sortedIDs(numRows);
    std::vector<int> sortedCols(numRows);
    for (int i=0; i<numRows; i++) {
      const int index = rowBegin[i];
      const int num_cols = rowBegin[i+1] - rowBegin[i];

      sortedIDs.resize(num_cols);
      sortedCols.resize(num_cols);
      for (int j=0; j<num_cols; j++) sortedIDs[j] = j;
      for (int j=0; j<num_cols; j++) sortedCols[j] = columns_in[index+j];
      std::sort(sortedIDs.begin(), sortedIDs.end(),
                [&sortedCols](size_t i1, size_t i2) {return sortedCols[i1] < sortedCols[i2];});
      for (int j=0; j<num_cols; j++) columns[index+j] = columns_in[index+sortedIDs[j]];
      for (int j=0; j<num_cols; j++) values[index+j] = values_in[index+sortedIDs[j]];
    }
  }
  /*{
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    char filename[100];
    sprintf(filename, "lAin_%d.dat", myRank);
    FILE *fp = fopen(filename, "w");
    for (int i=0; i<numRows; i++) {
      for (int k=rowBegin[i]; k<rowBegin[i+1]; k++)
        fprintf(fp,"%d %d %.16e\n",i,columns[k],values[k]);
    }
    fclose(fp);
  }*/
}

void GenerateProblem::scatterRhs(const std::vector<int> & numRowsProc,
                                 const std::vector<double> & rhs_mm)
{
  int numRows;
  MPI_Scatter(numRowsProc.data(), 1, MPI_INT, &numRows, 1, MPI_INT, root, comm);
  rhs.resize(numRows);
  std::vector<int> displs(numProc, 0);
  if (myPID == root) {
    for (int i=1; i<numProc; i++) {
      displs[i] = displs[i-1] + numRowsProc[i-1];
    }
  }
  MPI_Scatterv(rhs_mm.data(), numRowsProc.data(), displs.data(), MPI_DOUBLE,
               rhs.data(), numRows, MPI_DOUBLE, root, comm);
}

void GenerateProblem::readMatrixMarket(const std::string & matrixFile,
                                       int & num_rows,
                                       std::vector<std::tuple<int,int,double>> & A) const
{
  std::ifstream fileMatrix(matrixFile);
  if (!fileMatrix.is_open()) {
    throw std::runtime_error("Error: Could not open file " + matrixFile);
  } else {
    if (msg_level > 0) std::cout << "Reading Matrix from " + matrixFile << std::endl;
  }
  std::string line;
  bool header_read = false;
  int num_cols, num_terms, index(0);
  while (std::getline(fileMatrix, line)) {
    // Ignore comment lines (lines starting with '%')
    if (line[0] == '%') {
      continue;
    }
    std::istringstream iss(line);
    // Read header (matrix dimensions and number of non-zero entries)
    if (!header_read) {
      iss >> num_rows >> num_cols >> num_terms;
      header_read = true;
      A.resize(num_terms);
      continue;
    }
    
    // Read matrix entries (row, column, value)
    int row, col;
    double value;
    iss >> row >> col >> value;
    A[index++] = {row-1, col-1, value};
  }
  fileMatrix.close();
}

void GenerateProblem::readMatrixRhs(const std::string & rhsFile,
                                    const int numRows,
                                    std::vector<double> & rhs_mm)
{
  std::ifstream fileRhs(rhsFile);
  if (!fileRhs.is_open()) {
    throw std::runtime_error("Error: Could not open file " + rhsFile);
  }
  std::string line;
  bool header_read = false;
  int num_rows, num_cols, index(0);
  while (std::getline(fileRhs, line)) {
    // Ignore comment lines (lines starting with '%')
    if (line[0] == '%') {
      continue;
    }
    std::istringstream iss(line);
    // Read header (matrix dimensions and number of non-zero entries)
    if (!header_read) {
      iss >> num_rows >> num_cols;
      ThrowAssert(num_rows == numRows, "inconsistent dimensions");
      ThrowAssert(num_cols == 1, "number of columns must be 1");
      header_read = true;
      rhs_mm.resize(num_rows);
      continue;
    }
    
    // Read matrix entries (row, column, value)
    double value;
    iss >> value;
    rhs_mm[index++] = value;
  }
  fileRhs.close();
}
 
void GenerateProblem::scaleDiagonal()
{
  const int numRows = rowBegin.size() - 1;
  for (int i=0; i<numRows; i++) {
    const int row = startGID + i;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      if (col == row) values[j] *= 1.01;
    }
  }
}

void GenerateProblem::scaleUpperTriangle()
{
  const double scale_factor = 0.95;
  const int numRows = rowBegin.size() - 1;
  for (int i=0; i<numRows; i++) {
    const int row = startGID + i;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      if (col > row) values[j] *= scale_factor;
    }
  }     
}

void GenerateProblem::removeSomeMatrixEntries(const int maxRemove)
{
  if (maxRemove == 0) return;
  // remove up to maxRemove entries in upper triangular part of matrix
  const int numRows = rowBegin.size() - 1;
  std::vector<int> rowBeginNew(numRows+1, 0);
  int numRemoved(0), numTerms(0);
  for (int i=0; i<numRows; i++) {
    const int row = startGID + i;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      bool keepEntry = true;
      if (col > row) {
        if (numRemoved < maxRemove) {
          keepEntry = false;
          numRemoved++;
        }
      }
      if (keepEntry) {
        columns[numTerms] = columns[j];
        values[numTerms++] = values[j];
      }
    }
    rowBeginNew[i+1] = numTerms;
  }
  columns.resize(numTerms);
  values.resize(numTerms);
  rowBegin = rowBeginNew;
}

void GenerateProblem::zeroSomeMatrixEntries(const int maxZero)
{
  if (maxZero == 0) return;
  // zero out up to maxZero entries in upper triangular part of matrix
  const int numRows = rowBegin.size() - 1;
  int numZero(0);
  for (int i=0; i<numRows; i++) {
    const int row = startGID + i;
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      if (col > row) {
        if (numZero < maxZero) {
          values[j] = 0;
          numZero++;
        }
      }
    }
  }
}

void GenerateProblem::makeAdjustments()
{
  if (add_nonsymmetry) {
    scaleUpperTriangle();
  }
  const int maxRemove = 1;
  if (remove_some_entries) {
    removeSomeMatrixEntries(maxRemove);
  }
  if (zero_some_entries) {
    zeroSomeMatrixEntries(maxRemove);
  }
}

 void GenerateProblem::readParameters(const std::string & inputFile)
{
  std::ifstream fin;
  fin.open(inputFile);
  if (fin.is_open() == false) {
    std::cout << "could not open file " << inputFile << std::endl;
    ThrowAssert(0, "error opening file");
  }
  if (myPID == root) {
    std::cout << std::endl << "Reading Example options from " << inputFile << std::endl;
  }
  numElemDir.resize(3);
  fin >> matrix_option;
  ThrowAssert((matrix_option >= 1) && (matrix_option <= 3),
              "matrix_option must be 1, 2, or 3");
  fin >> filenameBase;
  fin >> filenameBaseRhs;
  fin >> problem_type;
  fin >> numElemDir[0];
  fin >> numElemDir[1];
  fin >> numElemDir[2];  
  fin >> numProcSolver;
  fin >> msg_level;
  fin >> num_threads;
  fin >> matching_option;
  fin >> reorder_option;
  fin >> debugLevel;
  fin >> add_nonsymmetry;
  fin >> remove_some_entries;
  fin >> zero_some_entries;
  fin >> num_factorizations;
  fin >> output_elem_matrix;
  fin >> output_matrix;
  fin.close();
  if (numElemDir[2] == 0) numElemDir.resize(2);
}

int GenerateProblem::getMyPID()
{
  return myPID;
}

void GenerateProblem::getLengthAndDirectionCosines(const int node1,
                                                   const int node2,
                                                   const int dim,
                                                   const std::vector<double> & xElem,
                                                   const std::vector<double> & yElem,
                                                   const std::vector<double> & zElem,
                                                   double & length,
                                                   double* axial_dcs) const
{
  const double dx = xElem[node2] - xElem[node1];
  const double dy = yElem[node2] - yElem[node1];
  double length2 = dx*dx + dy*dy;
  double dz(0);
  if (dim == 3) {
    dz = zElem[node2] - zElem[node1];
    length2 += dz*dz;
  }
  length = std::sqrt(length2);
  axial_dcs[0] = dx/length;
  axial_dcs[1] = dy/length;
  axial_dcs[2] = dz/length;
}

int GenerateProblem::getNumNodeElem(const int dim) const
{
  if (dim == 2) return 4;
  else return 8;
}

std::vector<double> GenerateProblem::getElemMatrix(const int dim) const
{
  std::cout << "problem type = " << problem_type << std::endl;
  ThrowAssert((problem_type == "Scalar") || (problem_type == "Elasticity"),
              "problem_type must be Scalar or Elasticity");
  ThrowAssert((dim == 2) || (dim == 3), "dimension must be 2 or 3");
  int numDofNode = 1; // for Scalar problems
  if (problem_type == "Elasticity") numDofNode = dim;
  const int numNodeElem = getNumNodeElem(dim);
  const int numRowsElem = numNodeElem * numDofNode;
  std::vector<double> xElem, yElem, zElem;
  std::vector<double> elemMatrix(numRowsElem*numRowsElem, 0);
  if (dim == 2) {
    xElem = {0, 1, 1, 0};
    yElem = {0, 0, 1, 1};
  }
  else if (dim == 3) {
    xElem = {0, 1, 1, 0, 0, 1, 1, 0};
    yElem = {0, 0, 1, 1, 0, 0, 1, 1};
    zElem = {0, 0, 0, 0, 1, 1, 1, 1};
  }
  // add small random numbers to coordinates so that matrix is dense
  srand(7);
  for (size_t i=0; i<xElem.size(); i++) {
    xElem[i] += 0.007*rand()/RAND_MAX;
    yElem[i] += 0.007*rand()/RAND_MAX;
  }
  for (size_t i=0; i<zElem.size(); i++) {
    zElem[i] += 0.007*rand()/RAND_MAX;
  }
  double axial_dcs[3], delta[6];
  int indices[6];
  for (int j=0; j<numNodeElem; j++) {
    for (int i=j; i<numNodeElem; i++) {
      if (i != j) {
        double length;
        getLengthAndDirectionCosines(i, j, dim, xElem, yElem, zElem, length, axial_dcs);
        const double scaleFactor = 1/length;
        if (problem_type == "Scalar") {
          elemMatrix[i+i*numRowsElem] += scaleFactor;
          elemMatrix[i+j*numRowsElem] -= scaleFactor;
          elemMatrix[j+i*numRowsElem] -= scaleFactor;
          elemMatrix[j+j*numRowsElem] += scaleFactor;
        }
        else if (problem_type == "Elasticity") {
          for (int k=0; k<dim; k++) {
            indices[k] = j*dim + k;
            indices[k+dim] = i*dim + k;
            delta[k] = -axial_dcs[k];
            delta[k+dim] = axial_dcs[k];
          }
          for (int k=0; k<2*dim; k++) {
            const int row = indices[k];
            for (int m=0; m<2*dim; m++) {
              const int col = indices[m];
              elemMatrix[row+numRowsElem*col] += scaleFactor*delta[k]*delta[m];
            }
          }
        }
      }
    }
  }
  // add diagonal mass to remove rigid body modes
  const double massFactor = 0.01;
  for (int i=0; i<numRowsElem; i++) {
    const int index = i + numRowsElem*i;
    elemMatrix[index] += massFactor;
  }
  return elemMatrix;
}

void GenerateProblem::outputElemMatrix(const std::vector<double> & elemMatrix) const
{
  const int numRows = std::sqrt(elemMatrix.size());
  std::ofstream fout;
  fout.open("elemMatrix.dat");
  int index = 0;
  for (int j=0; j<numRows; j++) {
    for (int i=0; i<numRows; i++) {
      fout << i+1 << " " << j+1 << " ";
      fout << std::setw(23) << std::setprecision(16) << elemMatrix[index++]
           << std::endl;
    }
  }
}

void GenerateProblem::outputMatrix(const std::vector<std::tuple<int,int,double>> & A) const
{
  std::ofstream fout;
  fout.open("matrix.dat");
  for (int i=0; i<A.size(); i++) {
    fout << std::get<0>(A[i])+1 << " " << std::get<1>(A[i])+1 << " ";
    fout << std::setw(23) << std::setprecision(16) << std::get<2>(A[i])
         << std::endl;
  }
  fout.close();
}

void GenerateProblem::extractRowBegin(const std::vector<int> & Arows,
                                      const int numRows)
{
  rowBegin.resize(numRows+1, 0);
  int previous_row = -1;
  int index = 0;
  for (size_t i=0; i<Arows.size(); i++) {
    if (Arows[i] != previous_row) {
      rowBegin[index] = i;
      previous_row = Arows[i];
      index++;
    }
  }
  rowBegin[numRows] = Arows.size();
}

void GenerateProblem::getNumTermsProc(const std::vector<std::tuple<int,int,double>> & A,
                                      const std::vector<int> & numRowsProc,
                                      std::vector<int> & numTermsProc) const
{
  numTermsProc.resize(numProc, 0);
  int currentProc(0), maxRow(numRowsProc[0]);
  for (size_t i=0; i<A.size(); i++) {
    const int row = std::get<0>(A[i]);
    if (row < maxRow) {
      numTermsProc[currentProc]++;
    }
    else {
      currentProc++;
      maxRow += numRowsProc[currentProc];
      ThrowAssert(row < maxRow, "logic error extracting matrix");
      numTermsProc[currentProc]++;
    }
  }
}

void GenerateProblem::generateNodalCoordinates(std::vector<double> & x,
                                               std::vector<double> & y,
                                               std::vector<double> & z) const
{
  const int dim = numElemDir.size();
  const int numNodeX = numElemDir[0] + 1;
  const int numNodeY = numElemDir[1] + 1;
  if (dim == 2) {
    const int numNode = numNodeX * numNodeY;
    x.resize(numNode);
    y.resize(numNode);
    int index = 0;
    for (int j=0; j<numNodeY; j++) {
      for (int i=0; i<numNodeX; i++) {
        x[index] = i;
        y[index++] = j;
      }
    }
  }
  else if (dim == 3) {
    const int numNodeZ = numElemDir[2] + 1;
    const int numNode = numNodeX * numNodeY * numNodeZ;
    x.resize(numNode);
    y.resize(numNode);
    z.resize(numNode);
    int index = 0;
    for (int k=0; k<numNodeZ; k++) {
      for (int j=0; j<numNodeY; j++) {
        for (int i=0; i<numNodeX; i++) {
          x[index] = i;
          y[index] = j;
          z[index++] = k;
        }
      }
    }
  }
}

void GenerateProblem::generateElemConnectivity(std::vector<std::vector<int>> & elemConn) const
{
  const int dim = numElemDir.size();
  int numElem = numElemDir[0] * numElemDir[1];
  if (dim == 3) numElem *= numElemDir[2];
  elemConn.resize(numElem);
  const int numNodeX = numElemDir[0] + 1;
  const int numNodeY = numElemDir[1] + 1;
  int index = 0;
  if (dim == 2) {
    for (int j=0; j<numElemDir[1]; j++) {
      for (int i=0; i<numElemDir[0]; i++) {
        const int node1 = j*numNodeX + i;
        const int node2 = node1 + 1;
        const int node3 = node2 + numNodeX;
        const int node4 = node3 - 1;
        elemConn[index++] = {node1, node2, node3, node4};
      }
    }
  }
  else if (dim == 3) {
    const int numNodeLayer = numNodeX*numNodeY;
    for (int k=0; k<numElemDir[2]; k++) {
      const int start = k*numNodeLayer;
      for (int j=0; j<numElemDir[1]; j++) {
        for (int i=0; i<numElemDir[0]; i++) {
          const int node1 = start + j*numNodeX + i;
          const int node2 = node1 + 1;
          const int node3 = node2 + numNodeX;
          const int node4 = node3 - 1;
          const int node5 = node1 + numNodeLayer;
          const int node6 = node2 + numNodeLayer;
          const int node7 = node3 + numNodeLayer;
          const int node8 = node4 + numNodeLayer;
          elemConn[index++] = {node1, node2, node3, node4, node5, node6, node7, node8};
        }
      }
    }
  }
}

void GenerateProblem::generateMesh(std::vector<std::vector<int>> & elemConn,
                                   std::vector<double> & x,
                                   std::vector<double> & y,
                                   std::vector<double> & z) const
{
  generateElemConnectivity(elemConn);
  generateNodalCoordinates(x, y, z);
}

void GenerateProblem::generateMatrix(const std::vector<std::vector<int>> & elemConn,
                                     const std::vector<double> & elemMatrix,
                                     std::vector<std::tuple<int,int,double>> & A) const
{
  const int numElem = elemConn.size();
  const int numRows = std::sqrt(elemMatrix.size());
  const int numTermsA = numElem*elemMatrix.size();
  const int numNodePerElem = elemConn[0].size();
  const int numDofPerNode = numRows/elemConn[0].size();
  A.resize(numTermsA);
  int indexA = 0;
  for (int i=0; i<numElem; i++) {
    int index = 0;
    for (int k=0; k<numNodePerElem; k++) {
      const int node1 = elemConn[i][k];
      for (int kk=0; kk<numDofPerNode; kk++) {
        const int colGID = node1*numDofPerNode + kk;
        for (int j=0; j<numNodePerElem; j++) {
          const int node2 = elemConn[i][j];
          for (int jj=0; jj<numDofPerNode; jj++) {
            const int rowGID = node2*numDofPerNode + jj;
            A[indexA++] = {rowGID, colGID, elemMatrix[index++]};
          }
        }
      }
    }
  }
  std::sort(A.begin(), A.end());
  int prev_row = std::get<0>(A[0]);
  int prev_col = std::get<1>(A[0]);
  int index = 0;
  for (size_t i=1; i<A.size(); i++) {
    const int row = std::get<0>(A[i]);
    const int col = std::get<1>(A[i]);
    const double value = std::get<2>(A[i]);
    if ((row != prev_row) || (col != prev_col)) {
      prev_row = row;
      prev_col = col;
      index++;
      A[index] = {row, col, value}; 
    }
    else {
      const double curr_value = std::get<2>(A[index]);
      A[index] = {row, col, curr_value+value};
    }
  }
  A.resize(index+1);
  // check for no repeats
  for (size_t i=1; i<A.size(); i++) {
    const int row1 = std::get<0>(A[i-1]);
    const int col1 = std::get<1>(A[i-1]);
    const int row2 = std::get<0>(A[i]);
    const int col2 = std::get<1>(A[i]);
    ThrowAssert((row2 != row1) || (col2 != col1), "repeated matrix entries");
  }
  // remove zero entries
  int numTerms = 0;
  index = 0;
  for (size_t i=0; i<A.size(); i++) {
    const double val = std::get<2>(A[i]);
    if (val != 0) {
      const int row = std::get<0>(A[i]);
      const int col = std::get<1>(A[i]);
      A[numTerms++] = {row, col, val};
    }    
  }
  A.resize(numTerms);
}

void GenerateProblem::generateModelMatrix(std::vector<std::tuple<int,int,double>> & A)
{
  // generate matrix on proc 0
  const int dim = numElemDir.size();
  if (myPID == 0) {
    std::vector<std::vector<int>> elemConn;
    std::vector<double> x, y, z;
    auto elemMatrix = getElemMatrix(dim);
    if (output_elem_matrix) outputElemMatrix(elemMatrix);
    generateMesh(elemConn, x, y, z);
    generateMatrix(elemConn, elemMatrix, A);
    if (output_matrix) outputMatrix(A);
  }
}

void GenerateProblem::generate_rhs(const int numRows)
{
  rhs.resize(numRows);
  for (int i=0; i<numRows; i++) {
    rhs[i] = startGID + i;
  }
}
