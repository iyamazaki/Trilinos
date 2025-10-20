#pragma once

#include <iostream>
#include <vector>
#include "mkl.h"
#include "mkl_pardiso.h"

enum timingsLinSolver {SYMBOLIC, NUMERIC, REFACTOR, SOLVE, LENGTH_LINSOLVER_TIMINGS};

class sc_pardiso
{

public:

  sc_pardiso();

  ~sc_pardiso();

  void initialize(const int numRows,
                  int* rowBegin,
                  int* columns,
                  int numRowsB,
                  int* rowsB,
                  int msg_level,
                  int num_threads,
                  int reorder_option,
                  int structurally_symmetric,
                  int print_level);
  
  void factorize(double* values);
  
  std::vector<double> getSchurComplement() const;
  
  int cleanup();
  
  void setIparam(int idx, int ivalue);
  
  int getIparam(int idx) const;
  
  void setMatrixType(int matrixType);
  
  void setMessageLevel(int messageLevel);
  
  void refactorMatrix(double* values);
  
  int solve(double* rhs,
            double* sol,
            const int phase);
  
  std::vector<double> getTimings();
  
  int getNumRows()
  {
    return m_numRows;
  }
  
private:
  
  inline double clockIt();
  
  void isFailed(int ierr, const char* where) const;
  
  void set_parameters(const int matrix_type,
                      const int msg_level,
                      const int reorder_option);
  
  int analysis_phase();
  
  int numeric_phase();
  
  int m_numRows = 0;
  int m_print_level = 0;
  int *m_rowBegin, *m_columns;
  double *m_values;
  int m_numRowsB, *m_rowsB, m_phase=-1;

  long m_pt[64];
  int m_iparam[64];
  int m_matrixType;
  int m_msgLvl;
  std::vector<double> m_timings, m_schur;
  std::vector<int> m_perm;

};


