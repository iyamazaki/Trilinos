#include "sc_pardiso.h"
#include "throwAssert.h"
#include <sys/time.h>

sc_pardiso::sc_pardiso()
{
  m_timings.resize(LENGTH_LINSOLVER_TIMINGS, 0);
}

void sc_pardiso::isFailed(int ierr, const char* where) const
{
  if (ierr!=0) {
    std::cerr << "MKL Pardiso linear solver failed in the "
              << where << " phase.\n";
  }
}

inline double sc_pardiso::clockIt()
{
  struct timeval start;
  gettimeofday(&start, NULL);
  double duration = 
    (double)(start.tv_sec + start.tv_usec/1000000.0);
  return duration;
}

int sc_pardiso::cleanup()
{
  if ( m_phase <= 0 ) return 0;
  m_phase = -1;
  int one(1);
  int ierr = 0;
  double* x=nullptr, *rhs=nullptr;
  int n = m_numRows;
  pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
          &m_matrixType, &m_phase, &n, m_values, m_rowBegin, m_columns,
          m_perm.data(), &one, m_iparam, &m_msgLvl, rhs, x, &ierr);
  isFailed(ierr,"clean up");
  return ierr;
}

int sc_pardiso::solve(double* rhs,
                      double* sol,
                      const int phase)
{
  if (m_numRows == 0) return 0;
  int ierr = 0;
  m_phase = phase;
  int one(1);
  int nrhs(1);
  int n = m_numRows;
  double startTime = clockIt();
  pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
          &m_matrixType, &m_phase, &n, m_values, m_rowBegin, m_columns,
          m_perm.data(), &nrhs, m_iparam, &m_msgLvl, rhs, sol, &ierr);
  m_timings[SOLVE] += clockIt() - startTime;
  std::string text = "solve with phase = " + std::to_string(phase);
  isFailed(ierr, text.c_str());
  return ierr;
}

int sc_pardiso::numeric_phase()
{
  int ierr = 0;
  m_phase = 22;
  int one(1);
  int nrhs(1);
  int n = m_numRows;
  double ddum;

  double startTime = clockIt();
  pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
          &m_matrixType, &m_phase, &n, m_values, m_rowBegin, m_columns,
          m_perm.data(), &nrhs, m_iparam, &m_msgLvl, &ddum, m_schur.data(), &ierr);
  m_timings[NUMERIC] += clockIt() - startTime;

  isFailed(ierr, "numeric");
  return ierr;
}

int sc_pardiso::analysis_phase()
{
  int ierr = 0;
  m_phase = 11;
  int one(1);
  int n = m_numRows;
  m_perm.resize(n, 0);
  for (int i=0; i<m_numRowsB; i++) m_perm[m_rowsB[i]] = 1;
  double *rhs=nullptr, *x=nullptr, *values=nullptr;

  double startTime = clockIt();
  pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
          &m_matrixType, &m_phase, &n, values, m_rowBegin, m_columns,
          m_perm.data(), &one, m_iparam, &m_msgLvl, rhs, x, &ierr);
  m_timings[SYMBOLIC] += clockIt() - startTime;

  isFailed(ierr, "analysis");
  return ierr;
}

void sc_pardiso::factorize(double* values)
{
  m_values = values;
  if (m_print_level) std::cout << "performing MKL/Pardiso numeric phase" << std::endl;
  numeric_phase();
}

void sc_pardiso::initialize(const int numRows,
                            int* rowBegin,
                            int* columns,
                            int numRowsB,
                            int* rowsB,
                            int msg_level,
                            int num_threads,
                            int reorder_option,
                            int structurally_symmetric,
                            int print_level)
{
  mkl_set_num_threads(num_threads);
  m_numRows = numRows;
  m_rowBegin = rowBegin;
  m_columns = columns;
  m_numRowsB = numRowsB;
  m_rowsB = rowsB;
  m_schur.resize(numRowsB*numRowsB);
  m_print_level = print_level;
  // matrix_type = 1:  real and structurally symmetric
  // matrix_type = 11: real and nonsymmetric
  int matrix_type;
  if (structurally_symmetric) matrix_type = 1;
  else matrix_type = 11;
  set_parameters(matrix_type, msg_level, reorder_option);
  if (m_print_level) std::cout << "performing MKL/Pardiso analysis phase" << std::endl;
  analysis_phase();
  /*
  m_solver = new MklPardisoSolver();
  // use Metis nested dissection for fill-reducing ordering
  m_solver->params()->setIparam(1, 2);
  m_solver->params()->setMessageLevel(msgLevel);

  double startTime = clockIt();
  m_solver->wrapAnalysis(numRows, values, rowBegin, columns);
  m_timings[SYMBOLIC] += clockIt() - startTime;

  startTime = clockIt();
  m_solver->wrapNumericFactor();
  m_timings[NUMERIC] += clockIt() - startTime;
#else
  ThrowAssert(0, "Pardiso solver only available for Intel builds");
#endif
  */
}

sc_pardiso::~sc_pardiso()
{
  //  delete m_solver;
}

void sc_pardiso::set_parameters(const int matrix_type,
                                const int msg_level,
                                const int reorder_option)
{
  m_matrixType = matrix_type;
  m_msgLvl = msg_level;
  for (int i=0; i<64; i++) {
    m_iparam[i] = 0;
    m_pt[i] = 0;
  }
  // See intel documentation on pardiso for more information:
  m_iparam[0] = 1; // do not use solver defaults
  const bool valid_reorder = (reorder_option == 0) || (reorder_option == 2) ||
                             (reorder_option == 3);
  ThrowAssert(valid_reorder, "invalid reordering option");
  m_iparam[1] = reorder_option; // use Metis for reordering
  m_iparam[9] = 13; // pivoting option (default for nonsym, may not be used)
  m_iparam[10] = 0; // scaling option (default for nonsym, may not be used)
  m_iparam[17] = -1; // print matrix diagnostics
  m_iparam[26] = 1; // check the matrix (can turn off later)
  m_iparam[34] = 1; // 0-based indexing
  m_iparam[35] = 1; // calculate Schur complement
  m_iparam[59] = 0; // use in-core mode
}

std::vector<double> sc_pardiso::getSchurComplement() const
{
  return m_schur;
}

void sc_pardiso::setIparam(int idx, int ivalue)
{
   m_iparam[idx] = ivalue;
}


int sc_pardiso::getIparam(int idx) const
{
   return m_iparam[idx];
}

void sc_pardiso::setMatrixType(int matrixType)
{
  m_matrixType = matrixType;
}

void sc_pardiso::setMessageLevel(int messageLevel)
{
  m_msgLvl = messageLevel;
}

void sc_pardiso::refactorMatrix(double* values)
{
}

std::vector<double> sc_pardiso::getTimings()
{
  return m_timings;
}
