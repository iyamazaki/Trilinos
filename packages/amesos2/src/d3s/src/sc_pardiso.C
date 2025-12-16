#include "sc_pardiso.h"
#include "throwAssert.h"
#include <sys/time.h>

#include <mpi.h>

sc_pardiso::sc_pardiso()
{
  m_timings.resize(LENGTH_LINSOLVER_TIMINGS, 0);
}

void sc_pardiso::isFailed(int ierr, const char* where) const
{
  if (ierr!=0) {
    std::cerr << "MKL Pardiso linear solver failed in the "
              << where << " phase with ierr = " << ierr << ".\n";
    switch( ierr ){
    case -1:
      std::cerr << "PardisoMKL reported error: 'Input inconsistent'\n";
      break;
    case -2:
      std::cerr << "PardisoMKL reported error: 'Not enough memory'\n";
      break;
    case -3:
      std::cerr << "PardisoMKL reported error: 'Reordering problem'\n";
      break;
    case -4:
      std::cerr <<
        "PardisoMKL reported error: 'Zero pivot, numerical "
        "factorization or iterative refinement problem'\n";
      break;
    case -5:
      std::cerr << "PardisoMKL reported error: 'Unclassified (internal) error'\n";
      break;
    case -6:
      std::cerr << "PardisoMKL reported error: 'Reordering failed'\n";
      break;
    case -7:
      std::cerr << "PardisoMKL reported error: 'Diagonal matrix is singular'\n";
      break;
    case -8:
      std::cerr << "PardisoMKL reported error: '32-bit integer overflow problem'\n";
      break;
    case -9:
      std::cerr << "PardisoMKL reported error: 'Not enough memory for OOC'\n";
      break;
    case -10:
      std::cerr << "PardisoMKL reported error: 'Problems with opening OOC temporary files'\n";
      break;
    case -11:
      std::cerr << "PardisoMKL reported error: 'Read/write problem with OOC data file'\n";
      break;
    }
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

  if (m_sparse_schur) {
    m_iparam[35] = 2; // to skip Schur complement
  }
  double startTime = clockIt();
  pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
          &m_matrixType, &m_phase, &n, m_values, m_rowBegin, m_columns,
          m_perm.data(), &nrhs, m_iparam, &m_msgLvl, rhs, sol, &ierr);
  m_timings[SOLVE] += clockIt() - startTime;
//#define MATRIX_OUT
#ifdef MATRIX_OUT
  {
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    char filename[250];
    sprintf(filename,"RHS%d.dat", myRank);
    FILE *fp = fopen(filename, "w");
    for (int i=0; i<n; i++) {
      fprintf(fp,"%d %.16e %.16e\n",i,rhs[i],sol[i]);
    }
    fclose(fp);
  }
#endif
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
#ifdef MATRIX_OUT
  {
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    char filename[250];
    sprintf(filename,"A%d.dat", myRank);
    FILE *fp = fopen(filename, "w");
    for (int i=0; i<n; i++) {
      for (int k=m_rowBegin[i]; k<m_rowBegin[i+1]; k++)
        fprintf(fp,"%d %d %.16e\n",i,m_columns[k],m_values[k]);
    }
    fclose(fp);
  }
#endif

  double startTime = clockIt();
  if (m_sparse_schur) {
    m_iparam[10] = 0;
    m_iparam[12] = 0;
    m_iparam[23] = 1;  // two-level factorization algorithm
    m_iparam[35] = -1; //-2; // to compute Schur complement
    pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
            &m_matrixType, &m_phase, &n, m_values, m_rowBegin, m_columns,
            m_perm.data(), &nrhs, m_iparam, &m_msgLvl, &ddum, &ddum, &ierr);
  } else {
    pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
            &m_matrixType, &m_phase, &n, m_values, m_rowBegin, m_columns,
            m_perm.data(), &nrhs, m_iparam, &m_msgLvl, &ddum, m_schur.data(), &ierr);
  }
  m_timings[NUMERIC] += clockIt() - startTime;
#ifdef MATRIX_OUT
  {
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
      //printf("s=[\n");
      //for (int i=0; i<m_numRowsB; i++) {
      //  for (int j=m_schur_rowptr[i]; j<m_schur_rowptr[i+1]; j++) printf("%d %d %e",i,m_schur_colind[j],m_schur_values[j]);
      //}
      //printf("];\n");
      printf("S=[\n");
      for (int i=0; i<m_numRowsB; i++) {
        for (int j=0; j<m_numRowsB; j++) printf("%e ",m_schur[i+j*m_numRowsB]);
        printf("\n");
      }
      printf("];\n");
    }
  }
#endif

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

#ifdef MATRIX_OUT
  {
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    {
      char filename[250];
      sprintf(filename,"G%d.dat", myRank);
      FILE *fp = fopen(filename, "w");
      for (int i=0; i<n; i++) {
        for (int k=m_rowBegin[i]; k<m_rowBegin[i+1]; k++)
          fprintf(fp,"%d, %d %d\n",k, i,m_columns[k]);
      }
      fclose(fp);
      sprintf(filename,"i%d.dat", myRank);
      fp = fopen(filename, "w");
      for (int i=0; i<n; i++) {
        fprintf(fp,"%d, %d\n", i,m_perm[i]);
      }
      fclose(fp);
    }
  }
#endif
  if (m_debug_level > 0) {
    // sanity check to make sure each row of interior has at least one entry
    for (int i=0; i<n && ierr == 0; i++) {
      if (m_perm[i] == 0) {
        int nnz = 0;
        for (int k=m_rowBegin[i]; k<m_rowBegin[i+1] && nnz == 0; k++) {
          if (m_perm[m_columns[k]] == 0) nnz ++;
        }
        if (nnz == 0) ierr = -20;
      }
    }
    if (ierr != 0) {
      std::cerr << "SC_Pardiso: sanity check failed 'potential structurally-singular interior'\n";
      return ierr;
    }
  }
  double startTime = clockIt();
  if (m_sparse_schur) {
    m_iparam[10] = 0;
    m_iparam[12] = 0;
    m_iparam[23] = 1;  // two-level factorization algorithm
    m_iparam[35] = -1; //-2; // to compute Schur complement
    pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
            &m_matrixType, &m_phase, &n, values, m_rowBegin, m_columns,
            m_perm.data(), &one, m_iparam, &m_msgLvl, rhs, x, &ierr);
    isFailed(ierr, "analysis");
    if (ierr == 0) {
      m_schur_nnz = m_iparam[35];
      m_schur_rowptr.resize(1+m_numRowsB, 0);
      m_schur_colind.resize(m_schur_nnz, 0);
      m_schur_values.resize(m_schur_nnz, 0);
      if (m_debug_level > 0) {
        printf( " m_size = %d, m_schur_nnz = %d\n",m_numRowsB,m_schur_nnz );
      }
      int step = 1;
      //m_iparam[35] = -1; //-2; // to compute Schur complement
      pardiso_export((_MKL_DSS_HANDLE_t*)m_pt, m_schur_values.data(), m_schur_rowptr.data(), m_schur_colind.data(),
                     &step, m_iparam, &ierr);
      isFailed(ierr, "export");
    }
  } else {
    if (m_debug_level > 0) {
      printf( " m_size = %d /  %d\n",m_numRowsB,n );
    }
    pardiso((_MKL_DSS_HANDLE_t*)m_pt, &one, &one,
            &m_matrixType, &m_phase, &n, values, m_rowBegin, m_columns,
            m_perm.data(), &one, m_iparam, &m_msgLvl, rhs, x, &ierr);
  }
  m_timings[SYMBOLIC] += clockIt() - startTime;
#ifdef MATRIX_OUT
  {
    for (int i=n-m_numRowsB+1; i<n; i++) if( m_perm[i] < m_perm[i-1]) printf(" ** not sorted **\n");
    int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    {
      char filename[250];
      sprintf(filename,"p%d.dat", myRank);
      FILE* fp = fopen(filename, "w");
      for (int i=0; i<n; i++) {
        fprintf(fp,"%d, %d\n", i,m_perm[i]);
      }
      fclose(fp);
    }
  }
#endif
  return ierr;
}

int sc_pardiso::factorize(double* values, bool verbose)
{
  int r_val = -1;
  m_values = values;
  if (m_debug_level > 0 && verbose) {
    std::cout << "performing MKL/Pardiso numeric phase" << std::endl;
  }
  r_val = numeric_phase();
  return r_val;
}

int sc_pardiso::initialize(const int numRows,
                           int* rowBegin,
                           int* columns,
                           int numRowsB,
                           int* rowsB,
                           int msg_level,
                           int num_threads,
                           int reorder_option,
                           int structurally_symmetric,
                           bool robust_option,
                           int debug_level,
                           bool verbose)
{
  mkl_set_num_threads(num_threads);
  m_numRows = numRows;
  m_rowBegin = rowBegin;
  m_columns = columns;
  m_numRowsB = numRowsB;
  m_rowsB = rowsB;
  m_schur.resize(numRowsB*numRowsB, 0);
  m_debug_level = debug_level;
  // matrix_type = 1:  real and structurally symmetric
  // matrix_type = 11: real and nonsymmetric
  int r_val = -1;
  int matrix_type;
  if (structurally_symmetric && !robust_option) matrix_type = 1;
  else matrix_type = 11;
  set_parameters(matrix_type, msg_level, reorder_option, robust_option, verbose);
  if (m_debug_level > 0 && verbose) {
    std::cout << "performing MKL/Pardiso analysis phase" << std::endl;
  }

  r_val = analysis_phase();
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
  return r_val;
}

sc_pardiso::~sc_pardiso()
{
  //  delete m_solver;
}

void sc_pardiso::set_parameters(const int matrix_type,
                                const int msg_level,
                                const int reorder_option,
                                const bool robust,
                                const bool verbose)
{
  m_sparse_schur = false;
  m_matrixType = matrix_type;
  m_msgLvl = msg_level;
  for (int i=0; i<64; i++) {
    m_iparam[i] = 0;
    m_pt[i] = 0;
  }
  pardisoinit((_MKL_DSS_HANDLE_t*)m_pt, &m_matrixType, m_iparam);
  // See intel documentation on pardiso for more information:
  const bool valid_reorder = (reorder_option == 0) || (reorder_option == 2) ||
                             (reorder_option == 3);
  ThrowAssert(valid_reorder, "invalid reordering option");
  m_iparam[27] = 0; // double-precision
  m_iparam[34] = 1; // 0-based indexing
  m_iparam[35] = 2; // calculate Schur complement

  bool use_default = false; //true;
  if (!use_default) {
    m_iparam[0]  = 1; // do not use solver defaults
    m_iparam[1] = reorder_option; // use Metis for reordering
    m_iparam[9] = 13; // pivoting option (default for nonsym, may not be used)
    m_iparam[10] = 0; // scaling option (default for nonsym, may not be used)
    m_iparam[17] = -1; // print matrix diagnostics
    m_iparam[26] = 1; // check the matrix (can turn off later)
    m_iparam[59] = 0; // use in-core mode

    if (robust) {
      m_iparam[9] = 16; // pivoting option 
      m_iparam[10] = 1; // use scaling option
      m_iparam[12] = 1; // use weighted matchings
    }
  }
  if (m_debug_level > 0 && verbose) {
    std::cout << "setting MKL/Pardiso parameters (reorder_option = " << reorder_option << ")" << std::endl;
  }
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
