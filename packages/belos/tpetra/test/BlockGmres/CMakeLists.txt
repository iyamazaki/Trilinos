

ASSERT_DEFINED(Anasazi_SOURCE_DIR)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)
ASSERT_DEFINED(Tpetra_INST_FLOAT)

IF (${PACKAGE_NAME}_ENABLE_Triutils)
  IF (Tpetra_INST_FLOAT)

   TRIBITS_ADD_EXECUTABLE_AND_TEST(
     Tpetra_BlockGMRES_hb_DoubleFloat_test
     SOURCES test_bl_gmres_hb_df.cpp 
     ARGS "--verbose"
     COMM serial mpi
     )

  ENDIF()
ENDIF()

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_BlockGMRES_hb_test
  SOURCES test_bl_gmres_hb.cpp
  ARGS "--verbose"
  COMM serial mpi
  )

ASSERT_DEFINED(Belos_ENABLE_TSQR)
IF (Belos_ENABLE_TSQR)
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_BlockGMRES_hb_TSQR_test
  SOURCES test_bl_gmres_hb.cpp
  ARGS "--verbose --num-rhs=3 --block-size=3 --ortho-type=TSQR"
  COMM serial mpi
  )
ENDIF()

TRIBITS_ADD_EXECUTABLE(
   nonsymm_matrix
   SOURCES nonsymm_matrix.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_mvtimemataddmv
   SOURCES Trilinos_mpi_mvtimemataddmv.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_mvaxpy
   SOURCES Trilinos_mpi_mvaxpy.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_mvdot
   SOURCES Trilinos_mpi_mvdot.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_mvtransmv
   SOURCES Trilinos_mpi_mvtransmv.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_hh_lvl2_fivesynch
   SOURCES Trilinos_mpi_hh_lvl2_fivesynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_profile_hh_lvl2
   SOURCES Trilinos_mpi_profile_hh_lvl2.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_hh_lvl1_manysynch
   SOURCES Trilinos_mpi_hh_lvl1_manysynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_profile_hh_lvl1
   SOURCES Trilinos_mpi_profile_hh_lvl1.cpp
   COMM serial mpi
)


TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_hh_lvl2_foursynch
   SOURCES Trilinos_mpi_hh_lvl2_foursynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_cgs2_threesynch
   SOURCES Trilinos_mpi_cgs2_threesynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_profile_cgs2
   SOURCES Trilinos_mpi_profile_cgs2.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_cgs2d_twosynch
   SOURCES Trilinos_mpi_cgs2d_twosynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_dcgs2_onesynch
   SOURCES Trilinos_mpi_dcgs2_onesynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_profile_dcgs2
   SOURCES Trilinos_mpi_profile_dcgs2.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_mpi_cgs_twosynch
   SOURCES Trilinos_mpi_cgs_twosynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE(
   Trilinos_gpu_mpi_cgs2_threesynch
   SOURCES Trilinos_gpu_mpi_cgs2_threesynch.cpp
   COMM serial mpi
)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_BlockFGMRES_hb_test
  SOURCES test_bl_fgmres_hb.cpp
  ARGS "--verbose"
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_HybridGMRES_hb_test
  SOURCES test_hybrid_gmres_hb.cpp
  ARGS "--verbose --max-degree=10 --poly-type=Arnoldi"
       "--verbose --max-degree=10 --poly-type=Gmres"
       "--verbose --max-degree=10 --poly-type=Roots"
  COMM serial mpi
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(Tpetra_BlockGMRES_hb_CopyFiles
  SOURCE_DIR ${Anasazi_SOURCE_DIR}/testmatrices
  SOURCE_FILES bcsstk14.hb
  EXEDEPS Tpetra_BlockGMRES_hb_test
  )

