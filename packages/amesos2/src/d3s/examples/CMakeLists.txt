
#
# Driver
#
TRIBITS_ADD_EXECUTABLE(
  d3_solver_driver
  SOURCES d3_solver_driver.C generate_problem.C
  COMM mpi
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(Amesos2_D3S_EXAMPLES_File
        SOURCE_FILES d3_driver.inp
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
)


