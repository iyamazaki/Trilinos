



#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

#
# B) Define the header and source files (and include directories)
#

SET(HEADERS "")
SET(SOURCES "")

# Set the base Trilinos directory in order to find Trilinos_version.h

ASSERT_DEFINED(PROJECT_SOURCE_DIR)

TRIBITS_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

# Set headers (for IDEs and installing)

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

APPEND_SET(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
  )

# NOTE: Above, we add the XXX_config.h file so that MS Visual Studio
# will know to include this file in the dependency analysis

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  d3_solver.h
  gather_to_root_simple.h
  throwAssert.h
  )

APPEND_SET(SOURCES
  d3_solver.C
  gather_to_root_simple.C
  )

#
# C) Define the targets for package's test
#

#TRIBITS_ADD_TEST_DIRECTORIES(tests)
