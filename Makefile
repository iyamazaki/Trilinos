# You must first set the TRILINOS_INSTALL_DIR variable.
#
# # Include Trilinos-related variables in your project.  If you only want
# # 1 package, replace “Trilinos” with the package’s name, e.g., “Epetra”.
#TRILINOS_INSTALL_DIR = ../../../../github/trilinos-install
#TRILINOS_INSTALL_DIR = ../../../../github/trilinos-install-gpu
include $(TRILINOS_INSTALL_DIR)/Makefile.export.Trilinos_install
#include Makefile.export.Trilinos
#
# # Add the Trilinos installation directory to the library and header search paths.
LIB_PATH = $(TRILINOS_INSTALL_DIR)/lib
INCLUDE_PATH = $(TRILINOS_INSTALL_DIR)/include $(CLIENT_EXTRA_INCLUDES)
#
# # Use the same C++ compiler, flags, & libraries that Trilinos uses.
CXX = $(Trilinos_CXX_COMPILER)

#CXXFLAGS = $(Trilinos_CXX_FLAGS) -std=c++11
CXXFLAGS = $(Trilinos_CXX_COMPILER_FLAGS) -std=c++11
CXXFLAGS = $(Trilinos_CXX_COMPILER_FLAGS)
LIBS = $(CLIENT_EXTRA_LIBS) $(SHARED_LIB_RPATH_COMMAND) \
       $(Trilinos_LIBRARIES) \
       $(Trilinos_TPL_LIBRARIES) \
       $(Trilinos_EXTRA_LD_FLAGS)

EXTRA_FLAGS = -I../src/kernels -I../src/operators -I../src/solvers
CXXFLAGS += -g

#
# Rules for building executables and objects.
%.exe : %.o $(EXTRA_OBJS)
	$(CXX) -o $@ $(LDFLAGS) $(CXXFLAGS) $< $(EXTRA_OBJS) -L$(LIB_PATH) $(LIBS)

%.o : %.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) -I$(INCLUDE_PATH) $(Trilinos_TPL_INCLUDE_DIRS) $(EXTRA_FLAGS) $<

clean:
	rm -f *.o *.exe


