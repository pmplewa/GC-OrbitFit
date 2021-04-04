REB_DIR = ./rebound
REBX_DIR = ./reboundx
MULTINEST_DIR = ./MultiNest

export MPI = 1
export CC = mpicc
include $(REB_DIR)/src/Makefile.defs

all:
	@echo "compiling problem file ..."
	$(CC) -I$(REBX_DIR)/src/ -I$(REB_DIR)/src/ -I$(MULTINEST_DIR)/include/ -Wl,-rpath,./ $(OPT) $(PREDEF) problem.c -L. -lreboundx -lrebound -lmultinest_mpi $(LIB) -o sample
	@echo ""
	@echo "compiled successfully"
