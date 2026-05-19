# Compilers
CC      = gcc
MPICC   = mpicc
MPICXX  = mpicxx
NVCC    = nvcc

# Flags
CFLAGS  = -O3
OMPFLAG = -fopenmp
LIBS    = -lm

# Targets
all: serial openmp mpi compare

# Serial version
serial: rocket_serial.c rocket_physics.c rocket_utils.c
	$(CC) $(CFLAGS) rocket_serial.c rocket_physics.c rocket_utils.c -o rocket_serial $(LIBS)

# OpenMP version
openmp: rocket_openmp.c rocket_physics.c rocket_utils.c
	$(CC) $(CFLAGS) $(OMPFLAG) rocket_openmp.c rocket_physics.c rocket_utils.c -o rocket_openmp $(LIBS)

# MPI version
mpi: rocket_mpi.c rocket_physics.c rocket_utils.c
	$(MPICC) $(CFLAGS) rocket_mpi.c rocket_physics.c rocket_utils.c -o rocket_mpi $(LIBS)

# Comprehensive comparison without CUDA. This works on any computer with MPI + OpenMP.
compare: compare.c rocket_physics.c rocket_utils.c
	$(MPICC) $(CFLAGS) $(OMPFLAG) compare.c rocket_physics.c rocket_utils.c -o compare $(LIBS)

# Comprehensive comparison with CUDA. If a CUDA GPU is not available at runtime,
# the CUDA test is skipped while Serial/OpenMP/MPI/Hybrid still run.
compare_cuda: compare.o rocket_physics.o rocket_utils.o rocket_cuda_kernel.o
	$(MPICXX) $(CFLAGS) $(OMPFLAG) compare.o rocket_physics.o rocket_utils.o rocket_cuda_kernel.o -o compare_cuda $(LIBS) -lcudart

compare.o: compare.c rocket_trajectory.h
	$(MPICC) $(CFLAGS) $(OMPFLAG) -DENABLE_CUDA -c compare.c -o compare.o

rocket_physics.o: rocket_physics.c rocket_trajectory.h
	$(MPICC) $(CFLAGS) -c rocket_physics.c -o rocket_physics.o

rocket_utils.o: rocket_utils.c rocket_trajectory.h
	$(MPICC) $(CFLAGS) -c rocket_utils.c -o rocket_utils.o

rocket_cuda_kernel.o: cuda/rocket_cuda_kernel.cu cuda/rocket_trajectory_cuda.h
	$(NVCC) $(CFLAGS) -I. -Icuda -c cuda/rocket_cuda_kernel.cu -o rocket_cuda_kernel.o

# Clean compiled files
clean:
	rm -f rocket_serial rocket_openmp rocket_mpi compare compare_cuda *.o
