# Compilers
CC      = gcc
MPICC   = mpicc

# Flags
CFLAGS  = -O3
OMPFLAG = -fopenmp
LIBS    = -lm

# Targets
all: serial openmp mpi

# Serial version
serial: rocket_serial.c rocket_physics.c rocket_utils.c
	$(CC) $(CFLAGS) rocket_serial.c rocket_physics.c rocket_utils.c -o rocket_serial $(LIBS)

# OpenMP version
openmp: rocket_openmp.c rocket_physics.c rocket_utils.c
	$(CC) $(CFLAGS) $(OMPFLAG) rocket_openmp.c rocket_physics.c rocket_utils.c -o rocket_openmp $(LIBS)

# MPI version
mpi: rocket_mpi.c rocket_physics.c rocket_utils.c
	$(MPICC) $(CFLAGS) rocket_mpi.c rocket_physics.c rocket_utils.c -o rocket_mpi $(LIBS)

# Clean compiled files
clean:
	rm -f rocket_serial rocket_openmp rocket_mpi