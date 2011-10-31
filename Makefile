srcdir       = .
builddir     = .
abs_builddir = /home/cummingb/downloads/sundials-2.4.0/src/nvec_par_cuda
top_builddir = ../..
prefix       = /opt/sundials2.4.0
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib

INSTALL        = /usr/bin/install -c
INSTALL_LIB    = ${INSTALL}
INSTALL_HEADER = ${INSTALL} -m 644

#MPICC       = /opt/intel/Compiler/11.1/072/bin/intel64/icc
MPICC       = /opt/intel/Compiler/11.1/072/bin/intel64/icc
MPI_INC_DIR = /opt/intel/impi/4.0.0.028/intel64/include
MPI_LIB_DIR = /opt/intel/impi/4.0.0.028/intel64/lib
CFLAGS      = -O3 -g
LIBS        = -lm -lcuda -lcublas

FCMIX_ENABLED = yes

top_srcdir = $(srcdir)/../..

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(MPI_INC_DIR)

NVECPAR_LIB       = libsundials_nvecparallel_cuda.la
NVECPAR_LIB_FILES = nvector_parallel_cuda.lo

mkinstalldirs = $(SHELL) $(top_srcdir)/config/mkinstalldirs
rminstalldirs = $(SHELL) $(top_srcdir)/config/rminstalldirs

all: libsundials_nvecparallel_cuda.a

libsundials_nvecparallel_cuda.a: nvector_parallel_cuda.o cuda_ops.o
	ar rvs libsundials_nvecparallel_cuda.a *.o 
	ranlib libsundials_nvecparallel_cuda.a

nvector_parallel_cuda.o: nvector_parallel_cuda.c
	$(MPICC) $(CFLAGS) $(MPI_FLAGS)  $(INCLUDES) -DMPICH_IGNORE_CXX_SEEK -c -lcublas -lmpi -I/usr/local/cuda/include -L/usr/local/cuda/lib64 nvector_parallel_cuda.c

cuda_ops.o: cuda_ops.h cuda_ops.cu
	/usr/local/cuda/bin/nvcc cuda_ops.cu -c -o cuda_ops.o -arch=compute_13

install: libsundials_nvecparallel_cuda.a
	cp ./libsundials_nvecparallel_cuda.a $(libdir)

clean:
	rm -f *.o
	rm -f *.a

