# a simple makefile for building the sample program.
# I use multiple versions of gcc, but cuda only supports
# gcc 4.4 or lower. The ccbin command tells nvcc to use
# gcc 4.4 as its regular (non-gpu) compiler.
#
# the uncommented line should do the trick for you.

all: ../bin/topology

../bin/topology: ../build/bfs_kernel.o ../build/Graph.o ../build/TopologyGenerator.o ../src/main.cpp
	/usr/local/cuda-7.5/bin/nvcc -std=c++11 -arch=sm_20 $^ -o $@

../build/TopologyGenerator.o: ../src/TopologyGenerator.cpp
	g++ -std=c++0x -c $^

../build/Graph.o: ../src/Graph.cpp
	g++ -std=c++0x -c $^

../build/bfs_kernel.o: ../src/bfs_kernel.cu
	/usr/local/cuda-7.5/bin/nvcc -std=c++11 -arch=sm_20 -c $^ -o $@


clean:
	rm -f *.o
	rm -f *~
	rm -f ../bin/topology
