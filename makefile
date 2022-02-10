CC = mpic++
CCFLAGS = -std=c++14 -Wall -Ofast -march=native -frename-registers -funroll-loops
OPENMP = -fopenmp
LDFLAGS = #-lfftw3
LIBS = #
INCLUDES = #

TARGETS = main
SRCS = analyzers.cpp init_system.cpp mc_moves.cpp membrane_mc.cpp neighborlist.cpp output_system.cpp run_mc.cpp simulation.cpp utilities.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: depend clean

all: $(TARGETS)

main: $(OBJS)
	$(CC) $(OBJS) -o $@ $(CCFLAGS) $(OPENMP) $(LDFLAGS) $(LIBS) $(INCLUDES)

.cpp.o:
	$(CC) -c $< -o $@ $(CCFLAGS) $(OPENMP) $(LDFLAGS) $(LIBS) $(INCLUDES)

clean:
	rm -f $(OBJS) $(TARGETS)

depend: $(SRCS)
	makedepend -fmakefile $(INCLUDES) $^

