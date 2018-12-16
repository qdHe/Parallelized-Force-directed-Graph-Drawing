APP_NAME=omp

OBJS=omp.o

default: $(APP_NAME)

# Compile for Xeon Phi
$(APP_NAME): CXX = icc -m64 -std=c++11
$(APP_NAME): CXXFLAGS = -I. -O3 -Wall -openmp -offload-attribute-target=mic -DRUN_MIC

# Compile for CPU
cpu: CXX = g++ -m64 -std=c++11
cpu: CXXFLAGS = -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas

# Compilation Rules
$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

cpu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(APP_NAME) $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

submit:
	qsub omp.job
clean:
	/bin/rm -rf *~ *.o $(APP_NAME)

# For a given rule:
# $< = first prerequisite
# $@ = target
# $^ = all prerequisite

