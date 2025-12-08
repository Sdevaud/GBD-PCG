CXX     = g++
NVCC = nvcc

# flags compilation
CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
-I../../include \
-I../../GLASS \
-I../include

LDFLAGS = 

SRC     = benchmark_CG_no_GPU.cpp
OBJ     = $(SRC:.cpp=.o)
TARGET  = benchmark_CG_no_GPU.exe

# Run clean before building
all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.exe *.o