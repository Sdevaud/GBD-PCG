CXX     = g++
NVCC    = nvcc

CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
-I../../include \
-I../../GLASS \
-I../include \
-I/usr/include/eigen3

LDFLAGS = 

SRC     = eigen.cpp
OBJ     = $(SRC:.cpp=.o)
TARGET  = eigen.exe

# Run clean before building
all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.exe *.o
