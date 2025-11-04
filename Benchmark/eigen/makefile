CXX     = g++
NVCC    = nvcc

CFLAGS  = -O3 -Wall -std=c++17 \
          -I../include -I.. -I/usr/include/eigen3

SRC     = eigen.cpp
OBJ     = $(SRC:.cpp=.o)
TARGET  = eigen.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
