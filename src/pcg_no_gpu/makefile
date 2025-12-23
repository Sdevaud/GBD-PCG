CXX     = g++

CFLAGS  = -O3 -Wall -std=c++17 \
          -I../../include 

LDFLAGS = 

SRC     = pcg_no_gpu.cpp
OBJ     = $(SRC:.cpp=.o)
TARGET  = pcg_no_gpu.exe

all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe