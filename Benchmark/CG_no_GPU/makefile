CXX     = g++

CFLAGS  = -O3 -Wall -std=c++17 \
          -I../include -I.. -I/usr/include/eigen3

LDFLAGS = 

SRC     = CG_no_GPU.cpp
OBJ     = $(SRC:.cpp=.o)
TARGET  = CG_no_GPU.exe

all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe