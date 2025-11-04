NVCC    = nvcc
CFLAGS  = -O3 --compiler-options -Wall -std=c++17 \
          -I../include -I..

LDFLAGS = -lcusolver -lcublas

SRC     = cu_solver.cu
OBJ     = $(SRC:.cu=.o)
TARGET  = cu_solver.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
