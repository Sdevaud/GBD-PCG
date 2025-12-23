NVCC = nvcc

CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
-I../../include \
-I../../GLASS \
-I../include

LDFLAGS = -lcudss -lcusolver -lcusparse -lcublas

SRC     = cu_solver.cu
OBJ     = $(SRC:.cu=.o)
TARGET  = cu_solver.exe

all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
