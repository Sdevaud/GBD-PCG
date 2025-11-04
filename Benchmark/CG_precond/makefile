NVCC = nvcc

# flags compilation
CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
-DBENCHMARK=1 \
-DDEBUG=0 \
-DMEMPCY=0 \
-I../../include \
-I../../GLASS \
-I../include

LDFLAGS = 

SRC     = CG_precond.cu
OBJ     = $(SRC:.cu=.o)
TARGET  = CG_precond.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.exe *.o
