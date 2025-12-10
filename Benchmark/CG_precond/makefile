NVCC = nvcc

# flags compilation
CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
-I../../include \
-I../../GLASS \
-I../include

LDFLAGS = 

SRC     = CG_precond.cu
OBJ     = $(SRC:.cu=.o)
TARGET  = CG_precond.exe

# Run clean before building
all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.exe *.o