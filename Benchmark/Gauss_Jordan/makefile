NVCC    = nvcc
CFLAGS  = -O3 --compiler-options -Wall -std=c++17 \
          -I../include -I..       # <-- inclut les headers communs
LDFLAGS = 

SRC     = benchmark_gauss_jordan.cu
OBJ     = $(SRC:.cu=.o)
TARGET  = benchmark_gauss_jordan.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe