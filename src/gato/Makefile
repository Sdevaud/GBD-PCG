NVCC = nvcc

# flags compilation
CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
         -I./include \
         -I../../include

LDFLAGS = 
SRC     = gato.cu
TARGET  = gato.exe

# Run clean before building
all: clean $(TARGET)

%.exe: %.o
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
