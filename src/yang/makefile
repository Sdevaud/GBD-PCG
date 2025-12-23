NVCC = nvcc

# flags compilation
CFLAGS = --compiler-options -Wall -O3 -std=c++17 \
         -I./include \
         -I../../include

LDFLAGS =
SRC = yang_precond.cu yang_no_precond.cu
TARGETS = $(SRC:.cu=.exe)

# Run clean before building
all: clean $(TARGETS)

%.exe: %.o
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
