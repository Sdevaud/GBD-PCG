NVCC    = nvcc

# Includes + warnings + optimisations
CFLAGS  = --compiler-options -Wall -O3 -I../include -I../GLASS

# Linking avec cuSOLVER + cuSPARSE + CUDA runtime
# Si ton CUDA n’est pas dans /usr/local/cuda, décommente la ligne -L
LDFLAGS = \
    -lcusolver \
    -lcusparse \
    -lcudart
#   -L/usr/local/cuda/lib64   # <-- décommente si nécessaire

SRC     = Benchmark.cu
OBJ     = $(SRC:.cu=.o)
DEP     = $(OBJ:.o=.d)

# Règle par défaut
all: run

# Compilation des .cu en .o avec génération des .d
%.o: %.cu
	$(NVCC) $(CFLAGS) -MMD -MP -c $< -o $@

# Edition de lien -> création de Benchmark.exe
Benchmark.exe: $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Exécution
run: Benchmark.exe
	./Benchmark.exe

# Nettoyage
clean:
	rm -f *.o *.d *.exe

# Inclusion des dépendances générées
-include $(DEP)
