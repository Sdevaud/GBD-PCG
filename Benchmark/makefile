NVCC    = nvcc
CFLAGS  = --compiler-options -Wall -O3 -I../include -I../GLASS
LDFLAGS = 

SRC     = Benchmark.cu
OBJ     = $(SRC:.cu=.o)
DEP     = $(OBJ:.o=.d)

# règle par défaut
all: run

# compilation de .cu en .o + génération des .d
%.o: %.cu
	$(NVCC) $(CFLAGS) -MMD -MP -c $< -o $@

# édition de lien
Benchmark.exe: $(OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# run dépend de l'exécutable
run: Benchmark.exe
	./Benchmark.exe

# nettoyage
clean:
	rm -f *.o *.d *.exe

# inclure les dépendances auto-générées
-include $(DEP)