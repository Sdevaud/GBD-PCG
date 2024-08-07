NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../GLASS

examples: pcg0.exe pcg1.exe pcg_trans0.exe pcg_trans1.exe

pcg0.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=0 pcg_solve.cu -o pcg0.exe

pcg1.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=1 pcg_solve.cu -o pcg1.exe

pcg_trans0.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=0 pcg_trans_solve.cu -o pcg_trans0.exe

pcg_trans1.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=1 pcg_trans_solve.cu -o pcg_trans1.exe

clean:
	rm -rf pcg0.exe pcg1.exe pcg_trans0.exe pcg_trans1.exe