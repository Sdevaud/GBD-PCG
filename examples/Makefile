NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../GLASS

examples: pcg0.exe pcg1.exe pcgT0.exe pcgT1.exe

pcg0.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=0 -DPCG_TYPE=false pcg_solve.cu -o pcg0.exe

pcg1.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=1 -DPCG_TYPE=false pcg_solve.cu -o pcg1.exe

pcgT0.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=0 -DPCG_TYPE=true pcg_solve.cu -o pcgT0.exe

pcgT1.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=15 -DPRECOND_POLY_ORDER=1 -DPCG_TYPE=true pcg_solve.cu -o pcgT1.exe

clean:
	rm -rf pcg0.exe pcg1.exe pcgT0.exe pcgT1.exe