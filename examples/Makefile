NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../GLASS

examples: pcg.exe pcg_block0.exe pcg_block1.exe

pcg.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=20 pcg_solve.cu -o pcg.exe

pcg_block0.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=20 -DPRECOND_POLY_ORDER=false pcg_block_solve.cu -o pcg_block0.exe

pcg_block1.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=20 -DSTATE_SIZE=20 -DPRECOND_POLY_ORDER=true pcg_block_solve.cu -o pcg_block1.exe

clean:
	rm -rf pcg.exe pcg_block0.exe pcg_block1.exe