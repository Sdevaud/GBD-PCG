NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../GLASS

examples: pcg.exe pcg_block.exe

pcg.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=100 -DSTATE_SIZE=15 pcg_solve.cu -o pcg.exe

pcg_block.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=100 -DSTATE_SIZE=15 pcg_block_solve.cu -o pcg_block.exe

clean:
	rm -rf pcg.exe pcg_block.exe