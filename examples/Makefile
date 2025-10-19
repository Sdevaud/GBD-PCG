NVCC = nvcc
# if need to debug with compute-sanitizer, use -G option
CFLAGS = --compiler-options -Wall -O3 -I../include -I../GLASS

all: precond #no_precond
precond: precond_m0 precond_m1 precond_T_m0 precond_T_m1
#no_precond: no_precond_m0 no_precond_m1 no_precond_T_m0 no_precond_T_m1

precond_m0:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=0 -DPCG_TYPE=false pcg_precond.cu -o precond_m0.exe

precond_m1:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=1 -DPCG_TYPE=false pcg_precond.cu -o precond_m1.exe

precond_T_m0:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=0 -DPCG_TYPE=true pcg_precond.cu -o precond_T_m0.exe

precond_T_m1:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=1 -DPCG_TYPE=true pcg_precond.cu -o precond_T_m1.exe

# no_precond_m0:
# 	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=0 -DPCG_TYPE=false pcg_no_precond.cu -o no_precond_m0.exe

# no_precond_m1:
# 	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=1 -DPCG_TYPE=false pcg_no_precond.cu -o no_precond_m1.exe

# no_precond_T_m0:
# 	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=0 -DPCG_TYPE=true pcg_no_precond.cu -o no_precond_T_m0.exe

# no_precond_T_m1:
# 	$(NVCC) $(CFLAGS) -DKNOT_POINTS=11 -DSTATE_SIZE=29 -DPRECOND_POLY_ORDER=1 -DPCG_TYPE=true pcg_no_precond.cu -o no_precond_T_m1.exe

clean:
	rm -rf *.exe