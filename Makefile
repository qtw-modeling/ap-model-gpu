all: compile_cpu compile_gpu

cpu:
	pgcc main_ap.c extra.c -o exe_ap_cpu

gpu:
	pgcc -acc -fast -Minfo=accel -ta=tesla main_ap.c extra.c -o exe_ap_gpu

#run:
#	./exec_br_gpu

clear_output:
	rm output/*.vtk
