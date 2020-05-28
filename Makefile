#all: compile_cpu run

compile_cpu:
	pgcc main_ap.c -o exec_ap_cpu

compile_gpu:
	pgcc main_ap.c -Minfo=accel -ta=nvidia -o exec_ap_gpu

#run:
#	./ap_model

clear_output:
	rm output/*.vtk
