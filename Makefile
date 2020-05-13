all: compile_cpu run

compile_cpu:
	g++ -std=c++11 main.cpp -o ap_model

compile_for_gpu:
	pgc++ main.cpp -Minfo=accel -ta=nvidia -o ap_model_gpu

run:
	./ap_model

clear_output:
	rm output/*.vtk
