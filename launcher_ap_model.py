import subprocess
import timeit
import numpy as np
import sys
import numpy.random as rnd
#import resource
import os


variants = ['cpu', 'gpu']
timings = {variants[0]: [], variants[1]: []}
exe_files = {variants[0]: './ap_model_cpu', variants[1]: './ap_model_gpu'}



#num_launches = 1
Nxes = [int(2**n) for n in range(5, 11)]# (8, 15) --- original] # number of segments in x-dimension
T = 100. # endtime in (ms)

tmp_output_file = './.tmp_output.txt'

#rnd.seed() # with no arguments the seed is set to current wall time;
# for explicit setting use this: timeit.default_timer())
#serie_of_launches_num = rnd.randint(0, 100)

# loop for Nxes = 32, 64, 128, ..., values
Nxes = Nxes[0:-3]  # uncomment this if testing
for Nx in Nxes:
    print 'Num points in x-direction: %d; calculations begin...' % Nx
    num_segments_x = Nx - 1

    for mode in variants:
	subprocess.call( [exe_files[mode], str(num_segments_x), str(num_segments_x), str(T), str(tmp_output_file)] )#, str(serie_of_launches_num)] ) # Ny == Nx, for now
    	#info = resource.getrusage(resource.RUSAGE_CHILDREN)
    	#elapsed_time = timeit.default_timer() - start
    	#elapsed_time = info[0]
    	tmp_file = open(tmp_output_file, 'r') 
    	elapsed_time = tmp_file.read() 
    	timings[mode].append(float(elapsed_time))
	
    speedup_local = timings['cpu'][-1]/timings['gpu'][-1]
    print 'speedup local: %.7f' % speedup_local # 7 --- for detailed output
	
# final actions...
for mode in variants:
    timings[mode] = np.array(timings[mode])

speedups = timings['cpu']/timings['gpu']
print 'Speedups =', speedups

# writing the data to a file
output_2d = np.vstack((Nxes, speedups)).transpose()
current_dir_name = os.path.basename(os.path.normpath(os.getcwd())) # displays only the last 'part' of the current working dir
np.savetxt('speedups_%s.txt' % current_dir_name, output_2d, fmt='%.7f', delimiter=',') # 7 --- for detailed output
