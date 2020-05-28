#include "common.hpp"
 
#include <iostream>
#include <map>
#include <fstream>
#include <cstdlib>
#include <string>

// C-includes
#include <stdio.h>
#include <math.h>
#include <time.h>
//#include "openacc.h"
//#include <cuda_profiler_api.h>


using namespace std;

typedef double real;

//---------------------------------- grid parameters
// FOR DEBUG: 2x2 == 3 cells x 3 cells, with 1 active cell in center other cells being ghosts
#define numSegmentsX 30 // 20 
#define numSegmentsY 30 // 20
#define numPointsX (numSegmentsX + 1)
#define numPointsY (numSegmentsY + 1)
#define numPointsTotal (numPointsX * numPointsY)
#define hx 0.07 // same as for YNI model // (1./numSegmentsX)
#define hy 0.07 // same as for YNI model // (1./numSegmentsY)

#define SCALING_FACTOR 12.9
#define APD0 (330.) // in (ms)
#define LongitudeStim (APD0)
#define PeriodStim (LongitudeStim*2.) // in (ms) //APD0
//#define i_Stim0 (0.1/1) // for varying stim current value in various places of the programm


#define T 7000. // endtime in (ms)
#define dt 0.005 //0.005 // timestep in (ms)
//#define dtOutput 5. // in (ms)
// (dtOutput/dt) must be an integer number!

#define T_scaled (T/SCALING_FACTOR)
#define dt_scaled (dt/SCALING_FACTOR)

// model parameters (original)
#define k 8.
#define a 0.15
#define eps0 0.002
#define mu1 0.2
#define mu2 0.3

// tissue parameters
#define Dx 7e-1 //7e-3 // check if scaling needed for the value
#define Dy 7e-1  //7e-3 // check if scaling needed for the value

//int numSegmentsX, numSegmentsY;


// a class for storing the unknows cellwise
struct State {
    real u; // scaled membrane potential
    real v; // "slow" variable
};


void Write2VTK(std::string fileName, const int n, real *p, const real h, const int step)
{
    // C style
    //char fn[256];
    //sprintf(fn, "./output/yni.%d.vtk", step);

    // C++ style
    std::string fn = "./output/" + fileName;
    char fnEnding[256];
    sprintf(fnEnding, ".%d.vtk", step);
    std::string str_fnEnding = fnEnding;
    fn += str_fnEnding;

    std::fstream f(fn, std::ios::out);
    f << "# vtk DataFile Version 3.0" << std::endl;
    f << "Solution" << std::endl;
    f << "ASCII" << std::endl;
    f << "DATASET RECTILINEAR_GRID" << std::endl;
    f << "DIMENSIONS " << n + 1 << " " << n + 1 << " 1" << std::endl;
    f << "X_COORDINATES " << n + 1 << " double" << std::endl;
    for (int i = 0; i < n + 1; i++)
        //for (int i = 1; i < n; i++)
        f << i * h << " ";
    f << std::endl;
    f << "Y_COORDINATES " << n + 1 << " double" << std::endl;
    for (int i = 0; i < n + 1; i++)
        //for (int i = 1; i < n; i++)
        f << i * h << " ";
    f << std::endl;
    f << "Z_COORDINATES 1 double\n0" << std::endl;
    f << "CELL_DATA " << (n * n) << std::endl;
    //f << "CELL_DATA " << (n-2) * (n-2) << std::endl;
    f << "SCALARS V_membrane double\nLOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
            f << p[j * n + i] << " ";
        f << std::endl;
    }
    f.close();
}


int CalculateLinearCoordinate(int i, int j) {
    // "same comment as in the function below"
    return i + j*numPointsX;
}


#pragma acc routine //seq //inline
int CalculateLinearCoordinateGPU(int i, int j)//, int numPointsX) 
{
    // this way of access to GPU global memory is optimal (but i'm not shuar)
    return i + j*numPointsX;
}


real MembranePotentialReal(real u)
{
    // returns unscaled membrane potential using scaled value "u" 

    return 100.*u - 80.;
}


real TimeViaPhase(real phase_, real Period_, real t0_)
{
    /* phi = phase; Period = T; t0 = time of first depolarization. */

    return t0_ + Period_ / (2. * M_PI) * phase_;
}

#pragma acc routine //seq //inline
real i_Stim(int i, int j, real valueNonScaled)//, int numPointsX)
{
    // scaled stimulation current // 

    real scaling_factor2 = 12.9/100.;
    return (0.3)*scaling_factor2 * valueNonScaled; // 0.2 --- is "additional" amp.
}


#pragma acc routine //seq
//inline
real RhsU(real u, real v)
{
    return -k * u * (u - a) * (u - 1) - u * v;
}

#pragma acc routine //seq
//inline
real Eps(real u, real v)
{
    return eps0 + (mu1 * v) / (u + mu2);
}

#pragma acc routine //seq
//inline
real RhsV(real u, real v)
{
    return Eps(u, v) * (-v - k * u * (u - a - 1));
}


real FuncSinglePeriod(real t)
{
    // We consider onlt t > 0! //

    if (t < PeriodStim/2.)
            return 0.;
    else
        return 1.;
}


real IsStim(real t)
{
   
    // We consider only t > 0! //

    // Periodic stimulation, with each stimulation's duration = LongitudeStim
    real offset = (int)(t/PeriodStim) * PeriodStim;
    return FuncSinglePeriod(t - offset);
    
    //return 1.; // stimulus is always ON
}


// phase calculations
std::map<std::string, real> VviaPhase(real phase)
{
    // we dont need to perform memalloc for members within these structs
    State* Old = new State;
    State* New = new State;
    // initial conditions: only for Old structs: New will be calculated in the loop
    Old->u = 0.; // VRest;
    Old->v = 0.; // 0.067; //m_inf_CPU(VRest); // 0.5
    

    real THRESHOLD = -30.; //-65.; //-30; // hardcoded for now
    real tEndTransient = 5000.; // in (ms)
    real time0;           // random value
    real time1;           // random value
    bool isThresholdFound = false;
    //real PeriodStim = APD0; // in (ms); hardcoded for now

    real tCurrent = 0;
    int counter = 0;
    // TODO:
    // main loop: timestepping
    while (1)
    {

        //DEBUG();
        ///////////////// gating variables: ode ("reaction") step
        // TODO: make only ONE read of Old->V, etc. from memory; to more speedup, esp. for GPU
        New->v = Old->v + dt_scaled * RhsV(Old->u, Old->v);

        // membrane potential calc
        // variable u: "discrete diffusion" step + reaction step
        New->u = Old->u + dt_scaled * RhsU(Old->u, Old->v)
                        + dt_scaled 
                        * IsStim(tCurrent)
                        * i_Stim(0, 0, 10.); // logical operators --- account for a periodical stimulation

        //printf("uNew: %.2f\n", MembranePotentialReal(New->u));
        
        // when threshold is found
        if ((tCurrent >= tEndTransient) &&  (MembranePotentialReal(New->u) > THRESHOLD) && (MembranePotentialReal(Old->u) < THRESHOLD))
        {
            // when 2nd threshold time (t1) is found: set t1 and then exit the loop
            if (isThresholdFound == true)
            {
                //DEBUG();
                time1 = tCurrent; // nearest-neighbour interpolaion; change to linear!
                break;            // phase(V)
            }
            else // when threshold time (t0) is found: set t0
            {
                time0 = tCurrent; // nearest-neighbour interpolaion; change to linear!
                isThresholdFound = true;
                
                //DEBUG();
                
                //return ; // phase(V)
            }
        }

        tCurrent += dt;
        counter += 1;

        // swapping time layers
        State* tmp;
        tmp = Old;
        Old = New;
        New = tmp;

        //printf("Iteration #%d\n", counter);

    } // while

    //printf("t0 = %.2f, t1 = %.2f\n", time0, time1);
    //DEBUG();

    // set vars, calculated within the loop
    real period = (time1 - time0); //*0.5; // period of oscillations; remove "0.5" when period calc bug is found!
    //printf("First loop is finished; period of oscillations: %.2f ms\n", period);

    // repeat the loop (calculations) again and find V(phi)
    tCurrent = 0.; // again

    real tOfPhase = TimeViaPhase(phase, period, time0); // tOfPhase must be > tEndTransient
    //printf("Phase: %.2f, tOfPhase: %.2f\n", phase, tOfPhase);
    //std::cin.get();

    //real VOfPhase; // to be determined in the loop below
    std::map<std::string, real> stateOfPhase;

    // (again) initial conditions: only for Old structs: New will be calculated in the loop
    Old->u = 0.; // VRest;
    Old->v = 0.; //0.067; //m_inf_CPU(VRest); // 0.5
   

    // (again): main loop: timestepping
    while (1)
    {
        // it means, dat we found the moment of time, corresponding to the phase value
        if (tCurrent >= tOfPhase) // tOfPhase must be > tEndTransient
        {
            //VOfPhase = Old->V; // nearest-neighbour iterpolation; change to linear!
            stateOfPhase["u"] = Old->u;
            stateOfPhase["v"] = Old->v;
        
            break;
        }

        ///////////////// gating variables: ode ("reaction") step
        // TODO: make only ONE read of Old->V, etc. from memory; to more speedup, esp. for GPU
        New->v = Old->v + dt_scaled * RhsV(Old->u, Old->v);

        // membrane potential calc
        // variable u: "discrete diffusion" step + reaction step
        New->u = Old->u + dt_scaled * RhsU(Old->u, Old->v) 
                        + dt_scaled 
                         * IsStim(tCurrent)
                         * i_Stim(0, 0, 10.); // logical operators --- account for a periodical stim
        
        tCurrent += dt;
        //stepNumber += 1;

        // swapping time layers
        State *tmp;
        tmp = Old;
        Old = New;
        New = tmp;

    } // while

    //printf("Second loop is finished; VOfPhase: %.1f mV\n", VOfPhase);
    //std::cin.get();
    // "return" --- is within the loop (look up)
    return stateOfPhase; //VOfPhase;
}


void SetInitialConditions(real *u, real *v, real value)
{
    int idx;
    std::srand(unsigned(1.)); // initial seed for random number generator
    real randomNumber;


    for (int j = 0; j < numPointsY; j++)
        for (int i = 0; i < numPointsX; i++)
        {

            int idxCenter = CalculateLinearCoordinate(i, j);
            randomNumber = ((real)(std::rand() % 20)) / 20.; // 4phase setting

            // for phase calculation: using angle in polar coords
            real LTotal = numPointsX * hx;                        // should = numPointsY*hy
            real L = (j * hy + hy / 2.) - (numPointsY * hy / 2.); // LTotal/2. - (j*hx + hx/2.) --- old formula, when j-order was incorrect
            real lsmall = LTotal / 2. - ((numPointsX - 1 - i) * hx + hx / 2.);

            real phase = atan2(L, lsmall); // = angle in polar coords; use atan2() func instead of atan() !

            // check sign: atan2() returns vals. from [-pi, pi]
            if (phase < 0)
            {
                phase += 2 * M_PI;
            }

            //real phaseShifted = phase - M_PI/2.; // phase from R.Syunyaev article

            //printf("Phase = %.2f deg.\n", phase*180/M_PI);
            //std::cin.get();
            // TODO //////////////////////////////////////////////////////////////

            // the func returns a std::map of all the vars' values
            std::map<std::string, real> stateForPhase = VviaPhase(phase); // VviaPhase(phase);

            //printf("Phase: %.2f deg., VOfPhase = %.2f\n", phase*180./M_PI, stateForPhase["V"]);
            //std::cin.get();

            u[idxCenter] = stateForPhase["u"]; //VviaPhase(phase); //M_PI/12. //VRest;
            v[idxCenter] = stateForPhase["v"]; //0.067;//m_inf_CPU(VRest); // 0.5
            
            // for progress checking: in percents
            printf("Set. initial cond: %.2f percent completed\n",
                   100. * idxCenter / CalculateLinearCoordinate(numSegmentsX, numSegmentsY));
        }

    // after filling the whole area: "fill" borders wiht Neumann boundary cond.
    // the borders: Neumann boundary conditions
    for (int j = 0; j < numPointsY; j++)
        for (int i = 0; i < numPointsX; i++)
        {
            int idxCenter = CalculateLinearCoordinate(i, j);

            // borrder cells, including corner cells
            if (i == 0 || j == 0 || i == (numSegmentsX) || j == (numSegmentsY))
            {
                int idxNear;

                if ((i == 0)) //&& (j >= 1) && (j <= numSegmentsY - 1)) // left border, except for corner cells
                    idxNear = CalculateLinearCoordinate(i + 1, j);
                if ((j == 0)) //&& (i >= 1) && (i <= numSegmentsX - 1)) // bottom, except for corner cells
                    idxNear = CalculateLinearCoordinate(i, j + 1);
                if ((j == numSegmentsY)) // && (i >= 1) && (i <= numSegmentsX - 1)) // top, except for corner cells
                    idxNear = CalculateLinearCoordinate(i, j - 1);
                if ((i == numSegmentsX)) // && (j >= 1) && (j <= numSegmentsY - 1)) // right, except for corner cells
                    idxNear = CalculateLinearCoordinate(i - 1, j);

                // what about corner cells? for now, they are not treated (?)
                u[idxCenter] = u[idxNear];
                v[idxCenter] = v[idxNear];
            }
        }
}



int main(int argc, char** argv) {

    // UNCOMMENT WHEN USING GPU
    //clock_t start = clock(); // is this a better place to start the timing7
    //acc_set_device_num(1, acc_device_nvidia); // default device's number == 1

    // reading the params from the console
    //int numSegmentsX = atoi(argv[1]);
    //int numSegmentsY = atoi(argv[2]);
    //int serieOfLaunchesNum = atoi(argv[3]);
    //string tmp_output_file(argv[3]);
    
    //int numPointsX = numSegmentsX + 1;
    //int numPointsY = numSegmentsY + 1;

    //int numPointsTotal = numPointsX * numPointsY;
    // end UNCOMMENT WHEN USING GPU

    // allocating memory
    //State* old = new State[numPointsTotal];  //(state_ap*)calloc(numPointsTotal, sizeof(state_ap));
    //State* niu = new State[numPointsTotal];

    // gating vars
    real* uOld = new real[numPointsTotal];
    real* vOld = new real[numPointsTotal];

    real* uNew = new real[numPointsTotal];
    real* vNew = new real[numPointsTotal];

    //State* tmp; // for swapping xOld and xNew pointers (x = u, v)
    real *tmp; // for swapping xOld and xNew pointers (x = u, v)

    // setting a file for writing program's timing
    //char fileName[256];
    //sprintf(fileName, "timing_test_%d.txt", serieOfLaunchesNum);
    
    // for timing's output
    //ofstream timing;
    //timing.open(tmp_output_file.c_str());    

    
    // for setting IC using PHASE's calcs
    SetInitialConditions(uOld, vOld, 0.); // no "value"=0 is used for now
    
    // for manual setting IC for 3x3 cell tissue
    /*
    for (int idx = 0; idx < numPointsTotal; idx++)
    {
        uOld[idx] = 0;
        vOld[idx] = 0;
    }

    int idxCellCentral = CalculateLinearCoordinate(1, 1);
    uOld[idxCellCentral] = 0;
    vOld[idxCellCentral] = 0;
    */

    int counterOutput = 0;
    
    // we write initial condition in the timestepping loop
    //Write2VTK(numPointsX, old, hx, 0); // 0 --- output's file number

    // last preparations for the timestepping...
    real tCurrent = 0.;
    real tCurrent_scaled = 0.;
    int stepNumber = 0;
    //int counterOutput = 1;
    
    //real PeriodStim = APD0;

    printf("Timestepping begins...\n");
    clock_t start = clock(); // probably, this is not the correct place to start the timer!

    
#pragma acc data copy(old[0:numPointsTotal], niu[0:numPointsTotal]) \
deviceptr(tmp)
{
// main loop: timestepping
    while (tCurrent_scaled < T_scaled) 
    {

        
	
	#pragma acc parallel //present(old[0:numPointsTotal], niu[0:numPointsTotal])	
	{
	
	
	// TODO: change order of indexing (i, j)
        #pragma acc loop collapse(2) independent	
	for (int j = 0; j < numPointsY; j++)
	    for (int i = 0; i < numPointsX; i++) 
        {

                int idxCenter = CalculateLinearCoordinateGPU(i, j);

                //uNew[idxCenter] = uOld[idxCenter];
		        //vNew[idxCenter] = vOld[idxCenter];
		    
		    

                // inner cells
                if (i >= 1 && j >= 1 && i <= (numSegmentsX - 1) && j <= (numSegmentsY - 1))
                {
                    // for short names
                    int idxUp = CalculateLinearCoordinateGPU(i, j + 1);
                    int idxDown = CalculateLinearCoordinateGPU(i, j - 1);
                    int idxLeft = CalculateLinearCoordinateGPU(i - 1, j);
                    int idxRight = CalculateLinearCoordinateGPU(i + 1, j);

                    ///////////////// slow variable "v": ode step
                    vNew[idxCenter] = vOld[idxCenter] + dt_scaled * RhsV(uOld[idxCenter], vOld[idxCenter]);

                    
		            // variable u: "discrete diffusion" step + reaction step
                    uNew[idxCenter] = uOld[idxCenter] + dt_scaled * (Dx * (uOld[idxRight] - 2 * uOld[idxCenter] + uOld[idxLeft]) 
                    + Dy * (uOld[idxUp] - 2 * uOld[idxCenter] + uOld[idxDown])) 
                    + dt_scaled * RhsU(uOld[idxCenter], vOld[idxCenter]) 
                    + dt_scaled
                    //* (tCurrent > 4700) // for defibrillation
                    * 0. * IsStim(tCurrent)
                    * i_Stim(i, j, 10); // logical operators --- account for a periodical stim

                } // if

                // the borders: Neumann boundary conditions
                else
                {
                    int idxNear;

                    if ((i == 0) && (j >= 1) && (j <= numSegmentsY - 1)) // left border, except for corner cells
                        idxNear = CalculateLinearCoordinateGPU(i + 1, j);
                    else if ((j == 0) && (i >= 1) && (i <= numSegmentsX - 1)) // bottom, except for corner cells
                        idxNear = CalculateLinearCoordinateGPU(i, j + 1);
                    else if ((j == numSegmentsY) && (i >= 1) && (i <= numSegmentsX - 1)) // top, except for corner cells
                        idxNear = CalculateLinearCoordinateGPU(i, j - 1);
                    else if ((i == numSegmentsX) && (j >= 1) && (j <= numSegmentsY - 1)) // right, except for corner cells
                        idxNear = CalculateLinearCoordinateGPU(i - 1, j);
                    else
                    {             // if corner cell
                        continue; // do nothing, continue the "i,j" loop
                    }

                    // what about corner cells? for now, they are not treated (?)
                    // Neumann boundary cond setting
                    uNew[idxCenter] = uNew[idxNear];
                    vNew[idxCenter] = vNew[idxNear];
                }
                
        } // for i
	} // acc parallel

    // output
    if ( stepNumber % 2000 == 0) // output each 10 msec: 10/dt = 2000
    {
    //if ((stepNumber % (int)(T/100/dt)) == 0) { // 1e3 --- the number for T = 600 ms; 20 --- the number for T = 10 ms
        #pragma acc update host(uOld[0:numPointsTotal])
                Write2VTK("V", numPointsX, uOld, hx, stepNumber); // for now: numPointsX == numPointsY
                
                printf("%.2f percent is completed\n", 100*tCurrent_scaled/T_scaled);
                counterOutput += 1;
	}

    tCurrent += dt;
    tCurrent_scaled += dt_scaled;
    stepNumber += 1;

    // swapping time-layers
    ////// swap V
    tmp = uOld; uOld = uNew; uNew = tmp;
    ///// swap m
    tmp = vOld; vOld = vNew; vNew = tmp;

    

    } // while T_scaled

} // acc data

    real elapsedTime = (real)( ((real)(clock() - start))/CLOCKS_PER_SEC );
    printf("\nCalculations finished. Elapsed time = %.2e sec\n", elapsedTime);
    
    //timing << elapsedTime;
    //timing.close();

    // cleaning up
    delete[] uOld;
    delete[] vOld;
    delete[] uNew;
    delete[] vNew;
    //delete[] old; //uOld;
    //delete[] niu; //uNew;

    //cudaProfilerStop();

    return 0;
}
