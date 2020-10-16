#include "extra.h"


char* concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}


void Write2VTK_2D(const int n, real* p, const real h, const int timestep, int varNum)
{
    char fn[256];
    //sprintf(fn, "./output/V.%d.vtk", step); 

    sprintf(fn, "./output/var_%d_step_%d.vtk", varNum, timestep); 

    // n --- dim of 2D array with ghost cells

    // C code
    FILE* f = fopen(fn, "w");
    fprintf(f, "# vtk DataFile Version 3.0\nSolution\nASCII\nDATASET RECTILINEAR_GRID\n");
    fprintf(f, "DIMENSIONS %d %d 1\n", n + 1, n + 1);
    fprintf(f, "X_COORDINATES %d double\n", n + 1);

    int i;
    for (i = 0; i < n + 1; i++)
    {
        fprintf(f, "%.2e ", i*h);
    }
    fprintf(f, "\n");

    fprintf(f, "Y_COORDINATES %d double\n", n + 1);
    for (i = 0; i < n + 1; i++)
    {
        fprintf(f, "%.2e ", i*h);
    }
    fprintf(f, "\n");

    fprintf(f, "Z_COORDINATES 1 double\n0\n");
    fprintf(f, "CELL_DATA %d\n", (n * n));
    fprintf(f, "SCALARS V_membrane double\nLOOKUP_TABLE default\n");    
    
    int j;
    // output with ghost cells
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++)
            fprintf(f, "%.2e ",  p[j * n + i]);
        fprintf(f, "\n");
    }

    fclose(f);


/* C++ code
    std::fstream f(fn, std::ios::out);
    f << "# vtk DataFile Version 3.0" << std::endl;
    f << "Solution" << std::endl;
    f << "ASCII" << std::endl;
    f << "DATASET RECTILINEAR_GRID" << std::endl;
    f << "DIMENSIONS " << n + 1 << " " << n + 1 << " 1" << std::endl;
    f << "X_COORDINATES " << n + 1 << " double" << std::endl;
    for (int i = 0; i < n + 1; i++)
        f << i * h << " ";
    f << std::endl;
    f << "Y_COORDINATES " << n + 1 << " double" << std::endl;
    for (int i = 0; i < n + 1; i++)
        f << i * h << " ";
    f << std::endl;
    f << "Z_COORDINATES 1 double\n0" << std::endl;
    f << "CELL_DATA " << (n * n) << std::endl;
    f << "SCALARS V_membrane double\nLOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            f << p[j * n + i] << " ";
        f << std::endl;
    }
    f.close();
    */
}

void Write2VTK_2D_noGhosts(const int n, real* p, const real h, const int timestep, int varNum /* char* varName */)
{
    char fn[256];
    
    sprintf(fn, "./output/var_%d_noGhosts_step_%d.vtk", varNum, timestep); 

    // version 4 char* varName, passed as an arg. of the func
    //char* varNameFull = concat( concat("./output/", varName), "NoGhosts.");
    //sprintf(fn, concat(varNameFull, "%d.vtk"), step);

    // original version
    //sprintf(fn, "./output/VNoGhosts.%d.vtk", step); 

    // n --- dim of 2D array with ghost cells
    const int nNoGhosts = n - 2; // 1 ghost exluded from top, 1  from bottom; same with left and right

    // C code
    FILE* f = fopen(fn, "w");
    fprintf(f, "# vtk DataFile Version 3.0\nSolution\nASCII\nDATASET RECTILINEAR_GRID\n");
    fprintf(f, "DIMENSIONS %d %d 1\n", nNoGhosts + 1, nNoGhosts + 1);
    fprintf(f, "X_COORDINATES %d double\n", nNoGhosts + 1);

    int i;
    for (i = 0; i < nNoGhosts + 1; i++)
    {
        fprintf(f, "%.2e ", i*h);
    }
    fprintf(f, "\n");

    fprintf(f, "Y_COORDINATES %d double\n", nNoGhosts + 1);
    for (i = 0; i < nNoGhosts + 1; i++)
    {
        fprintf(f, "%.2e ", i*h);
    }
    fprintf(f, "\n");

    fprintf(f, "Z_COORDINATES 1 double\n0\n");
    fprintf(f, "CELL_DATA %d\n", (nNoGhosts * nNoGhosts));
    fprintf(f, "SCALARS V_membrane double\nLOOKUP_TABLE default\n");    
    
    int j;
    /* // output with ghost cells
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++)
            fprintf(f, "%.2e ",  p[j * n + i]);
        fprintf(f, "\n");
    }
    */

   // output without ghost cells
   for (j = 1; j < n - 1; j++) {
        for (i = 1; i < n - 1; i++)
            fprintf(f, "%.2e ",  p[j * n + i]);
        fprintf(f, "\n");
    }

    fclose(f);


/* C++ code
    std::fstream f(fn, std::ios::out);
    f << "# vtk DataFile Version 3.0" << std::endl;
    f << "Solution" << std::endl;
    f << "ASCII" << std::endl;
    f << "DATASET RECTILINEAR_GRID" << std::endl;
    f << "DIMENSIONS " << n + 1 << " " << n + 1 << " 1" << std::endl;
    f << "X_COORDINATES " << n + 1 << " double" << std::endl;
    for (int i = 0; i < n + 1; i++)
        f << i * h << " ";
    f << std::endl;
    f << "Y_COORDINATES " << n + 1 << " double" << std::endl;
    for (int i = 0; i < n + 1; i++)
        f << i * h << " ";
    f << std::endl;
    f << "Z_COORDINATES 1 double\n0" << std::endl;
    f << "CELL_DATA " << (n * n) << std::endl;
    f << "SCALARS V_membrane double\nLOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            f << p[j * n + i] << " ";
        f << std::endl;
    }
    f.close();
    */
}


real CalculateLinearInterpolate(real x, real xL, real xR, real yL, real yR)
{
    // calc value at point x \in [xL, xR]
    
    real a = (yR - yL) / (xR - xL); // just the derivative
    real b = yR - a*xR;
    return a*x + b;
}