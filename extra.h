#ifndef __EXTRA_H
#define __EXTRA_H

#include "common.h"
#include <stdlib.h>
#include <string.h>

char* concat(const char *s1, const char *s2);
void Write2VTK_2D(const int n, real* p, const real h, const int timestep, int varNum);
void Write2VTK_2D_noGhosts(const int n, real* p, const real h, const int timestep, int varNum /* char* varName */);
real CalculateLinearInterpolate(real x, real xL, real xR, real yL, real yR);

#endif __EXTRA_H