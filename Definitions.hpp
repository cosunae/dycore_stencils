#pragma once

//TODO mark these in namespace
const int cCacheFlusherSize = 1024*1024*21;
const int cNumBenchmarkRepetitions = 10;
#define N_HORIDIFF_VARS 4
#define PI ((Real)3.14159265358979323846) // pi
//#define SINGLEPRECISION
#ifdef SINGLEPRECISION
    typedef float Real;
    #define MPITYPE MPI_FLOAT
#else
    typedef double Real;
    #define MPITYPE MPI_DOUBLE
#endif

#define DTR_STAGE (Real)3.0/(Real)20.0

// define some physical constants
#define BETA_V ((double)0.0)
#define BET_M ((double)0.5 * ((double)1.0 - BETA_V))
#define BET_P ((double)0.5 * ((double)1.0 + BETA_V))

#ifdef __CUDACC__
#define GT_FUNCTION __host__ __device__ __forceinline__
#else
#define GT_FUNCTION inline
#endif
