#ifndef SIMPLEX_CORE_H
#define SIMPLEX_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>

double hpc_gettime(void);

#define EPSILON 1e-10
#define PIVOT_TOL 1e-10
#define HARRIS_TOL 1e-5
#define PERTURB_EPS 1e-4
#define REFACTOR_INTERVAL 50
#define BLOCK_SIZE 256
#define TILE_SIZE 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef enum { OUTPUT_TEXT, OUTPUT_JSON, OUTPUT_CSV } OutputFormat;

typedef enum {
    CONSTRAINT_LE,
    CONSTRAINT_GE,
    CONSTRAINT_EQ
} ConstraintType;

typedef enum {
    MINIMIZE,
    MAXIMIZE
} OptSense;

typedef enum {
    OPTIMAL,
    INFEASIBLE,
    UNBOUNDED,
    TIMEOUT,
    ERROR
} SimplexStatus;

typedef struct {
    char name[256];
    int numVars;
    int numConstraints;
    double* objCoeffs;
    double** constraintMatrix;
    double* rhs;
    ConstraintType* constraintTypes;
    OptSense sense;
    double* lowerBounds;
    double* upperBounds;
    double objConstant;
    int* isInteger;
    double* rangeValues;
    char** varNames;
    char** constraintNames;
} LPProblem;

typedef struct {
    double* data;
    double* hostData;
    int rows;
    int cols;
    int* basicVars;
    int* hostBasicVars;
    int numOriginalVars;
    int numSlack;
    int numArtificial;
} Tableau;

typedef struct {
    int verbose;
    OutputFormat outputFormat;
    FILE* iterLog;
    int debug;
    int maxIter;
    double timeout;
} SolverConfig;

typedef struct {
    int phase;
    int totalIterations;
    double solveStartTime;
} RunContext;

extern SolverConfig g_config;
extern RunContext g_run;

#define g_verbose (g_config.verbose)
#define g_outputFormat (g_config.outputFormat)
#define g_iterLog (g_config.iterLog)
#define g_debug (g_config.debug)
#define g_maxIter (g_config.maxIter)
#define g_timeout (g_config.timeout)
#define g_phase (g_run.phase)
#define g_totalIterations (g_run.totalIterations)
#define g_solveStartTime (g_run.solveStartTime)

#endif
