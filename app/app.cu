#include "app.h"
#include "../parser/parser.h"
#include "../solver/solver.h"
#include "../io/io.h"
#include "../hpc.h"

// ===========================================================================

static int hasFileExtensionIgnoreCase(const char* filename, const char* extension) {
    if (!filename || !extension) return 0;
    size_t n = strlen(filename);
    size_t e = strlen(extension);
    if (n < e) return 0;

    const char* tail = filename + (n - e);
    for (size_t i = 0; i < e; i++) {
        char a = tail[i];
        char b = extension[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static int isSupportedInputFile(const char* filename) {
    return hasFileExtensionIgnoreCase(filename, ".mps") ||
           hasFileExtensionIgnoreCase(filename, ".dat");
}

/**
 * Solve a single file (shared logic for interactive + normal mode).
 * Returns the exit code (0 = OPTIMAL, 1 = other).
 */
int solveFile(const char* filename, cudaDeviceProp* prop, SolverConfig* config, RunContext* run) {
    LPProblem* lp = parseLP(filename, config);
    if (!lp) return EXIT_FAILURE;
    
    preprocessBounds(lp, config);
    Tableau* tab = createTableau(lp, config);
    
    run->totalIterations = 0;
    double tstart = hpc_gettime();
    SimplexStatus status = solveSimplex(tab, lp, config, run);
    double elapsed = hpc_gettime() - tstart;
    
    outputSolution(tab, lp, status, elapsed, config, run);
    
    freeTableau(tab);
    freeLPProblem(lp);
    return (status == OPTIMAL) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/**
 * Interactive REPL.
 * Commands: help, quit/exit, set <option> <value>, or a filename to solve.
 */
void interactiveMode(cudaDeviceProp* prop, SolverConfig* config, RunContext* run) {
    char line[1024];
    
    printf("CUDA Simplex — Interactive Mode\n");
    printf("Type a .mps or .dat filename to solve, or 'help' for commands.\n\n");
    
    while (1) {
        printf("simplex> ");
        fflush(stdout);
        
        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;  // EOF
        }
        
        // Strip trailing newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        
        // Skip empty lines
        if (len == 0) continue;
        
        // Commands
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) {
            break;
        } else if (strcmp(line, "help") == 0) {
            printf("Commands:\n");
            printf("  <file.mps|file.dat>     Solve an LP file\n");
            printf("  set verbose <0|1|2>     Set verbosity level\n");
            printf("  set debug <0|1>         Toggle tableau printing\n");
            printf("  set format <text|json|csv>  Output format\n");
            printf("  set maxiter <N>         Max iterations\n");
            printf("  set timeout <seconds>   Solve timeout (0=off)\n");
            printf("  status                  Show current settings\n");
            printf("  quit / exit             Exit\n");
        } else if (strcmp(line, "status") == 0) {
            printf("  verbose  = %d\n", config->verbose);
            printf("  debug    = %d\n", config->debug);
            printf("  format   = %s\n",
                   config->outputFormat == OUTPUT_JSON ? "json" :
                   config->outputFormat == OUTPUT_CSV  ? "csv"  : "text");
            printf("  maxiter  = %d\n", config->maxIter);
            printf("  timeout  = %.1f s\n", config->timeout);
            printf("  device   = %s\n", prop->name);
        } else if (strncmp(line, "set ", 4) == 0) {
            char key[64], val[64];
            if (sscanf(line + 4, "%63s %63s", key, val) == 2) {
                if (strcmp(key, "verbose") == 0) {
                    config->verbose = atoi(val);
                    printf("verbose = %d\n", config->verbose);
                } else if (strcmp(key, "debug") == 0) {
                    config->debug = atoi(val);
                    printf("debug = %d\n", config->debug);
                } else if (strcmp(key, "format") == 0) {
                    if (strcmp(val, "json") == 0)      config->outputFormat = OUTPUT_JSON;
                    else if (strcmp(val, "csv") == 0)  config->outputFormat = OUTPUT_CSV;
                    else                                 config->outputFormat = OUTPUT_TEXT;
                    printf("format = %s\n", val);
                } else if (strcmp(key, "maxiter") == 0) {
                    config->maxIter = atoi(val);
                    printf("maxiter = %d\n", config->maxIter);
                } else if (strcmp(key, "timeout") == 0) {
                    config->timeout = atof(val);
                    printf("timeout = %.1f s\n", config->timeout);
                } else {
                    printf("Unknown option: %s\n", key);
                }
            } else {
                printf("Usage: set <option> <value>\n");
            }
        } else {
            // Treat as filename
            struct stat st;
            if (stat(line, &st) != 0) {
                printf("File not found: %s\n", line);
                continue;
            }
            solveFile(line, prop, config, run);
        }
    }
}

void printUsage(const char* progName) {
    fprintf(stderr, "CUDA Two-Phase Simplex Solver\n\n");
    fprintf(stderr, "Usage: %s [options] <problem.{mps,dat}> [...]\n\n", progName);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s, --silent         Suppress solver output\n");
    fprintf(stderr, "  -d, --debug          Print initial, intermediate, and final tableaux\n");
    fprintf(stderr, "  --diag               Enable diagnostic output (verbose=2)\n");
    fprintf(stderr, "  -i, --interactive    Interactive REPL mode\n");
    fprintf(stderr, "  -m, --max-iter <N>   Set maximum iterations (default: 50000)\n");
    fprintf(stderr, "  -t, --timeout <sec>  Set solve timeout in seconds (0=off)\n");
    fprintf(stderr, "  --json               Output solution in JSON format\n");
    fprintf(stderr, "  --csv                Output solution in CSV format\n");
    fprintf(stderr, "  --batch              Batch mode: solve multiple files, print summary\n");
    fprintf(stderr, "  --log <file>         Write per-iteration log to CSV file\n");
    fprintf(stderr, "  -h, --help           Show this help message\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s problem.mps\n", progName);
    fprintf(stderr, "  %s problem.dat\n", progName);
    fprintf(stderr, "  %s -d problem.mps              # show all tableaux\n", progName);
    fprintf(stderr, "  %s -m 1000 -t 5 problem.dat    # max 1000 iters, 5s timeout\n", progName);
    fprintf(stderr, "  %s -i                           # interactive mode\n", progName);
    fprintf(stderr, "  %s --json problem.mps\n", progName);
    fprintf(stderr, "  %s --batch netlib/*.mps\n", progName);
    fprintf(stderr, "  %s --batch Dati-LP/*.dat\n", progName);
    fprintf(stderr, "  %s --batch netlib/\n", progName);
    fprintf(stderr, "  %s --log iter.csv problem.dat\n", progName);
}


int runApp(int argc, char* argv[]) {
    SolverConfig config = {1, OUTPUT_TEXT, NULL, 0, 50000, 0.0};
    RunContext run = {0, 0, 0.0};

    // Parse flags
    int batchMode = 0;
    int interactiveFlag = 0;
    const char* logFile = NULL;
    int inputCount = 0;
    const char** inputFiles = (const char**)malloc(argc * sizeof(const char*));
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--silent") == 0) {
            config.verbose = 0;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            config.debug = 1;
        } else if (strcmp(argv[i], "--diag") == 0) {
            config.verbose = 2;
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            interactiveFlag = 1;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--max-iter") == 0) {
            if (i + 1 < argc) {
                config.maxIter = atoi(argv[++i]);
                if (config.maxIter <= 0) { fprintf(stderr, "Error: --max-iter must be positive\n"); free(inputFiles); return EXIT_FAILURE; }
            } else {
                fprintf(stderr, "Error: --max-iter requires an integer argument\n");
                free(inputFiles); return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) {
            if (i + 1 < argc) {
                config.timeout = atof(argv[++i]);
                if (config.timeout < 0.0) { fprintf(stderr, "Error: --timeout must be non-negative\n"); free(inputFiles); return EXIT_FAILURE; }
            } else {
                fprintf(stderr, "Error: --timeout requires a numeric argument\n");
                free(inputFiles); return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--json") == 0) {
            config.outputFormat = OUTPUT_JSON;
        } else if (strcmp(argv[i], "--csv") == 0) {
            config.outputFormat = OUTPUT_CSV;
        } else if (strcmp(argv[i], "--batch") == 0) {
            batchMode = 1;
        } else if (strcmp(argv[i], "--log") == 0) {
            if (i + 1 < argc) {
                logFile = argv[++i];
            } else {
                fprintf(stderr, "Error: --log requires a filename argument\n");
                free(inputFiles);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            free(inputFiles);
            return EXIT_SUCCESS;
        } else if (argv[i][0] != '-') {
            inputFiles[inputCount++] = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            free(inputFiles);
            return EXIT_FAILURE;
        }
    }
    
    // In batch mode, auto-expand directory arguments into .mps/.dat files
    if (batchMode) {
        int expandedCount = 0;
        const char** expandedFiles = (const char**)malloc(4096 * sizeof(const char*));
        
        for (int i = 0; i < inputCount; i++) {
            struct stat st;
            if (stat(inputFiles[i], &st) == 0 && S_ISDIR(st.st_mode)) {
                DIR* dir = opendir(inputFiles[i]);
                if (dir) {
                    struct dirent* entry;
                    while ((entry = readdir(dir)) != NULL) {
                        if (isSupportedInputFile(entry->d_name)) {
                            char* fullpath = (char*)malloc(512);
                            snprintf(fullpath, 512, "%s/%s", inputFiles[i], entry->d_name);
                            expandedFiles[expandedCount++] = fullpath;
                        }
                    }
                    closedir(dir);
                }
            } else {
                expandedFiles[expandedCount++] = inputFiles[i];
            }
        }
        
        free(inputFiles);
        inputFiles = expandedFiles;
        inputCount = expandedCount;
    }
    
    // Open iteration log if requested
    if (logFile) {
        config.iterLog = fopen(logFile, "w");
        if (!config.iterLog) {
            fprintf(stderr, "Error: Cannot open log file %s\n", logFile);
            free((void*)inputFiles);
            return EXIT_FAILURE;
        }
        fprintf(config.iterLog, "iter,phase,pivot_col,pivot_row,reduced_cost,ratio,obj_rhs\n");
    }
    
    if (config.verbose && config.outputFormat == OUTPUT_TEXT) {
        printf("CUDA Two-Phase Simplex Solver\n");
        printf("=============================\n\n");
    }
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found!\n");
        if (config.iterLog) fclose(config.iterLog);
        free((void*)inputFiles);
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (config.verbose && config.outputFormat == OUTPUT_TEXT) {
        printf("Using CUDA device: %s\n", prop.name);
        printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    }
    
    int exitCode = EXIT_SUCCESS;
    
    // ===== INTERACTIVE MODE =====
    if (interactiveFlag) {
        interactiveMode(&prop, &config, &run);
        if (config.iterLog) fclose(config.iterLog);
        free((void*)inputFiles);
        return EXIT_SUCCESS;
    }
    
    if (batchMode && inputCount > 0) {
        // ===== BATCH MODE =====
        int savedVerbose = config.verbose;
        if (config.verbose < 2) config.verbose = 0;  // Silence per-problem output unless -d
        
        BatchResult* results = (BatchResult*)malloc(inputCount * sizeof(BatchResult));
        int resultCount = 0;
        
        for (int f = 0; f < inputCount; f++) {
            // Extract base filename for display
            const char* base = strrchr(inputFiles[f], '/');
            const char* displayName = base ? base + 1 : inputFiles[f];
            
            LPProblem* lp = parseLP(inputFiles[f], &config);
            if (!lp) {
                snprintf(results[resultCount].filename, sizeof(results[resultCount].filename),
                         "%s", displayName);
                results[resultCount].numVars = 0;
                results[resultCount].numConstraints = 0;
                results[resultCount].statusStr = "PARSE_ERROR";
                results[resultCount].objValue = 0.0;
                results[resultCount].iterations = 0;
                results[resultCount].elapsed = 0.0;
                resultCount++;
                continue;
            }
            
            snprintf(results[resultCount].filename, sizeof(results[resultCount].filename),
                     "%s", displayName);
            results[resultCount].numVars = lp->numVars;
            results[resultCount].numConstraints = lp->numConstraints;
            
            preprocessBounds(lp, &config);
            Tableau* tab = createTableau(lp, &config);
            
            run.totalIterations = 0;
            
            double tstart = hpc_gettime();
            SimplexStatus status = solveSimplex(tab, lp, &config, &run);
            double elapsed = hpc_gettime() - tstart;
            
            results[resultCount].statusStr = statusString(status);
            results[resultCount].iterations = run.totalIterations;
            results[resultCount].elapsed = elapsed;
            
            if (status == OPTIMAL) {
                double* solution;
                results[resultCount].objValue = extractSolutionValues(tab, lp, &solution);
                free(solution);
            } else {
                results[resultCount].objValue = 0.0;
            }
            
            freeTableau(tab);
            freeLPProblem(lp);
            resultCount++;
            
            if (savedVerbose && config.outputFormat == OUTPUT_TEXT) {
                fprintf(stderr, "\rSolved %d/%d problems...", resultCount, inputCount);
                fflush(stderr);
            }
        }
        
        if (savedVerbose && config.outputFormat == OUTPUT_TEXT)
            fprintf(stderr, "\r                              \r");
        
        config.verbose = savedVerbose;
        
        // Print batch summary
        switch (config.outputFormat) {
            case OUTPUT_JSON:
                printBatchSummaryJSON(results, resultCount);
                break;
            case OUTPUT_CSV:
                printBatchSummaryCSV(results, resultCount);
                break;
            case OUTPUT_TEXT:
            default:
                printBatchSummaryText(results, resultCount);
                break;
        }
        
        free(results);
        
    } else {
        // ===== SINGLE FILE MODE =====
        LPProblem* lp = NULL;
        
        if (inputCount > 0) {
            if (config.verbose && config.outputFormat == OUTPUT_TEXT)
                printf("Loading problem from: %s\n\n", inputFiles[0]);
            lp = parseLP(inputFiles[0], &config);
            if (!lp) {
                if (config.iterLog) fclose(config.iterLog);
                free((void*)inputFiles);
                return EXIT_FAILURE;
            }
        } else {
            if (config.outputFormat == OUTPUT_TEXT && config.verbose) {
                printf("No input file provided. Using test problem.\n\n");
                printf("Usage: %s [options] <problem.{mps,dat}>\n\n", argv[0]);
            }
            lp = createTestProblem();
            
            if (config.verbose && config.outputFormat == OUTPUT_TEXT) {
                printf("Test Problem:\n");
                printf("  Maximize: 3*x1 + 2*x2\n");
                printf("  Subject to:\n");
                printf("    x1 + x2 <= 4\n");
                printf("    2*x1 + x2 <= 6\n");
                printf("  Expected: x1=2, x2=2, z=10\n");
            }
        }
        
        // Preprocess variable bounds and range constraints
        preprocessBounds(lp, &config);
        
        // Create tableau
        Tableau* tab = createTableau(lp, &config);
        
        run.totalIterations = 0;
        
        // Time only the computation (solving), not I/O
        double tstart = hpc_gettime();
        
        // Solve
        SimplexStatus status = solveSimplex(tab, lp, &config, &run);
        
        double tfinish = hpc_gettime();
        double elapsed = tfinish - tstart;
        
        // Output solution in the requested format
        outputSolution(tab, lp, status, elapsed, &config, &run);
        
        // Cleanup
        freeTableau(tab);
        freeLPProblem(lp);
        
        exitCode = (status == OPTIMAL) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    // Final cleanup
    if (config.iterLog) fclose(config.iterLog);
    free((void*)inputFiles);
    
    return exitCode;
}
