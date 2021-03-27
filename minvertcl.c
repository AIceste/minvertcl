// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For time measurement
#include <math.h> // For error measurement
                  // Ah, also useful for iteration counting

// Project includes
#include "clcooker.h"

#define SECOND_DURATION_NS 1000000000ll

#define VERBOSITY CV_ALL
#define REDUCTION_WIDTH 4

#define KERNEL_BUILD_IDENTITY  0
#define KERNEL_MAX_REDUC_BEGIN 1
#define KERNEL_MAX_REDUC       2
#define KERNEL_STORE_COLUMN    3
#define KERNEL_NORMALISE_LINE  4
#define KERNEL_REDUCE_MATRIX   5
#define KERNEL_REARRANGE       6

#define KERNEL_COUNT 6

static int64_t time_point(struct timespec *ts) {
	return ts->tv_sec * SECOND_DURATION_NS + ts->tv_nsec;
}

static void invert_matrix(
	double *const in, double *const out, size_t const n, double *const out_duration
) {
	// Apply our classic twist of Gauss-Jordan-reduction-based 
	// matrix inversion to matrix m through usage of OpenCL GPGPU.

	// Initialise OpenCL usage context
	struct cooker_plate plate;
	if (cooker_plate_init(&plate, VERBOSITY) != CS_OK)
		exit(EXIT_FAILURE);

	// Initialise program from file
	char const *kernel_names[KERNEL_COUNT];
	kernel_names[KERNEL_BUILD_IDENTITY ] = "build_identity";
	kernel_names[KERNEL_MAX_REDUC_BEGIN] = "max_reduc_begin";
	kernel_names[KERNEL_MAX_REDUC      ] = "max_reduc";
	kernel_names[KERNEL_STORE_COLUMN   ] = "store_column";
	kernel_names[KERNEL_NORMALISE_LINE ] = "normalise_line";
	kernel_names[KERNEL_REDUCE_MATRIX  ] = "reduce";
	kernel_names[KERNEL_REARRANGE      ] = "rearrange";

	struct cooker_dish prg;	
	if (cooker_dish_init(
			&prg, &plate, "minvert.cl", 1, kernel_names, VERBOSITY
		) != CS_OK
	)
		exit(EXIT_FAILURE);
	
	// Begin time counting if needed
	int64_t tp_start;
	if (out_duration) {
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		int64_t const t = time_point(&ts);
		clock_gettime(CLOCK_REALTIME, &ts);
		tp_start = time_point(&ts);
		tp_start += tp_start - t;
	}

	cl_int status; // For error management
	cl_mem matrix; // Initially m on the left and initialised I n x n on the right
	cl_mem indices; // Used for maximum reduction and for rearrangement
	size_t const size = n * n * sizeof(double);

	// Create buffer objects for input & output matrix and reordering data.
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

	matrix = clCreateBuffer(
		plate.context, mem_flags | CL_MEM_COPY_HOST_PTR, 2 * size, in, &status
	);
	if (!matrix || status != CL_SUCCESS) {
		puts("Could not create buffer.");
		exit(EXIT_FAILURE);
	}

	indices = clCreateBuffer(
		plate.context, mem_flags, 2 * n * sizeof(size_t), NULL, &status
	);
	if (!indices || status != CL_SUCCESS) {
		puts("Could not create buffer.");
		exit(EXIT_FAILURE);
	}

	// Define procedures to run on the GPU
	size_t const linear_single_size = (n + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
	size_t const linean_double_size = 2 * n

	double const zero = 0.;
	clEnqueueFillBuffer(
		plate.queue, matrix, &zero, sizeof(zero), size, size, 0, NULL, NULL
	);
	
	
/*	
	// Associate the input and output buffers with the kernel 
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg failed\n");
		exit(EXIT_FAILURE);
	}
	
	// Define an index space (global work size) of threads for execution.  
	// A workgroup size (local work size) is not required, but can be used.
	size_t work_sizes[1];  // There are ELEMENTS threads
	work_sizes[0] = ELEMENTS;
	
	// Execute the kernel.
	if (clEnqueueNDRangeKernel(
			cmdQueue, kernel, 1, NULL, work_sizes, NULL, 0, NULL, NULL
		) != CL_SUCCESS
	) {
		printf("clEnqueueNDRangeKernel failed\n");
		exit(EXIT_FAILURE);
	}
*/	
	// Fetch the resulting matrix back to host memory (in output buffer).
	clEnqueueReadBuffer(plate.queue, matrix, CL_TRUE, 0, size, out, 0, NULL, NULL);
	
	clReleaseMemObject(matrix);
	clReleaseMemObject(indices);

	// Finalise time counting if needed
	if (out_duration) {
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);	
		*out_duration = (time_point(&ts) - tp_start) / (double)SECOND_DURATION_NS;
	}

	cooker_plate_destroy(&plate);
	cooker_dish_destroy(&prg);
}

int main(int argc, char const *const *const argv) {
	srand(argc > 2 ? atol(argv[2]) : time(NULL));

	// Build random matrix to invert
	size_t n = argc > 1 ? atol(argv[1]) : 5;
	size_t const size = n*n;

	// Store input matrix and output matrix in that buffer.
	double *const matrices = malloc(2*size*sizeof(double));
	if (!matrices) {
		perror("malloc");
		exit(EXIT_FAILURE);
	}
	double *const m = matrices;
	double *const r = matrices + size;
	for (size_t i = 0; i < size; ++i) {
		m[i] = rand() / (double)RAND_MAX;
	}
	
	// Launch invertion process
	double duration, error = n;
	invert_matrix(m, r, n, &duration);

	// Calculate error the same (strange) way it was done in TP3 
	// - multiply matrix with inverted matrix
	// - sum all cells of the resulting matrix
	// - error is the absolute difference between said sum and n
	// ...in theory, that is. Practically speaking, we can't afford to
	// waste as much time for this process, so we do not differentiate
	// between individual steps.
	for (size_t i = 0; i < n; ++i) {
		for (size_t c = 0; c < n; ++c) {
			for (size_t j = 0; j < n; ++j) {
				error -= m[i * n + j] * r[j * n + c];
			}
		}
	}	
	free(matrices);
	error = fabs(error);

	printf(
		"Matrix size: %lu\nInversion duration: %f\nError: %f\n",
		n, duration, error
	);

	exit(EXIT_SUCCESS);
}

