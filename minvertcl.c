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
#define KERNEL_NORMALISE_PIVOT 4
#define KERNEL_REDUCE          5
#define KERNEL_REARRANGE       6

#define KERNEL_COUNT 7

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
	kernel_names[KERNEL_NORMALISE_PIVOT] = "normalise_pivot";
	kernel_names[KERNEL_REDUCE         ] = "reduce";
	kernel_names[KERNEL_REARRANGE      ] = "rearrange";

	struct cooker_dish prg;	
	if (cooker_dish_init(
			&prg, &plate, "minvert.cl", 1, kernel_names, VERBOSITY
		) != CS_OK
	)
		exit(EXIT_FAILURE);

	// Set kernel constants for enhanced code readability
	cl_kernel const k_build_identity = prg.kernels[KERNEL_BUILD_IDENTITY];
	cl_kernel const k_max_reduc_begin = prg.kernels[KERNEL_MAX_REDUC_BEGIN];
	cl_kernel const k_max_reduc      = prg.kernels[KERNEL_MAX_REDUC];
	cl_kernel const k_store_column   = prg.kernels[KERNEL_STORE_COLUMN];
	cl_kernel const k_normalise_pivot = prg.kernels[KERNEL_NORMALISE_PIVOT];
	cl_kernel const k_reduce         = prg.kernels[KERNEL_REDUCE];
	cl_kernel const k_rearrange      = prg.kernels[KERNEL_REARRANGE];
	
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
	cl_mem indices; // Used for maximum reduction, reduction and rearrangement
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
		plate.context, mem_flags, 2 * n * sizeof(cl_ulong), NULL, &status
	);
	if (!indices || status != CL_SUCCESS) {
		puts("Could not create buffer.");
		exit(EXIT_FAILURE);
	}

	// Define procedures to run on the GPU
	size_t const linear_single_size = (n + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
	size_t const linear_double_size = 2 * n;
	size_t global_2d_sizes[2];
	global_2d_sizes[0] = n;
	global_2d_sizes[1] = linear_double_size;

	// Need to turn size_t to unsigned long for ensured kernel compatibility.
	cl_ulong const kp_n = n;
	cl_ulong const kp_width = REDUCTION_WIDTH;

	// Initialise a bunch of parameters for all kernels, those that will not
	// change accross executions.
	status = (
		  clSetKernelArg(k_build_identity, 0, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_build_identity, 1, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_build_identity, 2, sizeof(cl_ulong), &kp_width)

		| clSetKernelArg(k_max_reduc_begin, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_max_reduc_begin, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_max_reduc_begin, 3, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_max_reduc_begin, 4, sizeof(cl_ulong), &kp_width)

		| clSetKernelArg(k_max_reduc, 2, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_max_reduc, 3, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_max_reduc, 4, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_max_reduc, 5, sizeof(cl_ulong), &kp_width)

		| clSetKernelArg(k_store_column, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_store_column, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_store_column, 3, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_store_column, 4, sizeof(cl_ulong), &kp_width)

		| clSetKernelArg(k_normalise_pivot, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_normalise_pivot, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_normalise_pivot, 3, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_normalise_pivot, 4, sizeof(cl_ulong), &kp_width)

		| clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_reduce, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_reduce, 3, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_reduce, 4, sizeof(cl_ulong), &kp_width)

		| clSetKernelArg(k_rearrange, 0, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_rearrange, 1, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_rearrange, 2, sizeof(cl_ulong), &kp_n)
		| clSetKernelArg(k_rearrange, 3, sizeof(cl_ulong), &kp_width)
	);
	if (status != CL_SUCCESS) {
		puts("Could not define kernel arguments.");
		exit(EXIT_FAILURE);
	}

	double const zero = 0.;
	status = clEnqueueFillBuffer(
		plate.queue, matrix, &zero, sizeof(zero), size, size, 0, NULL, NULL
	);
	if (status != CL_SUCCESS) {
		puts("Could not enqueue buffer initialisation.");
		exit(EXIT_FAILURE);
	}
	status = clEnqueueNDRangeKernel(
		plate.queue, k_build_identity,
		1, NULL, &linear_single_size, NULL,
		0, NULL, NULL
	);
	if (status != CL_SUCCESS) {
		puts("Could not enqueue <build_identity> kernel execution.");
		exit(EXIT_FAILURE);
	}

	// Iterate through all lines for our custom Gauss-Jordan reduction
	for (cl_ulong k = 0; k < kp_n; ++k) {
		// Set k for all kernels
		status = (
			  clSetKernelArg(k_max_reduc_begin, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_max_reduc, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_store_column, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_normalise_pivot, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_reduce, 0, sizeof(cl_ulong), &k)
		);
		if (status != CL_SUCCESS) {
			puts("Could not define kernel arguments.");
			exit(EXIT_FAILURE);
		}

		// Find maximum of line 
		status = clEnqueueNDRangeKernel(
			plate.queue, k_max_reduc_begin,
			1, NULL, &linear_single_size, NULL,
			0, NULL, NULL
		);
		if (status != CL_SUCCESS) {
			puts("Could not enqueue <max_reduc_begin> kernel execution.");
			exit(EXIT_FAILURE);
		}

		cl_ulong depth = 0;
		size_t count = n;
		do {
			count = (count + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
			status = clSetKernelArg(k_max_reduc, 1, sizeof(cl_ulong), &depth);
			if (status != CL_SUCCESS) {
				puts("Could not define kernel argument.");
				exit(EXIT_FAILURE);
			}
			++depth;
			status = clEnqueueNDRangeKernel(
				plate.queue, k_max_reduc, 1, NULL, &count, NULL, 0, NULL, NULL
			);
			if (status != CL_SUCCESS) {
				puts("Could not enqueue <max_reduc> kernel execution.");
				exit(EXIT_FAILURE);
			}
			
			++depth;
		} while (count > 1);

		status = clEnqueueNDRangeKernel(
			plate.queue, k_store_column,
			1, NULL, &linear_single_size, NULL,
			0, NULL, NULL
		);
		if (status != CL_SUCCESS) {
			puts("Could not enqueue <store_column> kernel execution.");
			exit(EXIT_FAILURE);
		}

		status = clEnqueueNDRangeKernel(
			plate.queue, k_normalise_pivot,
			1, NULL, &linear_double_size, NULL,
			0, NULL, NULL
		);
		if (status != CL_SUCCESS) {
			puts("Could not enqueue <normalise_pivot> kernel execution.");
			exit(EXIT_FAILURE);
		}

		status = clEnqueueNDRangeKernel(
			plate.queue, k_reduce,
			2, NULL, global_2d_sizes, NULL,
			0, NULL, NULL
		);
		if (status != CL_SUCCESS) {
			puts("Could not enqueue <reduce> kernel execution.");
			exit(EXIT_FAILURE);
		}
	}

	global_2d_sizes[1] = linear_single_size;
	status = clEnqueueNDRangeKernel(
		plate.queue, k_rearrange, 2, NULL, global_2d_sizes, NULL, 0, NULL, NULL
	);
	if (status != CL_SUCCESS) {
		puts("Could not enqueue kernel execution.");
		exit(EXIT_FAILURE);
	}

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

