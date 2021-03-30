// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For time measurement
#include <math.h> // For error measurement
                  // Ah, also useful for iteration counting

// Project includes
#include "clcooker.h"
#include "error_code.h"
#include "reduction_width.h"

#define SECOND_DURATION_NS 1000000000ll

#define VERBOSITY CV_ALL
#define PRINT_PROCESS_INFO 1

#define KERNEL_BUILD_IDENTITY  0
#define KERNEL_MAX_REDUC_BEGIN 1
#define KERNEL_MAX_REDUC       2
#define KERNEL_STORE_COLUMN    3
#define KERNEL_NORMALISE_PIVOT 4
#define KERNEL_REDUCE          5
#define KERNEL_REARRANGE       6
#define KERNEL_COUNT 7

#if PRINT_PROCESS_INFO
static void print_indices(cl_ulong const *const indices, size_t const count) {
	for (size_t j = 0; j < count; ++j)
		printf("%lu ", indices[j]);
	puts("");
}
static void print_line(double const *const l, size_t const width) {
	for (size_t j = 0; j < width; ++j)
		printf("%f ", l[j]);
	puts("");
}
static void print_matrix(double const *const buf, size_t const height, size_t const width) {
	for (size_t i = 0; i < height; ++i)
		print_line(buf + i * width, width);
	puts("");
}
#define PRINT_STEP(step, msg) {\
	printf("%lu - %s:\n", step, msg); \
	puts("\nIndices:"); \
	status = clEnqueueReadBuffer( \
		plate.queue, indices, CL_TRUE, 0, 2*n*sizeof(cl_ulong), disp_indices, 0, NULL, NULL \
	); \
	if (status != CL_SUCCESS) { \
		printf("Could not fetch matrix back. (%d)", status); \
		exit(EXIT_FAILURE); \
	} \
	print_indices(disp_indices, 2*n); \
	print_line((double const*)disp_indices, 2*n); \
	puts("\nMatrix:"); \
	status = clEnqueueReadBuffer( \
		plate.queue, matrix, CL_TRUE, 0, 2*n*n*sizeof(double), disp_matrix, 0, NULL, NULL \
	); \
	if (status != CL_SUCCESS) { \
		printf("Could not fetch matrix back. (%d)", status); \
		exit(EXIT_FAILURE); \
	} \
	print_matrix(disp_matrix, n, 2*n); \
	fflush(stdout); \
}
#else
#define PRINT_STEP(step, msg)
#endif

static void check_status(cl_int const status, char const *const msg) {
	if (status != CL_SUCCESS) {
		puts(msg);
		printf("Status: %s\n", cl_status_string(status));
		exit(EXIT_FAILURE);
	}
}

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
			&prg, &plate, "minvert.cl", KERNEL_COUNT, kernel_names, VERBOSITY
		) != CS_OK
	)
		exit(EXIT_FAILURE);

#if PRINT_PROCESS_INFO
	cl_ulong *const disp_indices = malloc(2*n * sizeof(cl_ulong));
	double *const disp_matrix = malloc(2*n*n * sizeof(double));
	if (!disp_indices || !disp_matrix) {
		puts("Could not allocate output buffers.");
		exit(EXIT_FAILURE);
	}
#endif

	// Set kernel constants for enhanced code readability
	cl_kernel const k_build_identity  = prg.kernels[KERNEL_BUILD_IDENTITY];
	cl_kernel const k_max_reduc_begin = prg.kernels[KERNEL_MAX_REDUC_BEGIN];
	cl_kernel const k_max_reduc       = prg.kernels[KERNEL_MAX_REDUC];
	cl_kernel const k_store_column    = prg.kernels[KERNEL_STORE_COLUMN];
	cl_kernel const k_normalise_pivot = prg.kernels[KERNEL_NORMALISE_PIVOT];
	cl_kernel const k_reduce          = prg.kernels[KERNEL_REDUCE];
	cl_kernel const k_rearrange       = prg.kernels[KERNEL_REARRANGE];
	
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
	size_t const extended_line_length = 2 * n;
	size_t const extended_line_size = extended_line_length * sizeof(double);
	size_t const line_size = n * sizeof(double);
	size_t const size = n * line_size;

	// Create buffer objects for input & output matrix and reordering data.
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE /*| CL_MEM_HOST_NO_ACCESS*/;

	matrix = clCreateBuffer(plate.context, mem_flags, 2 * size, NULL, &status);
	check_status(status, "Could not create OpenCL buffer.");

	indices = clCreateBuffer(
		plate.context, mem_flags, 2 * n * sizeof(cl_ulong), NULL, &status
	);
	check_status(status, "Could not create OpenCL buffer.");

	// Define procedures to run on the GPU
	size_t const linear_single_size = (n + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
	size_t const linear_double_size = 
		(extended_line_length + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
	size_t global_2d_sizes[2];
	global_2d_sizes[0] = n;
	global_2d_sizes[1] = linear_double_size;

	// Need to turn size_t to unsigned long for ensured kernel compatibility.
	cl_ulong const kp_n = n;

	// Initialise a bunch of parameters for all kernels, those that will not
	// change accross executions.
	status = (
		  clSetKernelArg(k_build_identity, 0, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_build_identity, 1, sizeof(cl_ulong), &kp_n)

		| clSetKernelArg(k_max_reduc_begin, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_max_reduc_begin, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_max_reduc_begin, 3, sizeof(cl_ulong), &kp_n)

		| clSetKernelArg(k_max_reduc, 2, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_max_reduc, 3, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_max_reduc, 4, sizeof(cl_ulong), &kp_n)

		| clSetKernelArg(k_store_column, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_store_column, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_store_column, 3, sizeof(cl_ulong), &kp_n)

		| clSetKernelArg(k_normalise_pivot, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_normalise_pivot, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_normalise_pivot, 3, sizeof(cl_ulong), &kp_n)

		| clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_reduce, 2, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_reduce, 3, sizeof(cl_ulong), &kp_n)

		| clSetKernelArg(k_rearrange, 0, sizeof(cl_mem), &matrix)
		| clSetKernelArg(k_rearrange, 1, sizeof(cl_mem), &indices)
		| clSetKernelArg(k_rearrange, 2, sizeof(cl_ulong), &kp_n)
	);
	check_status(status, "Could not define kernel arguments.");

	double const zero = 0.;
	for (size_t i = 0; i < n; ++i) {
		status = clEnqueueWriteBuffer(
			plate.queue, matrix, CL_FALSE,
			i * extended_line_size, line_size, &in[i*n], 0, NULL, NULL
		);
		check_status(status, "Could not enqueue initial buffer copy.");
		status = clEnqueueFillBuffer(
			plate.queue, matrix, &zero, sizeof(zero),
			i * extended_line_size + line_size, line_size, 0, NULL, NULL
		);
		check_status(status, "Could not enqueue identity buffer initialisation.");
	}

	status = clEnqueueNDRangeKernel(
		plate.queue, k_build_identity,
		1, NULL, &linear_single_size, NULL,
		0, NULL, NULL
	);
	check_status(status, "Could not enqueue <build_identity> kernel execution.");

	// Iterate through all lines for our custom Gauss-Jordan reduction
	for (cl_ulong k = 0; k < kp_n; ++k) {
		PRINT_STEP(k, "Etat initial")
		// Set k for all kernels
		status = (
			  clSetKernelArg(k_max_reduc_begin, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_max_reduc, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_store_column, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_normalise_pivot, 0, sizeof(cl_ulong), &k)
			| clSetKernelArg(k_reduce, 0, sizeof(cl_ulong), &k)
		);
		check_status(status, "Could not define kernel arguments.");

		// Find maximum of line 
		status = clEnqueueNDRangeKernel(
			plate.queue, k_max_reduc_begin,
			1, NULL, &linear_single_size, NULL,
			0, NULL, NULL
		);
		check_status(status, "Could not enqueue <max_reduc_begin> kernel execution.");

		cl_ulong depth = 0;
		size_t count = n;
		do {
			count = (count + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
			status = clSetKernelArg(k_max_reduc, 1, sizeof(cl_ulong), &depth);
			check_status(status, "Could not define kernel argument.");

			++depth;
			status = clEnqueueNDRangeKernel(
				plate.queue, k_max_reduc, 1, NULL, &count, NULL, 0, NULL, NULL
			);
			check_status(status, "Could not enqueue <max_reduc> kernel execution.");
			
			PRINT_STEP(k, "Reduction")

			++depth;
		} while (count > 1);

		status = clEnqueueNDRangeKernel(
			plate.queue, k_store_column,
			1, NULL, &linear_single_size, NULL,
			0, NULL, NULL
		);
		check_status(status, "Could not enqueue <store_column> kernel execution.");

		PRINT_STEP(k, "Store column")

		status = clEnqueueNDRangeKernel(
			plate.queue, k_normalise_pivot,
			1, NULL, &linear_double_size, NULL,
			0, NULL, NULL
		);
		check_status(status, "Could not enqueue <normalise_pivot> kernel execution.");

		PRINT_STEP(k, "Normalise")

		status = clEnqueueNDRangeKernel(
			plate.queue, k_reduce,
			2, NULL, global_2d_sizes, NULL,
			0, NULL, NULL
		);
		check_status(status, "Could not enqueue <reduce> kernel execution.");
	}

	PRINT_STEP(n, "Final");

	global_2d_sizes[0] = n;
	global_2d_sizes[1] = linear_single_size;
	status = clEnqueueNDRangeKernel(
		plate.queue, k_rearrange, 2, NULL, global_2d_sizes, NULL, 0, NULL, NULL
	);
	check_status(status, "Could not enqueue kernel execution.");

	PRINT_STEP(n, "Rearrange");

	// Fetch the resulting matrix back to host memory (in output buffer).
	for (size_t i = 0; i < n; ++i) {
		status = clEnqueueReadBuffer(
			plate.queue, matrix, CL_TRUE,
			i * extended_line_size, line_size, &out[i*n], 0, NULL, NULL
		);
		check_status(status, "Could not fetch matrix back.");
	}

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
#if PRINT_PROCESS_INFO
	puts("Input:");
	print_matrix(m, n, n);
#endif
	
	// Launch invertion process
	double duration, error = n;
	invert_matrix(m, r, n, &duration);

#if PRINT_PROCESS_INFO
	puts("Result:");
	print_matrix(r, n, n);
#endif

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

