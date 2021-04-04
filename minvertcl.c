// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For time measurement
#include <math.h> // For error measurement
                  // And event counting

// Project includes
#include "clcooker.h"
#include "error_code.h"
#include "reduction_width.h"

#define SECOND_DURATION_NS 1000000000ll

#define VERBOSITY CV_ALL
#define PRINT_PROCESS_INFO 0
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
	check_status(status, "Could not fetch indices back."); \
	print_indices(disp_indices, step + n / REDUCTION_WIDTH + 1); \
	print_line((double const*)disp_indices + n, n); \
	status = clEnqueueReadBuffer( \
		plate.queue, matrix, CL_TRUE, 0, 2*n*n*sizeof(double), disp_matrix, 0, NULL, NULL \
	); \
	check_status(status, "Could not fetch matrix back."); \
	puts("\nMatrix (left):"); \
	print_matrix(disp_matrix, n, n); \
	puts("Matrix (extension):"); \
	print_matrix(disp_matrix + n * n, n, n); \
	fflush(stdout); \
}
#else
#define PRINT_STEP(step, msg)
#endif
#define HARD_SYNCH clEnqueueReadBuffer( \
	plate.queue, matrix, CL_TRUE, 0, sizeof(cl_double), out, 0, NULL, NULL);

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
	kernel_names[0] = "build_identity";
	kernel_names[1] = "max_reduc_begin";
	kernel_names[2] = "max_reduc";
	kernel_names[3] = "store_column";
	kernel_names[4] = "normalise_pivot";
	kernel_names[5] = "reduce";
	kernel_names[6] = "rearrange";

	struct cooker_dish prg;	
	if (cooker_dish_init(
			&prg, &plate, "minvert.cl", KERNEL_COUNT, kernel_names, VERBOSITY
		) != CS_OK
	)
		exit(EXIT_FAILURE);

	// Set kernel constants for enhanced code readability
	cl_kernel const k_build_identity  = prg.kernels[0];
	cl_kernel const k_max_reduc_begin = prg.kernels[1];
	cl_kernel const k_max_reduc       = prg.kernels[2];
	cl_kernel const k_store_column    = prg.kernels[3];
	cl_kernel const k_normalise_pivot = prg.kernels[4];
	cl_kernel const k_reduce          = prg.kernels[5];
	cl_kernel const k_rearrange       = prg.kernels[6];
	
#if PRINT_PROCESS_INFO
	cl_ulong *const disp_indices = malloc(2*n * sizeof(cl_ulong));
	double *const disp_matrix = malloc(2*n*n * sizeof(double));
	if (!disp_indices || !disp_matrix) {
		puts("Could not allocate output buffers.");
		exit(EXIT_FAILURE);
	}
#endif

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
	size_t const line_size = n * sizeof(double);
	size_t const size = n * line_size;

	// Create buffer objects for input & output matrix and reordering data.
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE /*| CL_MEM_HOST_NO_ACCESS*/;

	matrix = clCreateBuffer(plate.context, mem_flags, 2 * size, NULL, &status);
	check_status(status, "Could not create OpenCL buffer.");

	// Multipurpose buffer
	indices = clCreateBuffer(
		plate.context, mem_flags, 2 * line_size, NULL, &status
	);
	check_status(status, "Could not create OpenCL buffer.");

	// Define procedures to run on the GPU
	size_t const linear_size = (n + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
	size_t const normalise_dim_2d[2] = {2, linear_size};
	size_t const reduce_dim_3d[3] = {2, n - 1, linear_size};
	size_t const rearrange_dim_2d[2] = {n, linear_size};

	// Count and allocate events 
	// Could do with 3 events and cycles, but go the hard way.
	size_t const event_count = 
		5 + n * (5 + (size_t)(log2(n) / log2(REDUCTION_WIDTH)));
	cl_event *const events = malloc(event_count * sizeof(cl_event));
	if (!events) {
		puts("Could not mallocate event buffer.");
		exit(EXIT_FAILURE);
	}

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
	status = clEnqueueWriteBuffer(
		plate.queue, matrix, CL_FALSE, 0, size, in, 0, NULL, &events[1]
	);
	check_status(status, "Could not enqueue initial buffer copy.");
	status = clEnqueueFillBuffer(
		plate.queue, matrix, &zero, sizeof(zero),
		size, size, 0, NULL, &events[0]
	);
	check_status(status, "Could not enqueue identity buffer initialisation.");

	status = clEnqueueNDRangeKernel(
		plate.queue, k_build_identity,
		1, NULL, &linear_size, NULL,
		1, &events[0], &events[2]
	);
	check_status(status, "Could not enqueue <build_identity> kernel execution.");

	size_t event = 3;
	check_status(
		clEnqueueBarrierWithWaitList(plate.queue, 2, &events[1], &events[event]),
		"Could not wait for buffer initialisation."
	);

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
			1, NULL, &linear_size, NULL,
			1, &events[event], &events[event + 1]
		);
		check_status(status, "Could not enqueue <max_reduc_begin> kernel execution.");
		++event;

		PRINT_STEP(k, "Base Reduction")
		cl_ulong depth = 0;
		size_t count = n;
		do {
			count = (count + REDUCTION_WIDTH - 1) / REDUCTION_WIDTH;
			status = clSetKernelArg(k_max_reduc, 1, sizeof(cl_ulong), &depth);
			check_status(status, "Could not define kernel argument.");

			status = clEnqueueNDRangeKernel(
				plate.queue, k_max_reduc,
				1, NULL, &count, NULL,
				1, &events[event], &events[event + 1]
			);
			check_status(status, "Could not enqueue <max_reduc> kernel execution.");
			++event;
			
			PRINT_STEP(k, "Reduction")
			// This also does not work, obviously, since it's an issue with events.
			// check_status(
			// 	clEnqueueMarkerWithWaitList(plate.queue, 0, NULL, NULL),
			// 	"Could not synchronise during reduction."
			// );
			HARD_SYNCH // Slow things down but the event does not seem to work...
			// Removing this double the speed but breaks the result

			++depth;
		} while (count > 1);

		status = clEnqueueNDRangeKernel(
			plate.queue, k_store_column,
			1, NULL, &linear_size, NULL,
			1, &events[event], &events[event + 1]
		);
		check_status(status, "Could not enqueue <store_column> kernel execution.");
		++event;

		PRINT_STEP(k, "Store column")

		status = clEnqueueNDRangeKernel(
			plate.queue, k_normalise_pivot,
			2, NULL, normalise_dim_2d, NULL,
			1, &events[event], &events[event + 1]
		);
		check_status(status, "Could not enqueue <normalise_pivot> kernel execution.");
		++event;

		PRINT_STEP(k, "Normalise")

		status = clEnqueueNDRangeKernel(
			plate.queue, k_reduce,
			3, NULL, reduce_dim_3d, NULL,
			1, &events[event], &events[event + 1]
		);
		check_status(status, "Could not enqueue <reduce> kernel execution.");
		++event;
	}

	PRINT_STEP(n, "Final");

	status = clEnqueueNDRangeKernel(
		plate.queue, k_rearrange,
		2, NULL, rearrange_dim_2d, NULL,
		1, &events[event], &events[event + 1]
	);
	check_status(status, "Could not enqueue kernel execution.");
	++event;

	PRINT_STEP(n, "Rearrange");

	// Fetch the resulting matrix back to host memory (in output buffer).
	status = clEnqueueReadBuffer(
		plate.queue, matrix, CL_TRUE, 0, size, out, 1, &events[event], NULL
	);
	check_status(status, "Could not fetch matrix back.");

	free(events);
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
		"Matrix size: %lu\nInversion duration: %f\nError: %.9f\n",
		n, duration, error
	);

	exit(EXIT_SUCCESS);
}

