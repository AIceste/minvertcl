#ifndef CLCOOKER_H
#define CLCOOKER_H
/* A simple interface to work with OpenCL on a standard dev desktop.
 * Abstracts lot of things and allows easy prototyping whilst being
 * of low flexibility. 
 */

#include <CL/cl.h>

struct cooker_plate {
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
};

struct cooker_dish {
	cl_program program;
	char *source;
	cl_kernel *kernels;
	size_t kernel_count;
};

enum cooker_state {
	CS_OK = 0,
	CS_MEMORY,
	CS_IO,
	CS_HARDWARE,
	CS_CL,
};
enum cooker_verbosity {
	CV_NOTHING = 0,
	CV_ERROR = 1,
	CV_INFO = 2,
	CV_ALL = 3,
};

// Initialise a plate that allows using OpenCL with one device of one platform, 
// if any is found. 
enum cooker_state cooker_plate_init(struct cooker_plate *plate, enum cooker_verbosity v);
static inline void cooker_plate_destroy(struct cooker_plate *plate) {
	clReleaseCommandQueue(plate->queue);
	clReleaseContext(plate->context);
}

// Initialise a dish from a .cl file name and a null-terminated list of kernel
// names, which will be allocated and initialised with regard to order.
enum cooker_state cooker_dish_init(
	struct cooker_dish *dish, struct cooker_plate const *plate,
	char const *program_file, size_t name_count, char const *const *kernel_names,
	enum cooker_verbosity v
);
static inline void cooker_dish_destroy(struct cooker_dish *const dish) {
	for (size_t k = 0; k < dish->kernel_count; ++k)
		clReleaseKernel(dish->kernels[k]);
	free(dish->kernels);
	clReleaseProgram(dish->program);
	free(dish->source);
}

#endif
