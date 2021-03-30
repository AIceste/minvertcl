#include "clcooker.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

static cl_uint device_print(cl_device_id const d) {
	char buf[420] = "Device: ";
	size_t pos = sizeof("Device: ") - 1u;
	size_t rem = sizeof(buf) - pos;
	size_t off = 0;
	char *const beg = buf + pos;
	char const *const sep = " - ";
	size_t const ssz = sizeof(" - ") - 1;

	cl_int r = clGetDeviceInfo(d, CL_DEVICE_VENDOR, rem, beg, &off);
	off -= 1;
	r |= clGetDeviceInfo(d, CL_DEVICE_NAME, rem - off, beg + off + ssz, NULL);
	memcpy(beg + off, sep, ssz); 
	puts(buf);
	return r;
}

static int read_source(char const *const program_file, char **const out, int err) {
	char *source;
	struct stat st;
	int const fd = open(program_file, O_RDONLY);
	if (fd == -1) {
		if (err) printf("Could not open kernel file: %s\n", program_file);
		return 1;
	}
	
	if (fstat(fd, &st)) {
		if (err) printf("Error getting file size\n");
		return 1;
	}
	
	source = malloc(st.st_size + 1);
	if (!source) {
		if (err) printf("Error allocating %ld bytes for the program source\n", st.st_size + 1);
		return 2;
	}
	
	if (read(fd, source, st.st_size) != st.st_size) {
		if (err) printf("Did not read enough bytes.");
		return 1;
	}
	
	source[st.st_size] = '\0';
	*out = source;
	return 0;
}

static void build_print(cl_program const program, cl_device_id const device) {
	printf("Program failed to build.\n");
	cl_build_status buildStatus;
	clGetProgramBuildInfo(
		program, device, CL_PROGRAM_BUILD_STATUS,
		sizeof(cl_build_status), &buildStatus, NULL
	);
	
	char *buildLog;
	size_t buildLogSize;
	clGetProgramBuildInfo(
		program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize
	);
	buildLog = malloc(buildLogSize);
	if (buildLog == NULL) {
		perror("malloc");
		return;
	}
	clGetProgramBuildInfo(
		program, device, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL
	);
	buildLog[buildLogSize-1] = '\0';
	printf("Build Log:\n%s\n", buildLog);   
	free(buildLog);
}

enum cooker_state cooker_plate_init(
	struct cooker_plate *plate, enum cooker_verbosity const v
) {
	enum cooker_state ret = CS_OK;
	char const *message = NULL;
	cl_int status; // For error handling

	cl_uint p_count = 1;
	cl_platform_id platform;
	
	// Try to find one platform and use it	
	if (clGetPlatformIDs(p_count, &platform, &p_count) != CL_SUCCESS)
		goto failure_getplatformid;
	if (!p_count)
		goto failure_no_platform;
	
	cl_uint d_count = 1;
	cl_device_id device;
	
	// Retrive the number of devices present
	if (clGetDeviceIDs(
			platform, CL_DEVICE_TYPE_GPU, d_count, &device, &d_count
		) != CL_SUCCESS
	)
		goto failure_getdeviceid;
	if (!d_count)
		goto failure_no_device;
	
	// Print out device
	if (v & CV_INFO && device_print(device) != CL_SUCCESS)
		goto failure_info;

	// Create a context for using the device
	cl_context const context = clCreateContext(
		NULL, 1, &device, NULL, NULL, &status
	);
	if (status != CL_SUCCESS || !context)
		goto failure_createcontext;

	// Create a command queue for using the device
	cl_command_queue const queue = clCreateCommandQueueWithProperties(
		context, device, NULL, &status
	);
	if (status != CL_SUCCESS || !queue)
		goto failure_createcommandqueue;

	// Assign output fields
	plate->context = context;
	plate->device = device;
	plate->queue = queue;

	return ret;

failure_getplatformid:
	message = "Could not get OpenCL platform.";
	ret = CS_CL;
	goto cleanup_exit;
failure_no_platform:
	message = "No platforms detected.";
	ret = CS_HARDWARE;
	goto cleanup_exit;
failure_getdeviceid:
	message = "Could not get OpenCL device.";
	ret = CS_CL;
	goto cleanup_exit;
failure_no_device:
	message = "No device detected.";
	ret = CS_HARDWARE;
	goto cleanup_exit;
failure_info:
	message = "Could not get device information.";
	ret = CS_CL;
	goto cleanup_exit;
failure_createcontext:
	message = "Could not create OpenCL context.";
	ret = CS_CL;
	goto cleanup_exit;
failure_createcommandqueue:
	message = "Could not create OpenCL command queue.";
	ret = CS_CL;
	goto cleanup_context;

cleanup_context:
	clReleaseContext(context);

cleanup_exit:
	if (v & CV_ERROR)
		puts(message); 

	return ret;
}

enum cooker_state cooker_dish_init(
	struct cooker_dish *dish, struct cooker_plate const *plate, 
	char const *program_file,
	size_t const name_count, char const *const *const kernel_names,
	enum cooker_verbosity const v
) {
	enum cooker_state ret = CS_OK;
	char const *message = NULL;
	cl_int status; // For error handling

	char *source;
	if (v & CV_INFO)
		printf("Program source is: %s\n", program_file);

	if (read_source(program_file, &source, v & CV_ERROR))
		goto failure_readsource;
	
	// Create a program. 
	char const *opencl_is_stoopid = source;
	cl_program const program = clCreateProgramWithSource(
		plate->context, 1, &opencl_is_stoopid, NULL, &status
	);
	if (status != CL_SUCCESS)
		goto failure_createprogram;
	
	// Build (compile & link) the program for the device.
	if (clBuildProgram(
			program, 1, &plate->device, NULL, NULL, NULL
		) != CL_SUCCESS
	)
		goto failure_build;
	
	cl_kernel *const kernels = malloc(name_count * sizeof(cl_kernel));
	if (!kernels)
		goto failure_kernels_alloc;
	size_t kernel_count = 0;

	// Create a kernel for each requested kernel
	while (kernel_count < name_count) {
		kernels[kernel_count] = clCreateKernel(
			program, kernel_names[kernel_count], &status
		);
		if (status != CL_SUCCESS)
			goto failure_kernels_init;
		++kernel_count;
	}

	// Assign output fields
	dish->program = program;
	dish->source = source;
	dish->kernels = kernels;
	dish->kernel_count = kernel_count;

	return ret;

failure_readsource:
	message = "Could not read program source.";
	ret = CS_IO;
	goto cleanup_exit;
failure_createprogram:
	message = "Could not create OpenCL program.";
	ret = CS_CL;
	goto cleanup_source;
failure_build:
	message = "Program compilation failed.";
	ret = CS_CL;
	build_print(program, plate->device);
	goto cleanup_program;
failure_kernels_alloc:
	message = "Could not allocate memory for kernels.";
	ret = CS_MEMORY;
	goto cleanup_program;
failure_kernels_init:
	message = "Coult nod create OpenCL kernels.";
	ret = CS_CL;
	goto cleanup_kernels;

cleanup_kernels:
	while (kernel_count--)
		clReleaseKernel(kernels[kernel_count]);
	free(kernels);
	
cleanup_program:
	clReleaseProgram(program);

cleanup_source:
	free(source);

cleanup_exit:
	if (v & CV_ERROR)
		puts(message); 

	return ret;
}

