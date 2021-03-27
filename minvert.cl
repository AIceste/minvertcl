// Kernels for matrix inversion using OpenCL
// Flexible version (reduction width as a parameter)

// Single dimension kernel to assign diagonal to 1
__kernel void build_identity(
	__global double *const m, size_t const n, size_t const width
) {
	size_t line_width = 2 * n;
	size_t const id = get_global_id(0);
	size_t const pos = n + id * line_width + id;
	size_t const end = min(
		line_width * n, n + (id + width) * line_width
	);
	while (pos < end) {
		m[pos] = 1.;
		pos += line_width + 1;
	}
}

// Reduction that finds the maximum of a line - by position
__kernel void max_reduc_begin(
	size_t const k,
	__global double *const m, __global size_t *indices,
	size_t const n, size_t const width
) {

}
__kernel void max_reduc(
	size_t const k,
	__global double *const m, __global size_t *indices,
	size_t const n, size_t const width
) {

}

// Store values of column k in indice buffer since they are 
// needed untouched for parallel line reduction
__kernel void store_column(
	size_t const k,
	__global double *const m, __global size_t *indices,
	size_t const n, size_t const width
) {

}

// Divide line k by its maximum element
__kernel void normalise_line(
	size_t const k,
	__global double *const m, __global size_t *indices,
	size_t const n, size_t const width
) {

}

__kernel void reduce(
	size_t const k,
	__global double *const m, __global size_t *indices,
	size_t const n, size_t const width
) {

}

__kernel void rearrange(
	size_t const k,
	__global double *const m, __global size_t *indices,
	size_t const n, size_t const width
) {

}
