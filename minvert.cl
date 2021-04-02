// Kernels for matrix inversion using OpenCL
// Flexible version (reduction REDUCTION_WIDTH as a parameter)
// There *is* some code repetition / bloat, but given the
// context really it is not worth to spend time cleaning it up.

#include "reduction_width.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned long cl_ulong;

// Single dimension kernel to assign diagonal to 1
__kernel void build_identity(
	__global double *const m, cl_ulong const n
) {
	cl_ulong const size = n * n;
	cl_ulong const ij = get_global_id(0) * REDUCTION_WIDTH;
	cl_ulong pos = size + ij * n + ij;
	cl_ulong const end = min(
		2 * size, pos + n * REDUCTION_WIDTH
	);
	while (pos < end) {
		m[pos] = 1.;
		pos += n + 1;
	}
}

// Reduction that finds the maximum of line k - by position
__kernel void max_reduc_begin(
	cl_ulong const k,
	__global double const *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	__global double const *const row = m + k * n;
	cl_ulong const id = get_global_id(0);
	cl_ulong pos = id * REDUCTION_WIDTH;
	cl_ulong const end = min(n, pos + REDUCTION_WIDTH);
	cl_ulong max = pos;
	double val = fabs(row[max]);
	while (++pos < end) {
		double const tmp = fabs(row[pos]);
		if (tmp > val) {
			max = pos;
			val = tmp;
		}
	}
	// For debug purpose, puts all value visibly at the end of the buffer.
	indices[k + id + get_global_size(0)] = pos
		+ 1000000                        * id * REDUCTION_WIDTH
		+ 1000                           * end
		+ 1000000000 * get_global_size(0)
	;
	indices[k + id] = max;
}
__kernel void max_reduc(
	cl_ulong const k, cl_ulong const depth,
	__global double const *const m, __global cl_ulong *const indices,
	cl_ulong const n
) {
	__global double const *const row = m + k * n;
	cl_ulong const id = get_global_id(0);
	cl_ulong const step = REDUCTION_WIDTH_POW(depth);
	cl_ulong const kernel_width = step * REDUCTION_WIDTH;
	cl_ulong const left = k + id * kernel_width;

	// So long that we stay within buffer range nothing can 
	// go haray after running max_reduc_begin.
	cl_ulong const end = min(2 * n, left + kernel_width);
	cl_ulong pos = left;
	cl_ulong max = indices[pos];
	double val = fabs(row[max]);
	while ((pos += step) < end) {
		double const tmp = fabs(row[indices[pos]]);
		if (tmp > val) {
			max = pos;
			val = tmp;
		}
	}
	indices[left] = max;
}

// Store values of column k in indice buffer since they are 
// needed untouched for parallel line reduction
__kernel void store_column(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *const indices,
	cl_ulong const n
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const col = indices[k] % n;
	cl_ulong const beg = id * REDUCTION_WIDTH;
	cl_ulong const end = min(n, beg + REDUCTION_WIDTH);

	__global double *const save_to = (__global double*)indices + n;
	for (cl_ulong i = beg; i < end; ++i)
		save_to[i] = m[i * n + col];
}

// Divide line k by its maximum element
__kernel void normalise_pivot(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *const indices,
	cl_ulong const n
) {
	cl_ulong const size = n * n;
	cl_ulong const id = get_global_id(1);
	cl_ulong const off = get_global_id(0) * size + k * n;
	cl_ulong pos = off + id * REDUCTION_WIDTH;
	cl_ulong const end = min(off + n, pos + REDUCTION_WIDTH);

	double const div = ((__global double*)indices)[n + k];
	while (pos < end)
		m[pos++] /= div; // Woopsie no null check ^u^
}

__kernel void reduce(
	cl_ulong const k,
	__global double *m, __global cl_ulong const *const indices,
	cl_ulong const n
) {
	m += get_global_id(0) * n * n;
	cl_ulong i = get_global_id(1);
	cl_ulong j = get_global_id(2) * REDUCTION_WIDTH;
	i += (i >= k);
	cl_ulong const end = min(n, j + REDUCTION_WIDTH);
	__global double *const row = m + i * n;
	__global double const *const pivot = m + k * n;

	double const mul = ((__global double*)indices)[n + i];
	do {
		row[j] -= mul * pivot[j];
	} while(++j < end);
}

__kernel void rearrange(
	__global double *const m, __global cl_ulong const *const indices,
	cl_ulong const n
) {
	cl_ulong const size = n * n;
	cl_ulong i = get_global_id(0);
	cl_ulong j = get_global_id(1) * REDUCTION_WIDTH;
	cl_ulong const end = min(n, j + REDUCTION_WIDTH);
	__global double const *const src = m + size + i * n;
	__global double *const dst = m + (indices[i] % n) * n;

	do {
		dst[j] = src[j];
	} while(++j < end);
}
