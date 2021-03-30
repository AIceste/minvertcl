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
	cl_ulong row_size = 2 * n;
	cl_ulong const off = get_global_id(0) * REDUCTION_WIDTH;
	cl_ulong pos = n + off * row_size + off;
	cl_ulong const end = min(
		row_size * n, pos + row_size * REDUCTION_WIDTH
	);
	while (pos < end) {
		m[pos] = 1.;
		pos += row_size + 1;
	}
}

// Reduction that finds the maximum of line k - by position
__kernel void max_reduc_begin(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const off = 2 * k * n;
	cl_ulong pos = off + id * REDUCTION_WIDTH;
	cl_ulong const end = min(off + n, pos + REDUCTION_WIDTH);
	cl_ulong max = pos;
	double val = fabs(m[pos]);
	while (++pos < end) {
		double const tmp = fabs(m[pos]);
		if (tmp > val) {
			max = pos;
			val = tmp;
		}
	}
	indices[k + id] = max;
}
__kernel void max_reduc(
	cl_ulong const k, cl_ulong const depth,
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const step = REDUCTION_WIDTH_POW(depth);
	cl_ulong const left = k + id * step * REDUCTION_WIDTH;

	// So long that we stay within buffer range nothing can 
	// go haray after running max_reduc_begin.
	cl_ulong const end = min(k + n, left + REDUCTION_WIDTH * step);
	cl_ulong pos = left;
	cl_ulong max = indices[pos];
	double val = fabs(m[max]);
	while ((pos += step) < end) {
		double const tmp = fabs(m[indices[pos]]);
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
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const row_size = 2 * n;
	cl_ulong const col = indices[k] % n;
	cl_ulong const beg = id * REDUCTION_WIDTH;
	cl_ulong const end = min(n, beg + REDUCTION_WIDTH);

	__global double *const save_to = (__global double*)indices + n;
	for (cl_ulong i = beg; i < end; ++i)
		save_to[i] = m[i * row_size + col];
}

// Divide line k by its maximum element
__kernel void normalise_pivot(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const row_size = 2 * n;
	cl_ulong const off = k * row_size;
	cl_ulong pos = off + id * REDUCTION_WIDTH;
	cl_ulong const end = min(off + row_size, pos + REDUCTION_WIDTH);

	double const div = ((__global double*)indices)[n + k];
	while (pos < end)
		m[pos++] /= div; // Woopsie no null check ^u^
}

__kernel void reduce(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	cl_ulong i = get_global_id(0);
	cl_ulong j = get_global_id(1) * REDUCTION_WIDTH;
	i += (i >= k);
	cl_ulong const row_size = 2 * n;
	cl_ulong const end = min(row_size, j + REDUCTION_WIDTH);
	cl_ulong const off = i * row_size;
	cl_ulong const pivot = k * row_size;

	double const mul = ((__global double*)indices)[n + i];
	do {
		m[off + j] -= mul * m[pivot + j];
	} while(++j < end);
}

__kernel void rearrange(
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n
) {
	cl_ulong i = get_global_id(0);
	cl_ulong j = get_global_id(1) * REDUCTION_WIDTH;
	cl_ulong const row_size = 2 * n;
	cl_ulong const end = min(n, j + REDUCTION_WIDTH);
	cl_ulong const src = i * row_size + n;
	cl_ulong const dst = (indices[i] % n) * row_size;

	do {
		m[dst + j] = m[src + j];
	} while(++j < end);
}
