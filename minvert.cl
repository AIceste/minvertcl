// Kernels for matrix inversion using OpenCL
// Flexible version (reduction width as a parameter)
// There *is* some code repetition / bloat, but given the
// context really it is not worth to spend time cleaning it up.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned long cl_ulong;

// Single dimension kernel to assign diagonal to 1
__kernel void build_identity(
	__global double *const m, cl_ulong const n, cl_ulong const width
) {
	cl_ulong row_size = 2 * n;
	cl_ulong const off = get_global_id(0) * width;
	cl_ulong pos = n + off * row_size + off;
	cl_ulong const end = min(
		row_size * n, pos + row_size * width
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
	cl_ulong const n, cl_ulong const width
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const off = 2 * k * n;
	cl_ulong pos = off + id * width;
	cl_ulong const end = min(off + n, pos + width);
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
	cl_ulong const n, cl_ulong const width
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const step = depth * width;
	cl_ulong const left = k + id * step * width;

	// So long that we stay within buffer range nothing can 
	// go haray after running max_reduc_begin.
	cl_ulong const end = min(k + n, left + width * step);
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
	cl_ulong const n, cl_ulong const width
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const row_size = 2 * n;
	cl_ulong const col = indices[k] % n;
	cl_ulong const beg = id * width;
	cl_ulong const end = min(n, beg + width);

	__global cl_ulong *const save_to = indices + n;
	for (cl_ulong i = beg; i < end; ++i)
		save_to[i] = m[i * row_size + col];
}

// Divide line k by its maximum element
__kernel void normalise_pivot(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n, cl_ulong const width
) {
	cl_ulong const id = get_global_id(0);
	cl_ulong const off = 2 * k * n;
	cl_ulong pos = off + id * width;
	cl_ulong const end = min(off + n, pos + width);

	double const div = m[indices[n + k]];
	while (pos < end)
		m[pos++] /= div; // Woopsie no null check ^u^
}

__kernel void reduce(
	cl_ulong const k,
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n, cl_ulong const width
) {
	cl_ulong i = get_global_id(0);
	cl_ulong j = get_global_id(1);
	i += i >= k;
	cl_ulong const row_size = 2 * n;
	cl_ulong const end = min(row_size, j + width);
	cl_ulong const off = i * row_size;
	cl_ulong const pivot = k * row_size;

	double const mul = indices[n + i];
	do {
		m[off + j] -= mul * m[pivot + j];
	} while(++j < end);
}

__kernel void rearrange(
	__global double *const m, __global cl_ulong *indices,
	cl_ulong const n, cl_ulong const width
) {
	cl_ulong i = get_global_id(0);
	cl_ulong j = get_global_id(1);
	cl_ulong const row_size = 2 * n;
	cl_ulong const end = min(row_size, j + width);
	cl_ulong const src = i * row_size;
	cl_ulong const dst = (indices[i] % n) * row_size;

	do {
		m[dst + j] = m[src + j];
	} while(++j < end);
}
