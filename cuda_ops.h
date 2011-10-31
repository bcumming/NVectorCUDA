/* copyright Ben Cumming 2010
 *
 * Header for cuda operations that are required for implementation of the
 * sundials nvector_parallel module using CUDA to compute local
 * operations on the GPU
 */
#ifndef CUDA_OP_H
#define CUDA_OP_H

extern void cuda_sadd(int N, double s, double *lhs, double *x);
extern void cuda_sset(int N, double s, double *x);
extern void cuda_axpby(int N, double a, double b, double *lhs, double *u, double *v);
extern void cuda_mulvv(int N, double *lhs, double *u, double *v);
extern void cuda_sscale_assign(int N, double s, double *lhs, double *x);
extern void cuda_abs_assign(int N, double *lhs, double *x);
extern void cuda_asumxy(int N, double alpha, double *lhs, double *x, double *y);
extern void cuda_adifxy(int N, double alpha, double *lhs, double *x, double *y);
extern void cuda_axdify(int N, double alpha, double *lhs, double *x, double *y);
extern void cuda_gtcomp(int N, double alpha, double *lhs, double *x);
extern void cuda_inv(int N, double *lhs, double *x);
extern double cuda_wrms_sum(double *x, double *w, int N);
extern double cuda_wrms_mask_sum(double *x, double *w, double *mask, int N);
extern double cuda_min(double *x, int N);
extern double cuda_minquotient(double *x, double *d, int N);
extern double cuda_constr_mask(double *x, double *w, double *mask, int N);
extern size_t cuda_free_memory(void);
extern size_t cuda_total_memory(void);
extern int cuda_num_devices(void);
extern int HandleError( cudaError_t err );
extern int cuda_errhandle_cublas(cublasStatus err_stat);

#endif
