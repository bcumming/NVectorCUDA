#include <assert.h>
#include <cublas.h>
#include <iostream>
#include <limits>
#include <float.h>

#define BIG_DOUBLE DBL_MAX;

/********************************************************
   private functions
********************************************************/
const int threadsPerBlock = 512;
const int maxBlocksPerGrid = 128;

/********************************************************
   kernels
********************************************************/

// lhs = u .* v
__global__ void cu_mulvv(double* lhs, double* u, double* v, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = u[i] * v[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = u ./ v
__global__ void cu_divvv(double* lhs, double* u, double* v, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = u[i] / v[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = a*u + b*v
__global__ void cu_axpby(double* lhs, double a, double b, double* u, double* v, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = a*u[i] + b*v[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = a*x-y
__global__ void cu_axdify(double* lhs, double* x, double* y, double alpha, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = alpha*x[i] - y[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = x>=alpha ? 1 : 0
// tests if each entry in x is gte alpha{
__global__ void cu_gtcomp(double* lhs, double* x, double alpha, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = x[i]>=alpha ? 1. : 0.;
        i += blockDim.x * gridDim.x;
    }
}

// lhs = a*(x+y)
__global__ void cu_asumxy(double* lhs, double* x, double* y, double alpha, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = alpha*(x[i] + y[i]);
        i += blockDim.x * gridDim.x;
    }
}

// lhs = a*(x-y)
__global__ void cu_adifxy(double* lhs, double* x, double* y, double alpha, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = alpha*(x[i] - y[i]);
        i += blockDim.x * gridDim.x;
    }
}

// lhs[:] += s
__global__ void cu_sadd(double* lhs, double s, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] += s;
        i += blockDim.x * gridDim.x;
    }
}

// lhs = x(:) + s
__global__ void cu_sadd_assign(double* lhs, double *x, double s, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = x[i]+s;
        i += blockDim.x * gridDim.x;
    }
}

// lhs = 1./x
__global__ void cu_inv_assign(double* lhs, double *x, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = 1./x[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = 1./lhs
__global__ void cu_inv(double* lhs, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = 1./lhs[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = s*x
__global__ void cu_sscale_assign(double* lhs, double *x, double s, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = s*x[i];
        i += blockDim.x * gridDim.x;
    }
}

// lhs = abs(x)
__global__ void cu_abs_assign(double* lhs, double *x, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = abs(x[i]);
        i += blockDim.x * gridDim.x;
    }
}

// lhs[:] = s
__global__ void cu_sset(double* lhs, double s, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N){
        lhs[i] = s;
        i += blockDim.x * gridDim.x;
    }
} 

/////////////////////////////////////////////////////////
// reduction operations
/////////////////////////////////////////////////////////
template<unsigned int blockSize>
__global__ void cu_wrms(double* x, double *w, double *odata, int N)
{
    extern __shared__ double sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    double tmp=0;
    double tmp2;
    while (i < N)
    {         
        tmp2 = w[i]*x[i];
        tmp += tmp2*tmp2;
        // ensure we don't read out of bounds
        if (i + blockSize < N){
            tmp2 = w[i+blockSize]*x[i+blockSize];
            tmp += tmp2*tmp2;
        }
        i += gridSize;
    } 
    sdata[tid] = tmp; 

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s) 
            sdata[tid] = tmp = tmp + sdata[tid + s];
        __syncthreads();
    }

    // perform reduction over shared memory on the warp level
    // no need to explicitly synchronise over threads on the warp
    // as long as we tell the compiler that the shared memory is
    // volatile
    if (tid < 32)
    {
        volatile double *smem = sdata;
        if (blockSize >=  64) smem[tid] = tmp = tmp + smem[tid + 32];  
        if (blockSize >=  32) smem[tid] = tmp = tmp + smem[tid + 16];  
        if (blockSize >=  16) smem[tid] = tmp = tmp + smem[tid +  8];  
        if (blockSize >=   8) smem[tid] = tmp = tmp + smem[tid +  4];  
        if (blockSize >=   4) smem[tid] = tmp = tmp + smem[tid +  2];  
        if (blockSize >=   2) smem[tid] = tmp = tmp + smem[tid +  1];  
    }

    // write result for this block to global mem
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<unsigned int blockSize>
__global__ void cu_constr_mask(double* c, double *x, double *m, double *odata, int N)
{
    extern __shared__ double sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

    // just do this ugly - don't think this is used for any time-critical part
    // part of the code. And, besides, the original implementation isn't much
    // to look at.
    // This version accumlates the flags by addition
    double tmp=0;
    double tmp2;
    double tmp3;
    while (i < N)
    {         
        tmp2 = c[i];
        tmp = 1.;
        if( abs(tmp2)>1.5 && x[i]*tmp2<=0. )
            tmp = 0.;
        else if(abs(tmp2)>0.5 && x[i]*tmp2<0.)
            tmp = 0.;
        m[i] = tmp==0. ? 1. : 0.;

        // ensure we don't read out of bounds
        if (i + blockSize < N){
            tmp2 = c[i+blockSize];
            tmp3 = 1.;
            if( abs(tmp2)>1.5 && x[i+blockSize]*tmp2<=0. )
                tmp3 = 0.;
            else if(abs(tmp2)>0.5 && x[i+blockSize]*tmp2<0.)
                tmp3 = 0.;
            m[i+blockSize] = tmp3==0. ? 1. : 0.;
            tmp += tmp3;
        }

        i += gridSize;
    } 
    sdata[tid] = tmp; 

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s) 
            sdata[tid] = tmp = tmp + sdata[tid + s];
        __syncthreads();
    }

    // perform reduction over shared memory on the warp level
    // no need to explicitly synchronise over threads on the warp
    // as long as we tell the compiler that the shared memory is
    // volatile
    if (tid < 32)
    {
        volatile double *smem = sdata;
        if (blockSize >=  64) smem[tid] = tmp = tmp + smem[tid + 32];  
        if (blockSize >=  32) smem[tid] = tmp = tmp + smem[tid + 16];  
        if (blockSize >=  16) smem[tid] = tmp = tmp + smem[tid +  8];  
        if (blockSize >=   8) smem[tid] = tmp = tmp + smem[tid +  4];  
        if (blockSize >=   4) smem[tid] = tmp = tmp + smem[tid +  2];  
        if (blockSize >=   2) smem[tid] = tmp = tmp + smem[tid +  1];  
    }

    // write result for this block to global mem
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<unsigned int blockSize>
__global__ void cu_wrms_mask(double* x, double *w, double *m, double *odata, int N)
{
    extern __shared__ double sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    double tmp=0;
    double tmp2;
    while (i < N)
    {         
        tmp2 = m[i]>0. ? w[i]*x[i] : 0.;
        tmp += tmp2*tmp2;
        // ensure we don't read out of bounds
        if (i + blockSize < N){
            tmp2 = m[i+blockSize]>0. ? w[i+blockSize]*x[i+blockSize] : 0.;
            tmp += tmp2*tmp2;
        }
        i += gridSize;
    } 
    sdata[tid] = tmp; 

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s) 
            sdata[tid] = tmp = tmp + sdata[tid + s];
        __syncthreads();
    }

    // perform reduction over shared memory on the warp level
    // no need to explicitly synchronise over threads on the warp
    // as long as we tell the compiler that the shared memory is
    // volatile
    if (tid < 32)
    {
        volatile double *smem = sdata;
        if (blockSize >=  64) smem[tid] = tmp = tmp + smem[tid + 32];  
        if (blockSize >=  32) smem[tid] = tmp = tmp + smem[tid + 16];  
        if (blockSize >=  16) smem[tid] = tmp = tmp + smem[tid +  8];  
        if (blockSize >=   8) smem[tid] = tmp = tmp + smem[tid +  4];  
        if (blockSize >=   4) smem[tid] = tmp = tmp + smem[tid +  2];  
        if (blockSize >=   2) smem[tid] = tmp = tmp + smem[tid +  1];  
    }

    // write result for this block to global mem
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<unsigned int blockSize>
__global__ void cu_minquotient(double* x, double *d, double *odata, int N)
{
    extern __shared__ double sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

    double tmp = BIG_DOUBLE;
    double tmpd, tmpx, tmpr;
    while (i < N)
    {         
        tmpd = d[i];
        tmpx = x[i];
        if(tmpd){
            tmpr = tmpx/tmpd;
            tmp = tmp>tmpr ? tmpr : tmp;
        }
        // ensure we don't read out of bounds
        if ((i + blockSize) < N){
            tmpd = d[i+blockSize];
            tmpx = x[i+blockSize];
            if(tmpd){
                tmpr = tmpx/tmpd;
                tmp = tmp>tmpr ? tmpr : tmp;
            }
        }
        i += gridSize;
    } 
    sdata[tid] = tmp; 

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s){
            tmpr = sdata[tid+s];
            tmp = tmp<tmpr ? tmp : tmpr;
            sdata[tid] = tmp;
        }
        __syncthreads();
    }

    // perform reduction over shared memory on the warp level
    // no need to explicitly synchronise over threads on the warp
    // as long as we tell the compiler that the shared memory is
    // volatile
    if (tid < 32)
    {
        volatile double *smem = sdata;
        if (blockSize >=  64){ tmpr=smem[tid+32]; tmp = (tmp<tmpr ? tmp : tmpr); smem[tid] = tmp; }
        if (blockSize >=  32){ tmpr=smem[tid+16]; tmp = (tmp<tmpr ? tmp : tmpr); smem[tid] = tmp; }
        if (blockSize >=  16){ tmpr=smem[tid+ 8]; tmp = (tmp<tmpr ? tmp : tmpr); smem[tid] = tmp; }
        if (blockSize >=   8){ tmpr=smem[tid+ 4]; tmp = (tmp<tmpr ? tmp : tmpr); smem[tid] = tmp; }
        if (blockSize >=   4){ tmpr=smem[tid+ 2]; tmp = (tmp<tmpr ? tmp : tmpr); smem[tid] = tmp; }
        if (blockSize >=   2){ tmpr=smem[tid+ 1]; tmp = (tmp<tmpr ? tmp : tmpr); smem[tid] = tmp; }
    }

    // write result for this block to global mem
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<unsigned int blockSize>
__global__ void cu_min(double* x, double *odata, int N)
{
    extern __shared__ double sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

    double tmp = BIG_DOUBLE;
    double tmp2;
    while (i < N)
    {         
        tmp = tmp>x[i] ? x[i] : tmp;
        // ensure we don't read out of bounds
        if ((i + blockSize) < N){
            tmp = tmp>x[i+blockSize] ? x[i+blockSize] : tmp;
        }
        i += gridSize;
    } 
    sdata[tid] = tmp; 

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s){
            tmp2 = sdata[tid+s];
            tmp = tmp<tmp2 ? tmp : tmp2;
            sdata[tid] = tmp;
        }
        __syncthreads();
    }

    // perform reduction over shared memory on the warp level
    // no need to explicitly synchronise over threads on the warp
    // as long as we tell the compiler that the shared memory is
    // volatile
    if (tid < 32)
    {
        volatile double *smem = sdata;
        if (blockSize >=  64){ tmp2=smem[tid+32]; tmp = (tmp<tmp2 ? tmp : tmp2); smem[tid] = tmp; }
        if (blockSize >=  32){ tmp2=smem[tid+16]; tmp = (tmp<tmp2 ? tmp : tmp2); smem[tid] = tmp; }
        if (blockSize >=  16){ tmp2=smem[tid+ 8]; tmp = (tmp<tmp2 ? tmp : tmp2); smem[tid] = tmp; }
        if (blockSize >=   8){ tmp2=smem[tid+ 4]; tmp = (tmp<tmp2 ? tmp : tmp2); smem[tid] = tmp; }
        if (blockSize >=   4){ tmp2=smem[tid+ 2]; tmp = (tmp<tmp2 ? tmp : tmp2); smem[tid] = tmp; }
        if (blockSize >=   2){ tmp2=smem[tid+ 1]; tmp = (tmp<tmp2 ? tmp : tmp2); smem[tid] = tmp; }
    }

    // write result for this block to global mem
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

/********************************************************
 ********************************************************
   externally callable functions
*********************************************************
********************************************************/

// returns the amount of free memory on CUDA device
extern "C" size_t cuda_free_memory(){
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    return free;
}

// returns the total amount of memory on CUDA device
extern "C" size_t cuda_total_memory(){
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    return total;
}

// number of CUDA devices available
extern "C" int cuda_num_devices(){
    int device_count = 0;
    assert (cudaGetDeviceCount(&device_count) == cudaSuccess);
    return device_count;
}

extern "C" int HandleError( cudaError_t err ){
    if (err != cudaSuccess)
        std::cerr << "cuda error : " << cudaGetErrorString( err ) << std::endl;
    return err==cudaSuccess;
}


extern "C" int cuda_errhandle_cublas(cublasStatus err_stat){
    switch( err_stat ){
        case CUBLAS_STATUS_NOT_INITIALIZED:
            std::cerr << "Cublas error : CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            std::cerr << "Cublas error : CUBLAS_STATUS_ALLOC_FAILED" << std::endl;
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            std::cerr << "Cublas error : CUBLAS_STATUS_INVALID_VALUE" << std::endl;
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            std::cerr << "Cublas error : CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            std::cerr << "Cublas error : CUBLAS_STATUS_MAPPING_ERROR" << std::endl;
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            std::cerr << "Cublas error : CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            std::cerr << "Cublas error : CUBLAS_STATUS_INTERNAL_ERROR" << std::endl;
            break;
    }
    return err_stat==CUBLAS_STATUS_SUCCESS;
}

extern "C" void cuda_sadd(int N, double s, double *lhs, double *x){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    if(x==lhs)
        cu_sadd<<<blocksPerGrid, threadsPerBlock>>>
            (lhs, s, N);
    else
        cu_sadd_assign<<<blocksPerGrid, threadsPerBlock>>>
            (lhs, x, s, N);
}

extern "C" void cuda_inv(int N, double *lhs, double *x){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    if(x==lhs)
        cu_inv<<<blocksPerGrid, threadsPerBlock>>>
            (lhs, N);
    else
        cu_inv_assign<<<blocksPerGrid, threadsPerBlock>>>
            (lhs, x, N);
}

extern "C" void cuda_sscale_assign(int N, double s, double *lhs, double *x){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_sscale_assign<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, x, s, N);
}

extern "C" void cuda_sset(int N, double alpha, double *x){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_sset<<<blocksPerGrid, threadsPerBlock>>>
        (x, alpha, N);
}

//lhs = a*x + b*y
extern "C" void cuda_axpby(int N, double a, double b, double *lhs, double *u, double *v){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_axpby<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, a, b, u, v, N);
}

extern "C" void cuda_mulvv(int N, double *lhs, double *u, double *v){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_mulvv<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, u, v, N);
}

extern "C" void cuda_divvv(int N, double *lhs, double *u, double *v){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_divvv<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, u, v, N);
}

extern "C" void cuda_abs_assign(int N, double *lhs, double *x){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_abs_assign<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, x, N);
}

//lhs = alpha*(x+y)
extern "C" void cuda_asumxy(int N, double alpha, double *lhs, double *x, double *y){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_asumxy<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, x, y, alpha, N);
}
//lhs = alpha*(x-y)
extern "C" void cuda_adifxy(int N, double alpha, double *lhs, double *x, double *y){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_adifxy<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, x, y, alpha, N);
}
//lhs = alpha*(x)-y
extern "C" void cuda_axdify(int N, double alpha, double *lhs, double *x, double *y){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_axdify<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, x, y, alpha, N);
}

extern "C" void cuda_gtcomp(int N, double alpha, double *lhs, double *x){
    int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;
    blocksPerGrid = blocksPerGrid > maxBlocksPerGrid ? maxBlocksPerGrid : blocksPerGrid;
    cu_gtcomp<<<blocksPerGrid, threadsPerBlock>>>
        (lhs, x, alpha, N);
}
/******************************************
 *  REDUCTIONS
 ******************************************/
// helper functions

// find the next power of 2
// maxes out at 16 bit because 2^16=65536
// is the max number of threads in a block
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

// returns sum( (x[i]*w[i])^2 )
extern "C" double cuda_wrms_mask_sum(double *x, double *w, double *mask, int N)
{
    const int max_threads = 512;
    const int max_blocks = 128;

    //int threads = (N < max_threads) ? nextPow2(N) : max_threads;
    //int blocks = (N + threads - 1) / threads;
    int threads = (N < max_threads*2) ? nextPow2((N + 1)/ 2) : max_threads;
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);
    blocks = max_blocks<blocks ? max_blocks : blocks;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // allocate memory here for now, but a buffer should be preallocated at startup
    double *d_odata;
    cudaMalloc((void **)&d_odata, blocks * sizeof(double));

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    // call the kernel
    switch(threads){
        case 512 :
            cu_wrms_mask<512><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 256 :
            cu_wrms_mask<256><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 128 :
            cu_wrms_mask<128><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 64 :
            cu_wrms_mask<64><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 32 :
            cu_wrms_mask<32><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 16 :
            cu_wrms_mask<16><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 8 :
            cu_wrms_mask<8><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 4 :
            cu_wrms_mask<4><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 2 :
            cu_wrms_mask<2><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 1 :
            cu_wrms_mask<1><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
    }

    // now we can just hit the partial reduction with a cublas call
    double sum_squares = cublasDasum (blocks, d_odata, 1);

    // free up temporary device memory
    cudaFree( d_odata );

    return sum_squares;
}

// 
extern "C" double cuda_constr_mask(double *x, double *w, double *mask, int N)
{
    const int max_threads = 512;
    const int max_blocks = 128;

    //int threads = (N < max_threads) ? nextPow2(N) : max_threads;
    //int blocks = (N + threads - 1) / threads;
    int threads = (N < max_threads*2) ? nextPow2((N + 1)/ 2) : max_threads;
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);
    blocks = max_blocks<blocks ? max_blocks : blocks;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // allocate memory here for now, but a buffer should be preallocated at startup
    double *d_odata;
    cudaMalloc((void **)&d_odata, blocks * sizeof(double));

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    // call the kernel
    switch(threads){
        case 512 :
            cu_constr_mask<512><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 256 :
            cu_constr_mask<256><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 128 :
            cu_constr_mask<128><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 64 :
            cu_constr_mask<64><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 32 :
            cu_constr_mask<32><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 16 :
            cu_constr_mask<16><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 8 :
            cu_constr_mask<8><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 4 :
            cu_constr_mask<4><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 2 :
            cu_constr_mask<2><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
        case 1 :
            cu_constr_mask<1><<< dimGrid, dimBlock, smemSize >>>(x, w, mask, d_odata, N);
            break;
    }

    // now we can just hit the partial reduction with a cublas call to find
    // the position of the max value, then copy that value to the host
    int pos  = cublasIdamax(blocks, d_odata, 1)-1;
    double max_val;
    cudaMemcpy(&max_val, d_odata+pos, sizeof(double), cudaMemcpyDeviceToHost);

    // free up temporary device memory
    cudaFree( d_odata );

    return (max_val>=1.) ? 1. : 0.;
}
// returns sum( (x[i]*w[i])^2 )
extern "C" double cuda_wrms_sum(double *x, double *w, int N)
{
    const int max_threads = 512;
    const int max_blocks = 128;

    //int threads = (N < max_threads) ? nextPow2(N) : max_threads;
    //int blocks = (N + threads - 1) / threads;
    int threads = (N < max_threads*2) ? nextPow2((N + 1)/ 2) : max_threads;
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);
    blocks = max_blocks<blocks ? max_blocks : blocks;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // allocate memory here for now, but a buffer should be preallocated at startup
    double *d_odata;
    cudaMalloc((void **)&d_odata, blocks * sizeof(double));

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    // call the kernel
    switch(threads){
        case 512 :
            cu_wrms<512><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 256 :
            cu_wrms<256><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 128 :
            cu_wrms<128><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 64 :
            cu_wrms<64><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 32 :
            cu_wrms<32><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 16 :
            cu_wrms<16><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 8 :
            cu_wrms<8><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 4 :
            cu_wrms<4><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 2 :
            cu_wrms<2><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
        case 1 :
            cu_wrms<1><<< dimGrid, dimBlock, smemSize >>>(x, w, d_odata, N);
            break;
    }

    // now we can just hit the partial reduction with a cublas call
    double sum_squares = cublasDasum (blocks, d_odata, 1);

    // free up temporary device memory
    cudaFree( d_odata );

    return sum_squares;
}

// returns min_i( x[i] )
extern "C" double cuda_min(double *x, int N)
{
    const int max_threads = 512;
    const int max_blocks = 128;

    //int threads = (N < max_threads) ? nextPow2(N) : max_threads;
    //int blocks = (N + threads - 1) / threads;
    int threads = (N < max_threads*2) ? nextPow2((N + 1)/ 2) : max_threads;
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);
    blocks = max_blocks<blocks ? max_blocks : blocks;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // allocate memory here for now, but a buffer should be preallocated at startup
    double *d_odata;
    cudaMalloc((void **)&d_odata, blocks * sizeof(double));

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    // call the kernel
    switch(threads){
        case 512 :
            cu_min<512><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 256 :
            cu_min<256><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 128 :
            cu_min<128><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 64 :
            cu_min<64><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 32 :
            cu_min<32><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 16 :
            cu_min<16><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 8 :
            cu_min<8><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 4 :
            cu_min<4><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 2 :
            cu_min<2><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
        case 1 :
            cu_min<1><<< dimGrid, dimBlock, smemSize >>>(x, d_odata, N);
            break;
    }

    // recursively reduce the buffer
    while(blocks>1){
        int n = blocks;
        threads = nextPow2((blocks + 1)/ 2);
        blocks = (blocks + (threads * 2 - 1)) / (threads * 2);
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);

        int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

        // call the kernel
        switch(threads){
            case 512 :
                cu_min<512><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 256 :
                cu_min<256><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 128 :
                cu_min<128><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 64 :
                cu_min<64><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 32 :
                cu_min<32><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 16 :
                cu_min<16><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 8 :
                cu_min<8><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 4 :
                cu_min<4><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 2 :
                cu_min<2><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 1 :
                cu_min<1><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
        } 
    }

    double minr;
    cudaMemcpy(&minr, d_odata, sizeof(double), cudaMemcpyDeviceToHost);
    // free up temporary device memory
    cudaFree( d_odata );

    return minr;
}

// returns min_i( x[i]/d[i] )
extern "C" double cuda_minquotient(double *x, double *d, int N)
{
    const int max_threads = 512;
    const int max_blocks = 128;

    int threads = (N < max_threads*2) ? nextPow2((N + 1)/ 2) : max_threads;
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);
    blocks = max_blocks<blocks ? max_blocks : blocks;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // allocate memory here for now, but a buffer should be preallocated at startup
    double *d_odata;
    cudaMalloc((void **)&d_odata, blocks * sizeof(double));

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    // call the kernel
    switch(threads){
        case 512 :
            cu_minquotient<512><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 256 :
            cu_minquotient<256><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 128 :
            cu_minquotient<128><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 64 :
            cu_minquotient<64><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 32 :
            cu_minquotient<32><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 16 :
            cu_minquotient<16><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 8 :
            cu_minquotient<8><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 4 :
            cu_minquotient<4><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 2 :
            cu_minquotient<2><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
        case 1 :
            cu_minquotient<1><<< dimGrid, dimBlock, smemSize >>>(x, d, d_odata, N);
            break;
    }

    // recursively reduce the buffer
    // we perform the reduction using cu_min
    while(blocks>1){
        int n = blocks;
        threads = nextPow2((blocks + 1)/ 2);
        blocks = (blocks + (threads * 2 - 1)) / (threads * 2);
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);

        int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

        // call the kernel
        switch(threads){
            case 512 :
                cu_min<512><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 256 :
                cu_min<256><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 128 :
                cu_min<128><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 64 :
                cu_min<64><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 32 :
                cu_min<32><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 16 :
                cu_min<16><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 8 :
                cu_min<8><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 4 :
                cu_min<4><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 2 :
                cu_min<2><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
            case 1 :
                cu_min<1><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n);
                break;
        } 
    }
    // now we can just hit the partial reduction with a cublas call
    //double sum_squares = cublasDasum (blocks, d_odata, 1);

    double minr;
    cudaMemcpy(&minr, d_odata, sizeof(double), cudaMemcpyDeviceToHost);
    // free up temporary device memory
    cudaFree( d_odata );

    return minr;
}

