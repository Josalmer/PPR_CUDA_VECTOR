#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;
// Blocksize
#define BLOCKSIZE 1024

//*************************************************
// GLOBAL MEMORY  VERSION OF THE ALGORITHM
// ************************************************
__global__ void vectorNS(float *in, float *out, int n) {

  int i = threadIdx.x + blockDim.x * blockIdx.x + 2;
  int iB = i - 2;
  if (iB < n) {
    float Aim2 = in[i - 2];
    float Aim1 = in[i - 1];
    float Ai = in[i];
    float Aip1 = in[i + 1];
    float Aip2 = in[i + 2];
    out[iB] = (pow(Aim2, 2) + 2.0 * pow(Aim1, 2) + pow(Ai, 2) - 3.0 * pow(Aip1, 2) + 5.0 * pow(Aip2, 2)) / 24.0;
  }
}

//*************************************************
// TILING VERSION  (USES SHARED MEMORY) OF THE ALGORITHM
// ************************************************
__global__ void vectorS(float *in, float *out, int n) {
  int li = threadIdx.x + 2;                           //local index in shared memory vector
  int gi = blockDim.x * blockIdx.x + threadIdx.x + 2; // global memory index
  int lstart = 0;
  int lend = BLOCKSIZE + 2;
  __shared__ float s_phi[BLOCKSIZE + 4]; //shared mem. vector

  // Load Tile in shared memory
  if (gi < n + 3) {
    s_phi[li] = in[gi];
  }

  if (threadIdx.x == 0) { // First Thread (in the current block)
    s_phi[lstart] = in[gi - 2];
    s_phi[lstart + 1] = in[gi - 1];
  }

  if (threadIdx.x == BLOCKSIZE - 1) { // Last Thread
    if (gi >= n + 1) {                // Last Block
      s_phi[(n + 2) % BLOCKSIZE] = in[n + 2];
    } else {
      s_phi[lend - 1] = in[gi + 1];
      s_phi[lend] = in[gi + 2];
    }
  }
  __syncthreads();

  if (gi < n + 2) {
    out[gi] = (s_phi[li - 2] * s_phi[li - 2] + 2 * s_phi[li - 1] * s_phi[li - 1] + s_phi[li] * s_phi[li] - 3 * s_phi[li + 1] * s_phi[li +1] + 5 * s_phi[li + 2] * s_phi[li + 2]) / 24;
  }
}

//**************************************************************************
// FIND MAX IN VECTOR
__global__ void reduceMax(float * V_in, float * V_out, const int N) {
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : -1);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1) {
	  if (tid < s) {
      if(sdata[tid] < sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
      }
	  }
	  __syncthreads();
	}
	if (tid == 0) {
		V_out[blockIdx.x] = sdata[0];
	}
}

//**************************************************************************
int main(int argc, char *argv[]) {
  //******************************
  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if (err != cudaSuccess) {
    cout << "ERRORRR" << endl;
  }
  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

  int N;
  if (argc != 2) {
    cout << "Uso: transformacion Num_elementos  " << endl;
    return (0);
  }
  else {
    N = atoi(argv[1]);
  }

  //* pointers to host memory */
  float *A, *B;

  //* Allocate arrays a, b and c on host*/
  A = new float[N + 4];
  B = new float[N];
  float mx; // maximum of B

  //* Initialize array A */
  for (int i = 2; i < N + 2; i++)
    A[i] = (float)(1 - (i % 100) * 0.001);

  // Impose Boundary Conditions
  A[0] = 0.0;
  A[1] = 0.0;
  A[N + 2] = 0.0;
  A[N + 3] = 0.0;

  //**************************
  // GPU phase
  //**************************
  float *B_GPU = new float[N];

  int Nsize = N * sizeof(float);
  int NsizeWithBound = (N + 4) * sizeof(float);
  // Allocation in device mem
  float *A_GPU = NULL;
  err = cudaMalloc((void **)&A_GPU, NsizeWithBound);
  if (err != cudaSuccess) {
    cout << "ALLOCATION ERROR" << endl;
  }
  float *out = NULL;
  err = cudaMalloc((void **)&out, Nsize);
  if (err != cudaSuccess) {
    cout << "ALLOCATION ERROR" << endl;
  }

  // Take initial time
  cout << "Start GPU" << endl;
  double gt1 = clock();

  // Copy A values to device memory
  err = cudaMemcpy(A_GPU, A, NsizeWithBound, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    cout << "GPU COPY ERROR" << endl;
  }

  int blocksPerGrid = (int)ceil((float)(N) / BLOCKSIZE);

  cout << endl;
  // ********* Kernel Launch ************************************
  vectorNS<<<blocksPerGrid, BLOCKSIZE>>>(A_GPU, out, N);
  // ************************************************************

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel! %d \n", err);
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(B_GPU, out, Nsize, cudaMemcpyDeviceToHost);

  double Tgpu = clock();
  Tgpu = (Tgpu - gt1) / CLOCKS_PER_SEC;
  cout << "End GPU" << endl;

  //**************************
  // CPU phase
  //**************************
  cout << "Start CPU" << endl;
  // Time measurement
  double ct1 = clock();

  float Ai, Aim1, Aim2, Aip1, Aip2;
  // Compute B[i] and mx
  for (int i = 2; i < N + 2; i++) {
    const int iB = i - 2;
    Aim2 = A[i - 2];
    Aim1 = A[i - 1];
    Ai = A[i];
    Aip1 = A[i + 1];
    Aip2 = A[i + 2];
    B[iB] = (pow(Aim2, 2) + 2.0 * pow(Aim1, 2) + pow(Ai, 2) - 3.0 * pow(Aip1, 2) + 5.0 * pow(Aip2, 2)) / 24.0;
    mx = (iB == 0) ? B[0] : max(B[iB], mx);
  }

  double Tcpu = clock();
  Tcpu = (Tcpu - ct1) / CLOCKS_PER_SEC;
  cout << "End CPU" << endl;

  //**************************
  // CPU-GPU comparison and error checking
  //**************************

  int passed = 1;
  int i = 0;
  while (passed && i < N) {
    float diff = fabs(B[i] - B_GPU[i]);
    if (diff > 0) {
      passed = 0;
      cout << endl << i << endl;
      cout << "DIFF= " << diff << endl;
    }
    i++;
  }

  if (passed) {
    cout << "PASSED TEST !!!" << endl;
  }
  else {
    cout << "ERROR IN TEST !!!" << endl;
  }

	// c_d Maximum computation on GPU
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x));

	// Maximum vector on CPU
	float * vmax;
	vmax = (float*) malloc(numBlocks.x*sizeof(float));

	// Maximum vector  to be computed on GPU
	float *vmax_d; 
	cudaMalloc ((void **) &vmax_d, sizeof(float)*numBlocks.x);

	float smemSize = threadsPerBlock.x*sizeof(float);

	// Kernel launch to compute Minimum Vector
	reduceMax<<<numBlocks, threadsPerBlock, smemSize>>>(out,vmax_d, N);


	/* Copy data from device memory to host memory */
	cudaMemcpy(vmax, vmax_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

	// Perform final reduction in CPU
	float max_gpu = -1;
	for (int i=0; i<numBlocks.x; i++) {
		max_gpu =max(max_gpu,vmax[i]);
	}

  if (N < 16) {
    for (int i = 0; i < N; i++) {
      cout << "CPU[" << i << "] = " << B[i] << ", GPU[" << i << "] = " << B_GPU[i] << endl;
    }
  }
  cout << "................................." << endl;
  cout << "................................." << endl
       << "El valor máximo en B es (CPU):  " << mx << endl;
  cout << "................................." << endl
       << "El valor máximo en B es (GPU):  " << max_gpu << endl;
  cout << endl
       << "Tiempo gastado CPU= " << Tcpu << endl
       << endl;
  cout << endl
       << "Tiempo gastado GPU= " << Tgpu << endl
       << endl;
  cout << endl
       << "Speedup GPU= " << Tcpu / Tgpu << endl
       << endl;

  //* Free the memory */
  delete (A);
  delete (B);
  cudaFree(A_GPU);
  cudaFree(B_GPU);
  cudaFree(out);
}
