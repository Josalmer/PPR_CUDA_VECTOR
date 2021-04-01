#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

int index(int i) { return i + 2; }
// Blocksize
#define BLOCKSIZE 1024
// Number of mesh points
int n = 60000;

//*************************************************
// Swap two pointers to float
// ************************************************
void swap_pointers(float **a, float **b) {
  float *tmp = *a;
  *a = *b;
  *b = tmp;
}

//*************************************************
// GLOBAL MEMORY  VERSION OF THE FD UPDATE
// ************************************************
__global__ void vectorNS(float *d_phi, float *d_phi_new, float cu, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x + 2;

  // Inner point update
  if (i < n + 3) {
    d_phi_new[i] = (d_phi[i - 2] * d_phi[i - 2] + 2 * d_phi[i - 1] * d_phi[i - 1] + d_phi[i] * d_phi[i] - 3 * d_phi[i + 1] * d_phi[i +1] + 5 * d_phi[i + 2] * d_phi[i + 2]) / 24;
  }

  // Boundary Conditions
  if (i == 2) {
    d_phi_new[0] = 0;
    d_phi_new[1] = 0;
  }
  if (i == n + 2) {
    d_phi_new[n + 3] = 0;
    d_phi_new[n + 4] = 0;
  }
}

//*************************************************
// TILING VERSION  (USES SHARED MEMORY) OF THE FD UPDATE
// ************************************************
__global__ void FD_kernel2(float *d_phi, float *d_phi_new, float cu, int n) {
  int li = threadIdx.x + 1;                           //local index in shared memory vector
  int gi = blockDim.x * blockIdx.x + threadIdx.x + 1; // global memory index
  int lstart = 0;
  int lend = BLOCKSIZE + 1;
  __shared__ float s_phi[BLOCKSIZE + 2]; //shared mem. vector
  float result;

  // Load Tile in shared memory
  if (gi < n + 2) {
    s_phi[li] = d_phi[gi];
  }

  if (threadIdx.x == 0) { // First Thread (in the current block)
    s_phi[lstart] = d_phi[gi - 1];
  }

  if (threadIdx.x == BLOCKSIZE - 1) { // Last Thread
    if (gi >= n + 1) {                // Last Block
      s_phi[(n + 2) % BLOCKSIZE] = d_phi[n + 2];
    } else {
      s_phi[lend] = d_phi[gi + 1];
    }
  }
  __syncthreads();

  if (gi < n + 2) {
    // Lax-Friedrichs Update
    result = 0.5 * ((s_phi[li + 1] + s_phi[li - 1]) - cu * (s_phi[li + 1] - s_phi[li - 1]));
    d_phi_new[gi] = result;
  }

  // Boundary Conditions
  if (gi == 1) {
    d_phi_new[0] = d_phi_new[1];
  }
  if (gi == n + 1) {
    d_phi_new[n + 2] = d_phi_new[n + 1];
  }
}

//******************************
//**** MAIN FUNCTION ***********

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

  cout << "Introduce number of points (1000-200000)" << endl;
  cin >> n;

  // Domain size (periodic)
  float l = 10.0;
  // Grid
  float dx = l / n;
  // Advecting velocity
  float u = 1.0;

  //Timestep size
  float dt = 0.8 * u * dx;
  float tend = 2.5;
  // Courant number
  float cu = u * dt / dx;

  //Number of steps to take
  int nsteps = (int)ceil(tend / dt);

  cout << "dx=" << dx << "...  dt= " << dt << "...Courant= " << cu << endl;
  cout << endl;
  cout << "Number of time steps=" << nsteps << endl;

  //Mesh Definition    blockDim.x*blockIdx.x
  float *phi = new float[n + 5];
  float *phi_new = new float[n + 5];
  float *phi_GPU = new float[n + 5];
  float xx[n + 2];

  for (int i = 0; i <= n; i++) {
    xx[i] = -5.0 + i * dx;
  }

  // Initial values for phi--> Gaussian
  for (int i = 0; i <= n; i++) {
    // Gaussian
    phi[index(i)] = (1.0 / (2.0 * M_PI * 0.16)) * exp(-0.5 * (pow((xx[i] - 0.5), 2) / 0.01));
  }

  //**************************
  // GPU phase
  //**************************
  int size = (n + 3) * sizeof(float);

  // Allocation in device mem. for d_phi
  float *d_phi = NULL;
  err = cudaMalloc((void **)&d_phi, size);
  if (err != cudaSuccess) {
    cout << "ALLOCATION ERROR" << endl;
  }
  // Allocation in device mem. for d_phi_new
  float *d_phi_new = NULL;
  err = cudaMalloc((void **)&d_phi_new, size);
  if (err != cudaSuccess) {
    cout << "ALLOCATION ERROR" << endl;
  }

  // Take initial time
  cout << "Start GPU" << endl;
  double t1 = clock();

  // Impose Boundary Conditions
  phi[index(-2)] = 0;
  phi[index(-1)] = 0;
  phi[index(n + 1)] = 0;
  phi[index(n + 2)] = 0;

  // Copy phi values to device memory
  err = cudaMemcpy(d_phi, phi, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    cout << "GPU COPY ERROR" << endl;
  }
  // *******************
  // Time Step Iteration
  // *******************
  for (int k = 0; k < nsteps; k++) {
    int blocksPerGrid = (int)ceil((float)(n + 2) / BLOCKSIZE);

    // ********* Kernel Launch ************************************
    vectorNS<<<blocksPerGrid, BLOCKSIZE>>>(d_phi, d_phi_new, cu, n);
    // ************************************************************

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel! %d \n", err);
      exit(EXIT_FAILURE);
    }
    swap_pointers(&d_phi, &d_phi_new);
  }

  cudaMemcpy(phi_GPU, d_phi, size, cudaMemcpyDeviceToHost);

  double Tgpu = clock();
  Tgpu = (Tgpu - t1) / CLOCKS_PER_SEC;
  cout << "End GPU" << endl;

  //**************************
  // CPU phase
  //**************************

  cout << "Start CPU" << endl;
  double t1cpu = clock();    
  
  // Impose Boundary Conditions
  phi[index(-2)] = 0;
  phi[index(-1)] = 0;
  phi[index(n + 1)] = 0;
  phi[index(n + 2)] = 0;
  for (int k = 0; k < nsteps; k++) {
    for (int i = 0; i <= n; i++) {
      //Lax-Friedrichs
      phi_new[index(i)] = (phi[index(i - 2)] * phi[index(i - 2)] + 2 * phi[index(i - 1)] *phi[index(i - 1)] + phi[index(i)] * phi[index(i)] - 3 * phi[index(i + 1)] * phi[index(i + 1)] + 5 * phi[index(i + 2)] * phi[index(i + 2)]) / 24;
    }
    swap_pointers(&phi, &phi_new);
  }

  double Tcpu = clock();
  Tcpu = (Tcpu - t1cpu) / CLOCKS_PER_SEC;
  cout << "End CPU" << endl;

  cout << endl;
  cout << "GPU Time= " << Tgpu << endl << endl;
  cout << "CPU Time= " << Tcpu << endl << endl;

  //**************************
  // CPU-GPU comparison and error checking
  //**************************

  int passed = 1;
  int i = 0;
  while (passed && i < n) {
    double diff = fabs((double)phi_GPU[index(i)] - (double)phi[index(i)]);
    if (diff > 1.0e-5) {
      passed = 0;
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

  cout << endl;
  cout << "Speedup (T_CPU/T_GPU)= " << Tcpu / Tgpu << endl;

  return 0;
}
