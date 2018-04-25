#include <stdio.h>
#include <stdlib.h>

#define CSC(call) do { 		\
	cudaError_t e = call;	\
	if (e != cudaSuccess) {	\
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(0);			\
	}						\
} while(0)

#define MAX_R 1024

__global__ void kernel(double* dVector, int num)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	while (idx < num)
	{
		if (dVector[idx] < 0) 
		{
			dVector[idx] = -dVector[idx];	
		}

		idx += offset;
	}
}




class CUVector {
public:
	const int n;
	int size;	
	double* hVector;
	double* dVector;


	CUVector(int _n): n(_n) {
		size = sizeof(double) * n;
		
		hVector= (double*)malloc(size);
		CSC(cudaMalloc(&dVector, size));
	} 

	void init() {
		for (int i = 0; i < n; ++i)
			scanf("%lf", &hVector[i]);

		CSC(cudaMemcpy(dVector, hVector, size, cudaMemcpyHostToDevice));
	}

	void run_kernel() {
		kernel<<<256, 256>>>(dVector, n);

		CSC(cudaMemcpy(hVector, dVector, size, cudaMemcpyDeviceToHost));
	}

	void print() {
		for (int i = 0; i < n; ++i)
		printf("%.10e ", hVector[i]);

		printf("\n");
	}

	~CUVector(){
		CSC(cudaFree(dVector));
		free(hVector);		
	}
};

__constant__ float a[MAX_R * 2 + 1];

int main(void)
{
	int n;
	scanf("%d", &n);

	CUVector v = CUVector(n);

	v.init();
	v.run_kernel();
	v.print();

	return 0;
}
