#include "iohelper.h"

#define NAMELEN 255

#define CSC(call) do { 		\
	cudaError_t e = call;	\
	if (e != cudaSuccess) {	\
		fprintf(stderr, "ERROR: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(0);			\
	}						\
} while(0)
      
 
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *dev_data, int weight, int hieght, int radius, float *dev_a, bool isColumns){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int offsetx = gridDim.x * blockDim.x;
	int offsety = gridDim.y * blockDim.y;


	uchar4 pixel;
	float red, green, blue;

	for(int x = idx; x < weight; x += offsetx){
		for(int y = idy; y < hieght; y += offsety){
			red = 0.0;
			green = 0.0;
			blue = 0.0;

			for(int i = -radius; i <=radius; i++){

				if (isColumns) {
				pixel = tex2D(tex, x + i, y);
				}
				else {
				pixel = tex2D(tex, x, y + i);
				}

				red += dev_a[radius + i] * pixel.x;
				green += dev_a[radius + i] * pixel.y;
				blue += dev_a[radius + i] * pixel.z;
			}

			dev_data[y * weight + x] = make_uchar4(red, green, blue, 0.0);
		}
	}
}
  

class CUGaussianBlur {
public:

	float *dev_a;
	cudaArray *arr;
	uchar4 *data;
	uchar4 *dev_data;
	cudaChannelFormatDesc ch;

	int weight;
	int hieght;
	int radius;

	CUGaussianBlur(IOHelper<uchar4>  *reader) 
	{

		weight = (*reader).GetWeight();
		hieght = (*reader).GetHeight();
		radius = (*reader).GetRadius();


		data = (uchar4 *)malloc(sizeof(*data)  * weight * hieght);
		 (*reader).ReadData(data);
 
		int n = 2 * radius + 1;
		float sum = 0.0;

		float a[n]; 

		for(int i = -radius; i <= radius; i++) {
			a[i + radius] = exp(-1.0 * (i * i) / (2 * radius * radius));
			sum += a[i + radius];
		}

		for(int i = 0; i < n; i++){
			a[i] /= sum;
		}

		CSC(cudaMalloc(&dev_a, sizeof(float) * n));
		CSC(cudaMemcpy(dev_a, a, sizeof(float) * n, cudaMemcpyHostToDevice));

		
		ch = cudaCreateChannelDesc<uchar4>();
		CSC(cudaMallocArray(&arr, &ch, weight, hieght));
		CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(*data) * hieght * weight, cudaMemcpyHostToDevice));
 
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.channelDesc = ch;
		tex.filterMode = cudaFilterModePoint;
		tex.normalized = false;

		CSC(cudaBindTextureToArray(tex, arr, ch));
		CSC(cudaMalloc(&dev_data, sizeof(*dev_data) * weight * hieght));

	} 
 
	void run_kernel() {

		kernel <<<dim3(8,16), dim3(8,32) >>> (dev_data, weight, hieght, radius, dev_a, false);

		CSC(cudaUnbindTexture(tex));
		CSC(cudaMemcpyToArray(arr, 0, 0, dev_data, sizeof(uchar4) * weight * hieght, cudaMemcpyDeviceToDevice));
		CSC(cudaBindTextureToArray(tex, arr, ch));

		kernel <<< dim3(8,16), dim3(16,32) >>> (dev_data, weight, hieght, radius, dev_a, true);

		CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * weight * hieght, cudaMemcpyDeviceToHost));

	}

	~CUGaussianBlur(){

		free(data);

		CSC(cudaUnbindTexture(tex));
		CSC(cudaFreeArray(arr));
		CSC(cudaFree(dev_data));
		CSC(cudaFree(dev_a));	
	}
};

int main() {
	
	IOHelper<uchar4> helper = IOHelper<uchar4>();

	CUGaussianBlur image = CUGaussianBlur(&helper);
	if(helper.GetRadius() != 0){
		
		image.run_kernel();
	}

	helper.WriteData(image.data);

	return 0;
}