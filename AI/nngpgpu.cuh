#ifndef nngpgpu_h
#define nngpgpu_h
#include <cuda_runtime.h>
#include <chrono>

uint64_t micros();

class gpgpu
{
private:
	int threadsPerBlock;
	cudaError_t err;
	double** neurons;
	double*** weights;
	int nnSize;
	int* layers;
	void checkErr();
public:
	gpgpu(const int size, const int* inpLayers);
	bool getWeights(const char* fileName, const double weightsRange);
	void saveWeights(const char* fileName);
	void setFirstLayer(double* firstLayer);
	void neuralWork();
	void getOutNeurons(double* outNeurons);
	double correctWeights(double learningRate, double* answer);
	~gpgpu();
};

#endif