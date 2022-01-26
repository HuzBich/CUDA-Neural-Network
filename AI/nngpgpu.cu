#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "device_launch_parameters.h"
#include <chrono>

#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <string>
#include "helpLib.h"
#include <math.h>
#include <iostream>

#include "nngpgpu.cuh"

#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>


uint64_t micros()
{
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::
        now().time_since_epoch()).count();
    return us;
}


__global__ void gpuFirstMultiply(double neuronsIn, double* weights, double* neuronsOut, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        neuronsOut[thread] = neuronsIn * weights[thread];
    }
}


__global__ void gpuMultiply(double neuronsIn, double* weights, double* neuronsOut, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        neuronsOut[thread] += neuronsIn * weights[thread];
    }
}


__global__ void gpuNormalizeData(double* neurons, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        neurons[thread] = pow(2.71828, neurons[thread]);
        neurons[thread] /= (neurons[thread] + 1);
    }
}


__global__ void calcTrueAnswer(double* trueAnswer, double* answers, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        trueAnswer[thread] = answers[thread] - trueAnswer[thread];
    }
}


__global__ void gpuTrueAnsFirstMultiply(double* neuronsIn, double* weights, double neuronsOut, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        neuronsIn[thread] = neuronsOut * weights[thread];
    }
}


__global__ void gpuTrueAnsMultiply(double* neuronsIn, double* weights, double neuronsOut, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        neuronsIn[thread] += neuronsOut * weights[thread];
    }
}


__global__ void calcWeights(double* weights, double* trueAnswers, double neuron, double* synapseNeurons, double learningRate, int length)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread < length)
    {
        weights[thread] += trueAnswers[thread] * neuron * synapseNeurons[thread] * (1 - synapseNeurons[thread]) * learningRate;
    }
}


gpgpu::gpgpu(const int size, const int* inpLayers)
{
    this->threadsPerBlock = 256;
    this->err = cudaSuccess;
    this->nnSize = size;
    this->layers = new int[size];
    for (int layer = 0; layer < this->nnSize; layer++)
    {
        this->layers[layer] = inpLayers[layer];
    }

    std::cout << "Create neurons\n";
    this->neurons = new double* [this->nnSize];
    for (int layer = 0; layer < this->nnSize; layer++)
    {
        double* localLayerNeurons = NULL;
        err = cudaMalloc((void**)&localLayerNeurons, (this->layers[layer]+1) * sizeof(double));
        checkErr();
        this->neurons[layer] = localLayerNeurons;
        
        localLayerNeurons = new double[this->layers[layer] + 1];
        for (int neuron = 0; neuron < this->layers[layer]; neuron++)
        {
            localLayerNeurons[neuron] = 0;
        }
        localLayerNeurons[this->layers[layer]] = 1;
        err = cudaMemcpy(this->neurons[layer], localLayerNeurons, (this->layers[layer] + 1) * sizeof(double), cudaMemcpyHostToDevice);
        checkErr();
        delete[] localLayerNeurons;
        
    }
    std::cout << "Done\n";

    std::cout << "Create weights\n";
    this->weights = new double** [this->nnSize-1];
    for (int layer = 0; layer < this->nnSize-1; layer++)
    {
        this->weights[layer] = new double* [this->layers[layer]+1];
        for (int neuron = 0; neuron < this->layers[layer]+1; neuron++)
        {
            double* localNeuronWeights = NULL;
            this->err = cudaMalloc((void**)&localNeuronWeights, this->layers[layer+1] * sizeof(double));
            checkErr();
            this->weights[layer][neuron] = localNeuronWeights;
        }
    }
    std::cout << "Done\n";
}


bool gpgpu::getWeights(const char* fileName, const double weightsRange)
{
    double*** localWeights = new double** [this->nnSize - 1];;
    std::ifstream file(fileName);
    bool rtrn = 0;
    if (file.is_open())
    {
        std::cout << "Get weights > ";
        int size;
        file >> size;
        if (size != this->nnSize)
        {
            std::cout << "False size of nn\n";
            exit(1);
        }
        for (int layer = 0; layer < this->nnSize; layer++)
        {
            file >> size;
            if (size != this->layers[layer])
            {
                std::cout << "False size of layer nn\n";
                exit(1);
            }
        }
        for (int layer = 0; layer < this->nnSize - 1; layer++)
        {
            localWeights[layer] = new double* [this->layers[layer] + 1];
            for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
            {
                localWeights[layer][neuron] = new double[this->layers[layer + 1]];
                for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
                {
                    file >> localWeights[layer][neuron][synapse];
                    //std::cout << localWeights[layer][neuron][synapse] << ' ';
                }
            }
            std::cout << '#';
        }
        std::cout << " done\n";
    }
    else
    {
        rtrn = 1;
        std::cout << "Create weights > ";
        srand((unsigned int)time(NULL));
        for (int layer = 0; layer < this->nnSize - 1; layer++)
        {
            localWeights[layer] = new double* [(long long)this->layers[layer] + 1];
            for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
            {
                localWeights[layer][neuron] = new double[this->layers[layer + 1]];
                for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
                {
                    localWeights[layer][neuron][synapse] = (double)rand() / (double)RAND_MAX * weightsRange * 2.0 - weightsRange;
                }
            }
            std::cout << '#';
        }
        std::cout << " done\n";
    }
    file.close();
    std::cout << "Copying data" << '\n';
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
        {
            this->err = cudaMemcpy(this->weights[layer][neuron], localWeights[layer][neuron], this->layers[layer + 1] * sizeof(double), cudaMemcpyHostToDevice);
            checkErr();
        }
    }
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
        {
            delete[] localWeights[layer][neuron];
        }
        delete[] localWeights[layer];
    }
    delete[] localWeights;
    std::cout << "Done\n";
    return rtrn;
}


void gpgpu::saveWeights(const char* fileName)
{
    std::cout << "Save weights > ";
    std::ofstream file(fileName);
    file << this->nnSize << ' ';
    for (int layer = 0; layer < this->nnSize; layer++)
    {
        file << this->layers[layer] << ' ';
    }
    file << '\n';
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        double* localWeights = new double[this->layers[layer + 1]];
        for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
        {
            this->err = cudaMemcpy(localWeights, this->weights[layer][neuron], this->layers[layer+1] * sizeof(double), cudaMemcpyDeviceToHost);
            checkErr();
            for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
            {
                file << localWeights[synapse] << ' ';
            }
            file << '\n';
        }
        delete[] localWeights;
        file << '\n';
        std::cout << '#';
    }
    file.close();
    std::cout << " done\n";
}


void gpgpu::setFirstLayer(double* firstLayer)
{
    /*for (int i = 0; i < layers[0] + 1; i++)
    {
        std::cout << firstLayer[i] << ' ';
    }*/
    //std::cout << '\n';
    this->err = cudaMemcpy(this->neurons[0], firstLayer, this->layers[0] * sizeof(double), cudaMemcpyHostToDevice);
    checkErr();
}


void gpgpu::neuralWork()
{
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        double* localNeurons = new double[this->layers[layer] + 1];
        this->err = cudaMemcpy(localNeurons, this->neurons[layer], (this->layers[layer]+1) * sizeof(double), cudaMemcpyDeviceToHost);
        checkErr();
        //std::cout << "\nneur " << layer << '\n';
        //for (int i = 0; i < layers[layer] + 1; i++)
        //    std::cout << localNeurons[i] << ' ';
        //std::cout << '\n';
        int blocksPerGrid = (this->layers[layer+1] + this->threadsPerBlock - 1) / this->threadsPerBlock;
        gpuFirstMultiply KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (localNeurons[0], this->weights[layer][0], this->neurons[layer + 1], this->layers[layer + 1]);
        this->err = cudaGetLastError();
        checkErr();
        for (int neuron = 1; neuron <= this->layers[layer]; neuron++)
        {
            gpuMultiply KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (localNeurons[neuron], this->weights[layer][neuron], this->neurons[layer+1], this->layers[layer+1]);
        }
        gpuNormalizeData KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (this->neurons[layer+1], this->layers[layer+1]);
        this->err = cudaGetLastError();
        checkErr();
        delete[] localNeurons;
    }
}


void gpgpu::getOutNeurons(double* outNeurons)
{
    this->err = cudaMemcpy(outNeurons, this->neurons[this->nnSize-1], this->layers[this->nnSize-1] * sizeof(double), cudaMemcpyDeviceToHost);
    checkErr();
}


// funk to study neural network
double gpgpu::correctWeights(double learningRate, double* answer)
{
    double** trueAnswers = new double* [this->nnSize];
    for (int layer = 1; layer < this->nnSize; layer++)
    {
        double* localTrueAns = NULL;
        this->err = cudaMalloc((void**)&localTrueAns, this->layers[layer] * sizeof(double));
        checkErr();
        trueAnswers[layer] = localTrueAns;
    }
    this->err = cudaMemcpy(trueAnswers[this->nnSize-1], this->neurons[this->nnSize-1], this->layers[this->nnSize-1] * sizeof(double), cudaMemcpyDeviceToDevice);
    checkErr();
    double* localTrueAnswers = NULL;
    this->err = cudaMalloc((void**)&localTrueAnswers, this->layers[this->nnSize - 1] * sizeof(double));
    checkErr();
    this->err = cudaMemcpy(localTrueAnswers, answer, this->layers[this->nnSize - 1] * sizeof(double), cudaMemcpyHostToDevice);
    int blocksPerGrid = (this->layers[this->nnSize - 1] + this->threadsPerBlock - 1) / this->threadsPerBlock;
    calcTrueAnswer KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (trueAnswers[this->nnSize - 1], localTrueAnswers, this->layers[this->nnSize - 1]);
    this->err = cudaGetLastError();
    checkErr();
    cudaFree(localTrueAnswers);
    checkErr();
    
    for (int layer = this->nnSize - 2; layer > 0; layer--)
    {
        double* localAnswer = new double[this->layers[layer + 1]];
        this->err = cudaMemcpy(localAnswer, trueAnswers[layer + 1], this->layers[layer + 1] * sizeof(double), cudaMemcpyDeviceToHost);
        checkErr();

        double** localWeights = new double* [this->layers[layer]];
        for (int localNeuron = 0; localNeuron < this->layers[layer]; localNeuron++)
        {
            localWeights[localNeuron] = new double[this->layers[layer + 1]];
            this->err = cudaMemcpy(localWeights[localNeuron], this->weights[layer][localNeuron], this->layers[layer + 1] * sizeof(double), cudaMemcpyDeviceToHost);
        }
        checkErr();
        double** newLocalWeights = new double* [this->layers[layer + 1]];
        for (int localSynapse = 0; localSynapse < this->layers[layer + 1]; localSynapse++)
        {
            newLocalWeights[localSynapse] = new double[this->layers[layer] + 1];
            for (int localNeuron = 0; localNeuron < this->layers[layer]; localNeuron++)
            {
                newLocalWeights[localSynapse][localNeuron] = localWeights[localNeuron][localSynapse];
            }
        }
        for (int localNeuron = 0; localNeuron < this->layers[layer]; localNeuron++)
        {
            delete[] localWeights[localNeuron];
        }
        delete[] localWeights;
        
        double* inputWeights = NULL;
        this->err = cudaMalloc((void**)&inputWeights, this->layers[layer] * sizeof(double));
        checkErr();

        this->err = cudaMemcpy(inputWeights, newLocalWeights[0], (this->layers[layer]) * sizeof(double), cudaMemcpyHostToDevice);
        checkErr();

        blocksPerGrid = (this->layers[layer] + this->threadsPerBlock - 1) / this->threadsPerBlock;
        gpuTrueAnsFirstMultiply KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (trueAnswers[layer], inputWeights, localAnswer[0], this->layers[layer]);
        this->err = cudaGetLastError();
        checkErr();
        for (int synapse = 1; synapse < this->layers[layer + 1]; synapse++)
        {
            this->err = cudaMemcpy(inputWeights, newLocalWeights[synapse], (this->layers[layer] ) * sizeof(double), cudaMemcpyHostToDevice);
            gpuTrueAnsMultiply KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (trueAnswers[layer], inputWeights, localAnswer[synapse], this->layers[layer]);
        }
        this->err = cudaGetLastError();
        checkErr();
        for (int localSynapse = 0; localSynapse < this->layers[layer + 1]; localSynapse++)
        {
            delete[] newLocalWeights[localSynapse];
        }
        delete[] newLocalWeights;
        delete[] localAnswer;
        this->err = cudaFree(inputWeights);
        checkErr();
    }

    double totalErr = 0;
    for (int layer = 1; layer < this->nnSize; layer++)
    {
        double* errors = new double[this->layers[layer]];
        this->err = cudaMemcpy(errors, trueAnswers[layer], this->layers[layer] * sizeof(double), cudaMemcpyDeviceToHost);
        checkErr();
        for (int neuron = 0; neuron < this->layers[layer]; neuron++)
        {
            totalErr += errors[neuron] * errors[neuron];
        }
        delete[] errors;
    }
  
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        double* localNeurons = new double[this->layers[layer]+1];
        this->err = cudaMemcpy(localNeurons, this->neurons[layer], (this->layers[layer]+1) * sizeof(double), cudaMemcpyDeviceToHost);
        checkErr();
        blocksPerGrid = (this->layers[layer + 1] + this->threadsPerBlock - 1) / this->threadsPerBlock;
        for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
        {
            calcWeights KERNEL_ARGS2(blocksPerGrid, this->threadsPerBlock) (this->weights[layer][neuron], trueAnswers[layer + 1], localNeurons[neuron],
                this->neurons[layer + 1], learningRate, this->layers[layer + 1]);
        }
        this->err = cudaGetLastError();
        checkErr();
        delete[] localNeurons;
    }

    for (int layer = 1; layer < this->nnSize; layer++)
    {
        this->err = cudaFree(trueAnswers[layer]);
        checkErr();
    }
    delete[] trueAnswers;
    return totalErr;
}


void gpgpu::checkErr()
{
    if (this->err != cudaSuccess)
    {
        fprintf(stderr, "Error ", cudaGetErrorString(this->err));
        std::cout << cudaGetErrorString(this->err) << '\n';
        exit(EXIT_FAILURE);
    }
}

gpgpu::~gpgpu()
{
    std::cout << "Clearing data\n";
    for (int layer = 0; layer < this->nnSize; layer++)
    {
        this->err = cudaFree(this->neurons[layer]);
        checkErr();
    }
    delete[] this->neurons;
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        for (int neuron = 0; neuron < this->layers[layer] + 1; neuron++)
        {
            this->err = cudaFree(this->weights[layer][neuron]);
            checkErr();
        }
        delete[] this->weights[layer];
    }
    delete[] this->weights;
    delete[] this->layers;
    std::cout << "Done\n";
}