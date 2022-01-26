#include "nncpu.h"
#include<iostream>
#include<fstream>
#include<time.h>


double nncpu::activation(double neuron)
{
    neuron = pow(2.71828, neuron);
    neuron /= (neuron + 1);
    return neuron;
}


nncpu::nncpu(const int size, const int* inpLayers)
{
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
        this->neurons[layer] = new double[this->layers[layer] + 1];
        for (int neuron = 0; neuron < this->layers[layer]; neuron++)
        {
            this->neurons[layer][neuron] = 0;
        }
        this->neurons[layer][this->layers[layer]] = 1;
    }
    std::cout << "Done\n";

    std::cout << "Create weights\n";
    this->weights = new double** [this->nnSize - 1];
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        this->weights[layer] = new double* [this->layers[layer] + 1];
        for (int neuron = 0; neuron < this->layers[layer] + 1; neuron++)
        {
            this->weights[layer][neuron] = new double[this->layers[layer+1]];
        }
    }
    std::cout << "Done\n";
}


bool nncpu::getWeights(const char* fileName, const double weightsRange)
{
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
            for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
            {
                for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
                {
                    file >> this->weights[layer][neuron][synapse];
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
            for (int neuron = 0; neuron <= this->layers[layer]; neuron++)
            {
                for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
                {
                    this->weights[layer][neuron][synapse] = (double)rand() / (double)RAND_MAX * weightsRange * 2.0 - weightsRange;
                }
            }
            std::cout << '#';
        }
        std::cout << " done\n";
    }
    file.close();
    return rtrn;
}


void nncpu::saveWeights(const char* fileName)
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
        for (int neuron = 0; neuron < this->layers[layer] + 1; neuron++)
        {
            for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
            {
                file << this->weights[layer][neuron][synapse] << ' ';
            }
            file << '\n';
        }
        file << '\n';
        std::cout << '#';
    }
    file.close();
    std::cout << " done\n";
}


void nncpu::setFirstLayer(double* firstLayer)
{
    for (int neuron = 0; neuron < layers[0]; neuron++)
    {
        this->neurons[0][neuron] = firstLayer[neuron];
    }
}


void nncpu::neuralWork()
{
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
        {
            this->neurons[layer + 1][synapse] = this->neurons[layer][0] * this->weights[layer][0][synapse];
        }
        for (int neuron = 1; neuron < this->layers[layer] + 1; neuron++)
        {
            for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
            {
                this->neurons[layer + 1][synapse] += this->neurons[layer][neuron] * this->weights[layer][neuron][synapse];
            }
        }
        for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
        {
            this->neurons[layer + 1][synapse] = activation(neurons[layer + 1][synapse]);
        }
    }
}


void nncpu::getOutNeurons(double* outNeurons)
{
    for (int neuron = 0; neuron < this->layers[this->nnSize-1]; neuron++)
    {
        outNeurons[neuron] = this->neurons[this->nnSize-1][neuron];
    }
}


void nncpu::createGeneration(int numMutation)
{
    std::cout << "createGeneration\n";
    this->mutationSize = numMutation;
    this->genreWeights = new double*** [numMutation];
    for (int mutation = 0; mutation < numMutation; mutation++)
    {
        this->genreWeights[mutation] = new double** [nnSize];
        for (int layer = 0; layer < nnSize - 1; layer++)
        {
            this->genreWeights[mutation][layer] = new double* [layers[layer] + 1];
            for (int neuron = 0; neuron < layers[layer] + 1; neuron++)
            {
                this->genreWeights[mutation][layer][neuron] = new double[layers[layer + 1]];
            }
        }
    }
}


void nncpu::resetMutation()
{
    std::cout << "resetMutations\n";
    for (int mutation = 0; mutation < this->mutationSize; mutation++)
    {
        for (int layer = 0; layer < nnSize - 1; layer++)
        {
            for (int neuron = 0; neuron < layers[layer] + 1; neuron++)
            {
                for (int synapse = 0; synapse < layers[layer + 1]; synapse++)
                {
                    this->genreWeights[mutation][layer][neuron][synapse] = this->weights[layer][neuron][synapse];
                }
            }
        }
    }
}


void nncpu::mutationWork(double power)
{
    for (int mutation = 0; mutation < this->mutationSize; mutation++)
    {
        for (int layer = 0; layer < nnSize - 1; layer++)
        {
            for (int neuron = 0; neuron < layers[layer] + 1; neuron++)
            {
                for (int synapse = 0; synapse < layers[layer + 1]; synapse++)
                {
                    this->genreWeights[mutation][layer][neuron][synapse] += ((double)rand() / (double)RAND_MAX * power * 2.0 - power) * (2500.0 - this->genreWeights[mutation][layer][neuron][synapse]* this->genreWeights[mutation][layer][neuron][synapse]) / 2500.0;
                }
            }
        }
    }
}


nncpu::~nncpu()
{
    std::cout << "Clearing data\n";
    for (int layer = 0; layer < this->nnSize; layer++)
    {
        delete[] this->neurons[layer];
    }
    delete[] this->neurons;
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        for (int neuron = 0; neuron < this->layers[layer] + 1; neuron++)
        {
            delete[] weights[layer][neuron];
        }
        delete[] this->weights[layer];
    }
    delete[] this->weights;
    delete[] this->layers;
    std::cout << "Done\n";
}

void nncpu::mutationNeuralWork(int numMutation)
{
    for (int layer = 0; layer < this->nnSize - 1; layer++)
    {
        for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
        {
            // std::cout << layer << ' ' << synapse << '\n';
            this->neurons[layer + 1][synapse] = this->neurons[layer][0] * this->genreWeights[numMutation][layer][0][synapse];
        }
        for (int neuron = 1; neuron < this->layers[layer] + 1; neuron++)
        {
            for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
            {
                this->neurons[layer + 1][synapse] += this->neurons[layer][neuron] * this->genreWeights[numMutation][layer][neuron][synapse];
            }
        }
        for (int synapse = 0; synapse < this->layers[layer + 1]; synapse++)
        {
            this->neurons[layer + 1][synapse] = activation(neurons[layer + 1][synapse]);
        }
    }
}

void nncpu::chooseBestMutation(int numBest)
{
    for (int layer = 0; layer < nnSize - 1; layer++)
    {
        for (int neuron = 0; neuron < layers[layer] + 1; neuron++)
        {
            for (int synapse = 0; synapse < layers[layer + 1]; synapse++)
            {
                this->weights[layer][neuron][synapse] = genreWeights[numBest][layer][neuron][synapse];
            }
        }
    }
}

void nncpu::clearMutationMem()
{
    for (int mutation = 0; mutation < this->mutationSize; mutation++)
    {
        for (int layer = 0; layer < nnSize - 1; layer++)
        {
            for (int neuron = 0; neuron < layers[layer] + 1; neuron++)
            {
                delete[] this->genreWeights[mutation][layer][neuron];
            }
            delete[] this->genreWeights[mutation][layer];
        }
        delete[] this->genreWeights[mutation];
    }
    delete[] this->genreWeights;
}