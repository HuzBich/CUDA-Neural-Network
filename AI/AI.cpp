#include "AI.h"
#include "nncpu.h"
#include <iostream>

geneticAI::geneticAI(const int size, const int* inpLayers, const int mutations, const char* fileName, const double weightsRange)
{
	this->nnSize = size;
	this->numMutations = mutations;
	this->layers = new int[size];
	for (int layer = 0; layer < size; layer++)
	{
		this->layers[layer] = inpLayers[layer];
	}
	this->network = new nncpu(size, inpLayers);
	this->network->getWeights(fileName, weightsRange);
	this->network->createGeneration(mutations);
}

int geneticAI::study(double** data, int dataSize, double mutationPower, int (*testResults)(double*, int, double*, int*), int numRotates, double** answerData)
{
	std::cout << "Work\n";
	this->network->resetMutation();
	this->network->mutationWork(mutationPower);
	double* output = new double[this->layers[this->nnSize - 1]];
	int* results = new int[this->numMutations];
	for (int mutation = 0; mutation < this->numMutations; mutation++)
	{
		results[mutation] = 0;
		for (int test = 0; test < dataSize; test++)
		{
			this->network->setFirstLayer(data[test]);
			this->network->mutationNeuralWork(mutation);
			this->network->getOutNeurons(output);
			int shift = 0;
			results[mutation] += testResults(output, numRotates, answerData[test], &shift);
			test += shift;
		}
	}
	int bestMutation = 0;
	for (int mutation = 1; mutation < this->numMutations; mutation++)
	{
		if (results[mutation] >= results[bestMutation])
		{
			bestMutation = mutation;
		}
	}
	int best = results[bestMutation];
	std::cout << "Best result: " << best << '\n';
	this->network->chooseBestMutation(bestMutation);
	delete[] output;
	delete[] results;
	return best;
}

void geneticAI::saveWeights(const char* fileName)
{
	network->saveWeights(fileName);
}

geneticAI::~geneticAI()
{
	delete[] layers;
	network->clearMutationMem();
	delete network;
}