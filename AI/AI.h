#include "nncpu.h"

#ifndef AI_h
#define AI_h

class geneticAI
{
private:
	int nnSize;
	int* layers;
	int numMutations;
	nncpu* network;
public:
	geneticAI(const int size, const int* inpLayers, const int mutations, const char* fileName, const double weightsRange);
	void saveWeights(const char* fileName);
	int study(double** data, int dataSize, double mutationPower, int (*testResults)(double*, int, double*, int*), int numRotates, double** answerData);
	~geneticAI();
};

#endif
