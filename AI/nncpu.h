#ifndef nncpu_h
#define nncpu_h

class nncpu
{
private:
	double activation(double neuron);
	int nnSize = 0;
	int* layers = 0;
	double** neurons = 0;
	double*** weights = 0;
	int mutationSize = 0;
	double**** genreWeights = 0;
public:
	nncpu(const int size, const int* inpLayers);
	bool getWeights(const char* fileName, const double weightsRange);
	void saveWeights(const char* fileName);
	void setFirstLayer(double* firstLayer);
	void neuralWork();
	void getOutNeurons(double* outNeurons);
	void createGeneration(int numMutation);
	void resetMutation();
	void mutationWork(double power);
	void mutationNeuralWork(int numMutation);
	void chooseBestMutation(int numBest);
	void clearMutationMem();
	~nncpu();
};

#endif