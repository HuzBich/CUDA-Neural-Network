#include "AI.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#define TEST 0
#define SAVE 0

void getData(std::vector<int>* data, const char* fileName)
{
	std::cout << "Get train data\n";
	std::ifstream file(fileName);
	while (true)
	{
		std::string s;
		int val;
		file >> s >> val;
		if (val == -1)
		{
			break;
		}
		if (val >= 1 && val <= 7) val = 1;
		else if (val >= 8 && val <= 14) val = 2;
		data->push_back(val);
	}
	file.close();
	std::cout << "Done: " << data->size() << '\n';
}

void convertData(std::vector<int>* inputData, double** trainData, double** answerData, int numTrain, int numAnswer)
{
	std::cout << "Converting data\n";
	for (int train = 0; train < inputData->size() - numTrain - numAnswer + 1; train++)
	{
		trainData[train] = new double[numTrain];
		answerData[train] = new double[numAnswer];
		for (int data = 0; data < numTrain; data++)
		{
			if ((*inputData)[train + data] == 1)
				trainData[train][data] = 0;
			else if ((*inputData)[train + data] == 2)
				trainData[train][data] = 1;
			else
				trainData[train][data] = 0.5;
		}
		for (int data = 0; data < numAnswer; data++)
		{
			if ((*inputData)[train + numTrain + data] == 1)
				answerData[train][data] = 0;
			else if ((*inputData)[train + numTrain + data] == 2)
				answerData[train][data] = 1;
			else 
				answerData[train][data] = 0.5;
		}
	}
	std::cout << "Done\n";
}

int testResults(double* result, int numRotates, double* answers, int* shift)
{
#if TEST == 0
	int res = 0;
	int stat = 1;
	int bots = 8;
	int bet = 1;
	
	for (int color = 0; color < numRotates; color++)
	{
		if (color < 2)
		{
			if (answers[color] == 0.5)
			{
				res += bet * 12;
			}
			else if (result[color] > 0.5 && answers[color] > 0.5)
			{
				res += 0;
			}
			else if (result[color] < 0.5 && answers[color] < 0.5)
			{
				res += 0;
			}
			else
			{
				res -= bet * 2;
				*shift = color;
				stat = 0;
				break;
			}
		}
		else if (bots > 1)
		{
			if (answers[color] == 0.5)
			{
				res += bet * 12 * bots;
			}
			else
			{
				res -= bet * 1 * bots;
				bots /= 2;
			}
		}
		else
		{
			if (answers[color] == 0.5)
			{
				res += bet * 12;
			}
			else if (result[color] > 0.5 && answers[color] > 0.5)
			{
				res += 0;
			}
			else if (result[color] < 0.5 && answers[color] < 0.5)
			{
				res += 0;
			}
			else
			{
				res -= bet * 2;
				*shift = color;
				stat = 0;
				break;
			}
		}
		bet++;
	}
	if (stat)
	{
		res += 1000;
		*shift = 9;
	}
	return res;
#else
	int res = 0;

	for (int color = 0; color < numRotates; color++)
	{
		
		if (result[color] > 0.4 && result[color] < 0.6)
		{
			if (answers[color] == 0.5)
			{
				res += 13;
			}
			else
				res -= 1;
			//std::cout << "GREEN!!\n";
			return res;
			
		}
		else if (result[color] > 0.5 && answers[color] > 0.5)
		{
			res += 1;
			return res;
		}
		else if (result[color] < 0.5 && answers[color] < 0.5)
		{
			res += 1;
			return res;
		}
		else
		{
			//std::cout << color << '\n';
			res -= 1;
			*shift = color;
			return res;
		}
		

	}
	//if (stat)res += 1000;
	*shift = 9;
	return res;
#endif
}

int main()
{
	const int numRotates = 10;
	const int nnSize = 3;
	const int numMutations = 100;
	const double weightsRange = 0.1;
	const double mutationPower = 0;
	const int layers[nnSize] = { numRotates, 10, 10 };
	const char* dataFileName = "datashit2.txt";
	const char* weightsFileName = "weights.txt";
	const char* bestWeightsFileName = "bestWeights.txt";
	std::cout << "Preparing data\n";
	std::vector<int>inputData;
	getData(&inputData, dataFileName);
	int trainSize = inputData.size() - layers[0] - layers[nnSize - 1] + 1;
	double** trainData = new double* [trainSize];
	double** answerData = new double* [trainSize];
	convertData(&inputData, trainData, answerData, numRotates, layers[nnSize-1]);

	geneticAI AI(nnSize, layers, numMutations, weightsFileName, weightsRange);

	int result;
	int bestResult = 0;
	while (true)
	{
		
		result = AI.study(trainData, trainSize, mutationPower, testResults, numRotates, answerData);
	
#if SAVE == 1
		if (result > bestResult)
		{
			bestResult = result;
			AI.saveWeights(bestWeightsFileName);
		}
		AI.saveWeights(weightsFileName);
#endif
	}
	

	for (int train = 0; train < trainSize; train++)
	{
		delete[] trainData[train];
		delete[] answerData[train];
	}
	delete[] trainData;
	delete[] answerData;
}