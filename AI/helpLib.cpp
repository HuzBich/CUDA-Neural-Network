#include <string>

// funk which return index first elem in string
int findChar(std::string string, int index, char elem)
{
	while (string[index] != '/0')
	{
		if (string[index] == elem)
		{
			return index;
		}
		index++;
	}
	return -1;
}

double strToDouble(std::string string)
{
	double value = 0;
	int index = 0;
	bool minus = false;
	while (string[index] != '.' && string[index] != '\0')
	{
		if (string[index] == '-')
		{
			minus = true;
		}
		else
		{
			value *= 10;
			value += (double)string[index] - 48;
		}
		index++;
	}
	int discharge = 0;
	if (string[index] == '.')
	{
		index++;  // skip point
	}
	while (string[index] != '\0')
	{
		double val = (double)string[index] - 48;
		for (int disch = 0; disch <= discharge; disch++)
		{
			val /= 10.0;
		}
		value += val;
		discharge++;
		index++;
	}
	if (minus)
	{
		value = -value;
	}
	return value;
}

std::string getSubstring(std::string string, int startIndex, int endIndex)
{
	char result[32];
	for (int firstIndex = 0, secondIndex = startIndex; secondIndex < endIndex; firstIndex++, secondIndex++)
	{
		result[firstIndex] = string[secondIndex];
	}
	result[endIndex - startIndex] = '\0';
	std::string strResult = result;
	return strResult;
}