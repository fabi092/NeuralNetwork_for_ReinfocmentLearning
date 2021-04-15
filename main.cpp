#include "NNTrainer.h"
#include "TrainingFileReader.h"
#include <iostream>

using namespace std;

int main()
{
	std::string trainingDataPath = "iris_original.data";

	uint32_t const numInputs = 4;
	uint32_t const numHidden = 3;
	uint32_t const numOutputs = 3;


	BPN::TrainingFileReader dataReader(trainingDataPath, numInputs, numOutputs);
	if (!dataReader.ReadData())
	{
		return 1;
	}

	// Create neural network
	BPN::Network::Settings networkSettings{ numInputs, numHidden, numOutputs };
	BPN::Network nn(networkSettings);

	// Create neural network trainer
	BPN::NNTrainer::Settings trainerSettings;
	trainerSettings.m_learningRate = 0.001;
	trainerSettings.m_momentum = 0.9;
	trainerSettings.m_maxGenerations = 1000;
	trainerSettings.m_desiredAccuracy = 85;


	BPN::NNTrainer trainer(trainerSettings, &nn);

	bool programEnd = 0;
	string input;

	while (!programEnd) {

		cout << endl << "Filepath: " << trainingDataPath << ", desiredAccuracy:" << trainerSettings.m_desiredAccuracy << ", maxGenerations:" << trainerSettings.m_maxGenerations << ", momentum:"
			<< trainerSettings.m_momentum << ", LearnRate:" << trainerSettings.m_learningRate << endl << " Enter a command for IrisNN: " << endl;
		cin >> input;
		cin.clear();
		cout << endl;
		string command = input.substr(0, input.find(' '));

		if (!getline(cin, input))
		{
			cout << "Critical error occurred while reading the inputstream.";
			return 0;
		}

		// Start of "switch cases" that will read the different inputs.
		if (command == "end")
		{
			programEnd = 1;
		}
		else if (command == "train")
		{
			nn = BPN::Network (networkSettings);
			trainer = BPN::NNTrainer(trainerSettings, &nn);
			trainer.Train(dataReader.GetTrainingData());
		}
		else if (command == "check")
		{
			vector<double> test;
			for (int i = 0; i < numInputs; i++)
			{
				input.erase(0, input.find(' ') + 1);
				std::string stringNumber = input.substr(0, input.find(' '));
				bool isANumber = (stringNumber.find_first_not_of("0123456789.") == std::string::npos);
				if (isANumber) {
					test.push_back(std::stof(stringNumber));
				}
				else
					break;
			}
			if (test.size() == numInputs)
				cout << nn.Evaluate(test) << endl;
		}
		else if (command == "accuracy")
		{
			// read second part of input
			input.erase(0, input.find(' ') + 1);
			string stringNumber = input;
			bool has_only_digits = (stringNumber.find_first_not_of("0123456789") == string::npos);

			if (has_only_digits) {
				trainerSettings.m_desiredAccuracy = stoi(input.substr(0, input.find(' ')));

			}
		}
			else if (command == "generations")
			{
				// read second part of input
				input.erase(0, input.find(' ') + 1);
				string stringNumber = input;
				bool has_only_digits = (stringNumber.find_first_not_of("0123456789") == string::npos);

				if (has_only_digits) {
					trainerSettings.m_maxGenerations = stoi(input.substr(0, input.find(' ')));

				}
			}
			else if (command == "learnrate")
			{
				// read second part of input
				input.erase(0, input.find(' ') + 1);
				string stringNumber = input;
				bool has_only_digits = (stringNumber.find_first_not_of("0123456789.") == string::npos);

				if (has_only_digits) {
					trainerSettings.m_learningRate = stof(input.substr(0, input.find(' ')));

				}
			}
			else if (command == "momentum")
			{
				// read second part of input
				input.erase(0, input.find(' ') + 1);
				string stringNumber = input;
				bool has_only_digits = (stringNumber.find_first_not_of("0123456789.") == string::npos);

				if (has_only_digits) {
					trainerSettings.m_momentum = stof(input.substr(0, input.find(' ')));

				}
			}

			else if (command == "filepath")
			{
				// read second part of input
				input.erase(0, input.find(' ') + 1);
				trainingDataPath = input;
				BPN::TrainingFileReader dataReader(trainingDataPath, numInputs, numOutputs);
				if (!dataReader.ReadData())
				{
					return 1;
				}
			}
			else
			{
				cout << "Invalid Command! The following commands are available:" << endl <<
					"train, check (double) (double) (double) (double), accuracy (integer), generations (integer)," << endl <<
					" learnrate (double), momentum (double), filepath (string) end" << endl;
			}
		}
		return 0;
}
