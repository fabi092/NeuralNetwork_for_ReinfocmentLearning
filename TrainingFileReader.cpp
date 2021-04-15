#include "TrainingFileReader.h"
#include <cassert>
#include <iosfwd>
#include <algorithm>
#include <iostream>

//-------------------------------------------------------------------------


namespace BPN
{
	TrainingFileReader::TrainingFileReader(std::string const& filename, int32_t numInputs, int32_t numOutputs)
		: m_filename(filename)
		, m_numInputs(numInputs)
		, m_numOutputs(numOutputs)
	{
		assert(!filename.empty() && m_numInputs > 0 && m_numOutputs > 0);
	}

	bool TrainingFileReader::ReadData()
	{
		assert(!m_filename.empty());

		std::fstream inputFile;
		inputFile.open(m_filename, std::ios::in);

		if (inputFile.is_open())
		{
			std::string line;

			// Read data
			//-------------------------------------------------------------------------

			int32_t const totalValuesToRead = m_numInputs + m_numOutputs;

			while (!inputFile.eof())
			{
				std::getline(inputFile, line);
				if (line.length() > 2)
				{
					m_entries.push_back(TrainingEntry());
					TrainingEntry& entry = m_entries.back();

					char* cstr = new char[line.size() + 1];
					strcpy_s(cstr, line.size() + 1, line.c_str());

					for (int i = 0; i < m_numInputs; i++)
					{
						std::string stringNumber = line.substr(0, line.find(","));
						bool isANumber = (stringNumber.find_first_not_of("0123456789.") == std::string::npos);
						if (isANumber) {
							entry.m_inputs.push_back(std::stof(stringNumber));
						}
						else
							return false;

						line.erase(0, line.find(",") + 1);
					}
					if (line._Equal("Iris-setosa"))
					{
					entry.m_expectedOutputs.push_back(1);
					entry.m_expectedOutputs.push_back(0);
					entry.m_expectedOutputs.push_back(0);
					}
					else if (line._Equal("Iris-versicolor"))

					//else if (pToken == "Iris-versicolor")
					{
						entry.m_expectedOutputs.push_back(0);
						entry.m_expectedOutputs.push_back(1);
						entry.m_expectedOutputs.push_back(0);
					}
					else if (line._Equal("Iris-virginica"))
					{
						entry.m_expectedOutputs.push_back(0);
						entry.m_expectedOutputs.push_back(0);
						entry.m_expectedOutputs.push_back(1);
					}
				}
			}

			inputFile.close();

			if (!m_entries.empty())
			{
				CreateTrainingData();
			}

			std::cout << "Input file: " << m_filename << "\nRead complete: " << m_entries.size() << " inputs loaded" << std::endl;
			return true;
		}
		else
		{
			std::cout << "Error Opening Input File: " << m_filename << std::endl;
			return false;
		}
	}

	void TrainingFileReader::CreateTrainingData()
	{
		assert(!m_entries.empty());

		std::random_shuffle(m_entries.begin(), m_entries.end());

		// Training set
		int32_t const numEntries = (int32_t)m_entries.size();
		int32_t const numTrainingEntries = (int32_t)(0.75 * numEntries);

		int32_t entryIdx = 0;
		for (; entryIdx < numTrainingEntries; entryIdx++)
		{
			m_data.m_trainingSet.push_back(m_entries[entryIdx]);
		}

			for (; entryIdx < numEntries; entryIdx++)
		{
			m_data.m_testSet.push_back(m_entries[entryIdx]);
		}
	}
}