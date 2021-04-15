#include "NNTrainer.h"
#include <iostream>
#include <cassert>

//-------------------------------------------------------------------------

namespace BPN
{
    NNTrainer::NNTrainer( Settings const& settings, Network* networkToTrain )
        : m_networkToTrain( networkToTrain )
        , m_learningRate( settings.m_learningRate )
        , m_momentum( settings.m_momentum )
        , m_desiredAccuracy( settings.m_desiredAccuracy )
        , m_maxGenerations( settings.m_maxGenerations )
        , m_currentGeneration( 0 )
        , m_trainingSetAccuracy( 0 )
        , m_testSetAccuracy( 0 )
        , m_trainingSetMSE( 0 )
        , m_testSetMSE( 0 )
    {
        assert( networkToTrain != nullptr );

        m_deltaInputHidden.resize( networkToTrain->m_weightsInputHidden.size() );
        m_deltaHiddenOutput.resize( networkToTrain->m_weightsHiddenOutput.size() );
        m_errorGradientsHidden.resize( networkToTrain->m_hiddenNeurons.size() );
        m_errorGradientsOutput.resize( networkToTrain->m_outputNeurons.size() );

        memset( m_deltaInputHidden.data(), 0, sizeof( double ) * m_deltaInputHidden.size() );
        memset( m_deltaHiddenOutput.data(), 0, sizeof( double ) * m_deltaHiddenOutput.size() );
        memset( m_errorGradientsHidden.data(), 0, sizeof( double ) * m_errorGradientsHidden.size() );
        memset( m_errorGradientsOutput.data(), 0, sizeof( double ) * m_errorGradientsOutput.size() );
		
    	if (!logFile.is_open())
		{
			logFile.open("IrisNNtrainingResult.csv", std::ios::out);

			if (logFile.is_open())
			{
				//write log file header
				logFile << "Generation,Training Set Accuracy, Training Set MSE, Test Set Accuracy, Test Set MSE" << std::endl;
			}
		}
    }

    void NNTrainer::Train( TrainingData const& trainingData )
    {
        // Reset training state
        m_currentGeneration = 0;
        m_trainingSetAccuracy = 0;
        m_testSetAccuracy = 0;
        m_trainingSetMSE = 0;
        m_testSetMSE = 0;

        // Print header
        //-------------------------------------------------------------------------

		std::cout << std::endl << " Neural Network Starting: " << std::endl;

        // Train network using training dataset for training and test dataset for testing

        while ( ( m_trainingSetAccuracy < m_desiredAccuracy || m_testSetAccuracy < m_desiredAccuracy ) && m_currentGeneration < m_maxGenerations )
        {
            // Use training set to train network
            RunGeneration( trainingData.m_trainingSet );

            // Get test set accuracy and MSE
            GetSetAccuracyAndMSE( trainingData.m_testSet, m_testSetAccuracy, m_testSetMSE );

			if (logFile.is_open())
			{
				logFile << m_currentGeneration << "," << m_trainingSetAccuracy << "," << m_trainingSetMSE << "," << m_testSetAccuracy << "," << m_testSetMSE << std::endl;
			}
            std::cout << "Generation: " << m_currentGeneration;
            std::cout << " Training Accuracy:" << m_trainingSetAccuracy << "%, MSE: " << m_trainingSetMSE;
            std::cout << " Test Accuracy:" << m_testSetAccuracy << "%, MSE: " << m_testSetMSE << std::endl;

            m_currentGeneration++;
		}
		logFile.close();
    }

    double NNTrainer::GetHiddenErrorGradient( int32_t hiddenIdx ) const
    {
        // Get sum of hidden->output weights * output error gradients
        double weightedSum = 0;
        for ( auto outputIdx = 0; outputIdx < m_networkToTrain->m_numOutputs; outputIdx++ )
        {
            int32_t const weightIdx = m_networkToTrain->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
            weightedSum += m_networkToTrain->m_weightsHiddenOutput[weightIdx] * m_errorGradientsOutput[outputIdx];
        }
        
        // Return error gradient
        return m_networkToTrain->m_hiddenNeurons[hiddenIdx] * ( 1.0 - m_networkToTrain->m_hiddenNeurons[hiddenIdx] ) * weightedSum;
    }

    void NNTrainer::RunGeneration( TrainingSet const& trainingSet )
    {
        double incorrectEntries = 0;
        double MSE = 0;

        for ( auto const& trainingEntry : trainingSet )
        {
            // Feed inputs through network and back propagate errors
            m_networkToTrain->Evaluate( trainingEntry.m_inputs );
            Backpropagate( trainingEntry.m_expectedOutputs );

            // Check all outputs from neural network against desired values
            bool resultCorrect = true;
            for ( int outputIdx = 0; outputIdx < m_networkToTrain->m_numOutputs; outputIdx++ )
            {
                if ( m_networkToTrain->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx] )
                {
                    resultCorrect = false;
                }

                // Calculate MSE
                MSE += pow( ( m_networkToTrain->m_outputNeurons[outputIdx] - trainingEntry.m_expectedOutputs[outputIdx] ), 2);
            }

            if ( !resultCorrect )
            {
                incorrectEntries++;
            }
        }

        // Update training accuracy and MSE
        m_trainingSetAccuracy = 100.0 - ( incorrectEntries / trainingSet.size() * 100.0 );
        m_trainingSetMSE = MSE / ( m_networkToTrain->m_numOutputs * trainingSet.size() );
    }

    void NNTrainer::Backpropagate( std::vector<int32_t> const& expectedOutputs )
    {
        // Modify deltas between hidden and output layers
        //--------------------------------------------------------------------------------------------------------
        for ( auto OutputIdx = 0; OutputIdx < m_networkToTrain->m_numOutputs; OutputIdx++ )
        {
            // Get error gradient for every output node
            m_errorGradientsOutput[OutputIdx] = GetOutputErrorGradient
        	( static_cast<double>(expectedOutputs[OutputIdx]), m_networkToTrain->m_outputNeurons[OutputIdx] );

            // For all nodes in hidden layer and bias neuron
            for ( auto hiddenIdx = 0; hiddenIdx <= m_networkToTrain->m_numHidden; hiddenIdx++ )
            {
                int32_t const weightIdx = m_networkToTrain->GetHiddenOutputWeightIndex( hiddenIdx, OutputIdx );

                // Calculate change in weight

                    m_deltaHiddenOutput[weightIdx]
                	= m_learningRate * m_networkToTrain->m_hiddenNeurons[hiddenIdx] * m_errorGradientsOutput[OutputIdx]
                	+ m_momentum 
                	* m_deltaHiddenOutput[weightIdx];
                
            }
        }

        // Modify deltas between input and hidden layers
        //--------------------------------------------------------------------------------------------------------

        for ( auto hiddenIdx = 0; hiddenIdx <= m_networkToTrain->m_numHidden; hiddenIdx++ )
        {
            // Get error gradient for every hidden node
            m_errorGradientsHidden[hiddenIdx] = GetHiddenErrorGradient( hiddenIdx );

            // For all nodes in input layer and bias neuron
            for ( auto inputIdx = 0; inputIdx <= m_networkToTrain->m_numInputs; inputIdx++ )
            {
                int32_t const weightIdx = m_networkToTrain->GetInputHiddenWeightIndex( inputIdx, hiddenIdx );

                // Calculate change in weight 

                    m_deltaInputHidden[weightIdx] = m_learningRate * m_networkToTrain->m_inputNeurons[inputIdx] * m_errorGradientsHidden[hiddenIdx] + m_momentum * m_deltaInputHidden[weightIdx];
                
            }
        }
            UpdateWeights();
    }

    void NNTrainer::UpdateWeights()
    {
        // Input -> hidden weights
        //--------------------------------------------------------------------------------------------------------

        for ( auto InputIdx = 0; InputIdx <= m_networkToTrain->m_numInputs; InputIdx++ )
        {
            for ( auto hiddenIdx = 0; hiddenIdx <= m_networkToTrain->m_numHidden; hiddenIdx++ )
            {
                int32_t const weightIdx = m_networkToTrain->GetInputHiddenWeightIndex( InputIdx, hiddenIdx );
                m_networkToTrain->m_weightsInputHidden[weightIdx] += m_deltaInputHidden[weightIdx];
            }
        }

        // Hidden -> output weights
        //--------------------------------------------------------------------------------------------------------

        for ( auto hiddenIdx = 0; hiddenIdx <= m_networkToTrain->m_numHidden; hiddenIdx++ )
        {
            for ( auto outputIdx = 0; outputIdx < m_networkToTrain->m_numOutputs; outputIdx++ )
            {
                int32_t const weightIdx = m_networkToTrain->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
                m_networkToTrain->m_weightsHiddenOutput[weightIdx] += m_deltaHiddenOutput[weightIdx];
            }
        }
    }

    void NNTrainer::GetSetAccuracyAndMSE( TrainingSet const& trainingSet, double& accuracy, double& MSE ) const
    {
        accuracy = 0;
        MSE = 0;

        double numIncorrectResults = 0;
        for ( auto const& trainingEntry : trainingSet )
        {
            m_networkToTrain->Evaluate( trainingEntry.m_inputs );

            // Check if the network outputs match the expected outputs
            bool correctResult = true;
            for ( int32_t outputIdx = 0; outputIdx < m_networkToTrain->m_numOutputs; outputIdx++ )
            {
                if ( static_cast<double>(m_networkToTrain->m_clampedOutputs[outputIdx]) != trainingEntry.m_expectedOutputs[outputIdx] )
                {
                    correctResult = false;
                }

                MSE += pow( ( m_networkToTrain->m_outputNeurons[outputIdx] - trainingEntry.m_expectedOutputs[outputIdx] ), 2 );
            }

            if ( !correctResult )
            {
                numIncorrectResults++;
            }
        }

        accuracy = 100.0f - ( numIncorrectResults / trainingSet.size() * 100.0 );
        MSE = MSE / ( m_networkToTrain->m_numOutputs * trainingSet.size() );
    }

}