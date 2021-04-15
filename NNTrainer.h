// Feed forward NN Trainer using gradient descent with Momentum
#pragma once

#include "NeuralNetwork.h"
#include <fstream>

namespace BPN
{
    struct TrainingEntry
    {
        std::vector<double>         m_inputs;
        std::vector<int32_t>        m_expectedOutputs;
    };

    typedef std::vector<TrainingEntry> TrainingSet;

    struct TrainingData
    {
        TrainingSet m_trainingSet;
        TrainingSet m_testSet;
    };

    //-------------------------------------------------------------------------

    class NNTrainer
    {
    public:

        struct Settings
        {
            // Learning params
            double      m_learningRate = 0.01;
            double      m_momentum = 0.9;

            // Stopping conditions
            uint32_t    m_maxGenerations = 1500;
            double      m_desiredAccuracy = 85;
        };

    public:

        NNTrainer( Settings const& settings, Network* networkToTrain );

        void Train( TrainingData const& trainingData );

    private:

        inline double GetOutputErrorGradient( double desiredValue, double outputValue ) const { return outputValue * ( 1.0 - outputValue ) * ( desiredValue - outputValue ); }
        double GetHiddenErrorGradient( int32_t hiddenIdx ) const;

        void RunGeneration( TrainingSet const& trainingSet );
        void Backpropagate( std::vector<int32_t> const& expectedOutputs );
        void UpdateWeights();

        void GetSetAccuracyAndMSE( TrainingSet const& trainingSet, double& accuracy, double& mse ) const;

    private:
        
        Network*                    m_networkToTrain;                 // Network to train

        // Training settings
        double                      m_learningRate;             // Sets the step size of the weight update
        double                      m_momentum;                 // Improves stochastic learning 
        double                      m_desiredAccuracy;          // Target accuracy for training
        uint32_t                    m_maxGenerations;                // Max number of training Generations

        // Training data
        std::vector<double>         m_deltaInputHidden;         // Delta of input hidden layer
        std::vector<double>         m_deltaHiddenOutput;        // Delta of hidden output layer
        std::vector<double>         m_errorGradientsHidden;     // Error gradients for the hidden layer
        std::vector<double>         m_errorGradientsOutput;     // Error gradients for the outputs

        uint32_t                    m_currentGeneration;             // Generation counter
        double                      m_trainingSetAccuracy;
        double                      m_testSetAccuracy;
        double                      m_trainingSetMSE;
        double                      m_testSetMSE;
		std::fstream logFile;
    };
}