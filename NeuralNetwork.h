// Neural network with a single hidden layer
#pragma once
#include <stdint.h>
#include <vector>

//-------------------------------------------------------------------------

namespace BPN
{
    class Network
    {
        friend class NNTrainer;

    public:

        struct Settings
        {
            uint32_t                        m_numInputs;
            uint32_t                        m_numHidden;
            uint32_t                        m_numOutputs;
        };

    public:

        Network( Settings const& settings );
		std::string const& Evaluate(std::vector<double> const& input);

        std::vector<double> const& GetInputHiddenWeights() const { return m_weightsInputHidden; }
        std::vector<double> const& GetHiddenOutputWeights() const { return m_weightsHiddenOutput; }
		std::string				m_suggestedFlower;

    private:
        void InitializeNetwork();
        void InitializeWeights();

        int32_t GetInputHiddenWeightIndex( int32_t inputIdx, int32_t hiddenIdx ) const { return inputIdx * m_numHidden + hiddenIdx; }
        int32_t GetHiddenOutputWeightIndex( int32_t hiddenIdx, int32_t outputIdx ) const { return hiddenIdx * m_numOutputs + outputIdx; }

    private:

        int32_t                 m_numInputs;
        int32_t                 m_numHidden;
        int32_t                 m_numOutputs;

        std::vector<double>     m_inputNeurons;
        std::vector<double>     m_hiddenNeurons;
        std::vector<double>     m_outputNeurons;

        std::vector<int32_t>    m_clampedOutputs;

        std::vector<double>     m_weightsInputHidden;
        std::vector<double>     m_weightsHiddenOutput;

    };
}