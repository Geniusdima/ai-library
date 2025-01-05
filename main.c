#include "aimath.h"

#define NUM_INPUTS 2
#define NUM_HIDDEN 1
#define NUM_OUTPUTS 1
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.1
#define EPOCHS 400000

float inputs[NUM_SAMPLES][NUM_INPUTS] = 
{
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

float outputs[NUM_SAMPLES][NUM_OUTPUTS] = 
{
    {0},
    {1},
    {1},
    {0}
};


int main()
{   
    srand(time(NULL));
    neuralNetwork * net = setupPerceptronNetwork(NUM_INPUTS,NUM_OUTPUTS,1,3);

    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        for (int i = 0; i < NUM_SAMPLES; i++) 
        {
            net->inputLayer->neurons[0].value = inputs[i][0];
            net->inputLayer->neurons[1].value = inputs[i][1];

            backpropagation(net, outputs[i], LEARNING_RATE, &sigmoid);
        }
    }
    

    for (int i = 0; i < NUM_SAMPLES; i++) 
    {
        net->inputLayer->neurons[0].value = inputs[i][0];
        net->inputLayer->neurons[1].value = inputs[i][1];
        forwardpropagation(net,&sigmoid);

        printf("Input: (%.1f, %.1f) - Output: %.7f\n", inputs[i][0], inputs[i][1], net->outputLayer->neurons[0].value);
    }


    clearNeuralNetwork(net);
}