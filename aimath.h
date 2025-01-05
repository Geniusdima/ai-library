#ifndef AIMATH

#define AIMATH

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//a neuron stores incoming connections, not outgoing ones
typedef struct neuron
{
    float * weight;
    float value;
} neuron;

typedef struct layer
{
    int count;
    neuron * neurons;
} layer;


typedef struct neuralNetwork
{
    int count;
    layer * hiddenLayers;
    layer * inputLayer;
    layer * outputLayer;
} neuralNetwork;

extern neuralNetwork * setupPerceptronNetwork(int inputCount, int outCount, int hlayersCount, int neuronHLCount);
extern layer * setupLayers(int neuronCount, int layersCount,int prevLayerNCount);
extern neuron * setupNeurons(int neuronCount, int weightCount);
extern void clearNeuralNetwork(neuralNetwork * net);
extern void backpropagation(neuralNetwork * net, float correctAnswers[], float learningRate,float (*act)(float x));
extern void forwardpropagation(neuralNetwork * net,float (*act)(float x));
extern float ReLU(float x);
extern float sigmoid(float x);
extern float MSE(neuralNetwork * net, float correctAnswers[]);

#endif