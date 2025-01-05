#include "aimath.h"

neuralNetwork * setupPerceptronNetwork(int inputCount, int outCount, int hlayersCount, int neuronHLCount)
{
    neuralNetwork * net = (neuralNetwork*) malloc(sizeof(neuralNetwork));
    
    net->count=hlayersCount;
    net->hiddenLayers = setupLayers(neuronHLCount, hlayersCount,inputCount);
    
    net->inputLayer = setupLayers(inputCount,1,0);
    net->outputLayer = setupLayers(outCount,1,neuronHLCount);
    
    return net;
}

layer * setupLayers(int neuronCount, int layersCount,int prevLayerNCount)
{
    layer * layers = (layer*) malloc(sizeof(layer)*layersCount);

    layers[0].count=neuronCount;
    layers[0].neurons=setupNeurons(neuronCount,prevLayerNCount);

    for (int i = 1; i < layersCount; i++)
    {
        layers[i].count=neuronCount;
        layers[i].neurons=setupNeurons(neuronCount,neuronCount);
    }

    return layers; 
}

neuron * setupNeurons(int neuronCount, int weightCount)
{
    neuron * neurons = (neuron*) malloc(sizeof(neuron)*neuronCount);

    for(int i=0;i<neuronCount;i++)
    {   
        neurons[i].weight = (float*) malloc(sizeof(float)*weightCount);
    }

    for(int i=0;i<neuronCount;i++)
    {
        for (int j = 0; j< weightCount; j++)
        {
            neurons[i].weight[j] = (rand() % 2001 / 1000.0 - 1);
        }        
    }

    return neurons;
}

void clearNeuralNetwork(neuralNetwork * net)
{
    for(int i=0;i<net->count;i++)
    {
        for(int j=0;j < net->hiddenLayers[i].count;j++)
            free(net->hiddenLayers[i].neurons[j].weight);
        free(net->hiddenLayers[i].neurons);
    }
    free(net->hiddenLayers);


    for(int i=0;i<net->inputLayer->count;i++)
        free(net->inputLayer->neurons[i].weight);

    free(net->inputLayer->neurons);
    free(net->inputLayer);


    for(int i=0;i<net->outputLayer->count;i++)
        free(net->outputLayer->neurons[i].weight);

    free(net->outputLayer->neurons);
    free(net->outputLayer);


    free(net);
}