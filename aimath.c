#include "aimath.h"

float ReLU(float x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

float dReLU(float x)
{
    if (x > 0)
        return 1;
    else
        return 0;
}

float dSigmoid(float x)
{
    return x * (1 - x);
}

float MSE(neuralNetwork * net, float correctAnswers[])
{
    float sum = 0;

    for (int i = 0; i < net->outputLayer->count; i++)
    {
        sum += pow(net->outputLayer->neurons[i].value - correctAnswers[i],2);
    }
    return (1.0 / net->outputLayer->count)*sum;
}


float sigmoid(float x)
{
    return 1/(1+exp(-x));
}

void forwardpropagation(neuralNetwork * net, float (*act)(float x))
{
    for (int i = 0; i < net->hiddenLayers[0].count; i++)
    {
        float sum=0;

        for (int j = 0; j < net->inputLayer->count; j++)
        {
            sum += net->inputLayer->neurons[j].value * net->hiddenLayers[0].neurons[i].weight[j];
        }
        net->hiddenLayers[0].neurons[i].value = act(sum);
    }

    for (int i = 0; i < net->count-1; i++)
    {
        for (int j = 0; j < net->hiddenLayers[i+1].count; j++)
        {
            float sum = 0;

            for(int k = 0; k < net->hiddenLayers[i].count; k++)
            {
                sum += net->hiddenLayers[i].neurons[k].value * net->hiddenLayers[i+1].neurons[j].weight[k];
            }
            net->hiddenLayers[i+1].neurons[j].value=act(sum);
        }
    }
    

    for (int i = 0; i < net->outputLayer->count; i++)
    {
        float sum=0;

        for (int j = 0; j < net->hiddenLayers[net->count-1].count; j++)
        {
            sum += net->hiddenLayers[net->count-1].neurons[j].value * net->outputLayer->neurons[i].weight[j];
        }
        net->outputLayer->neurons[i].value = act(sum);
    }
}


void backpropagation(neuralNetwork * net, float correctAnswers[], float learningRate,float (*act)(float x))
{
    forwardpropagation(net,act);

    float (*dact)(float x);
    if(act == &sigmoid)
    {
        dact = dSigmoid;
    }
    else if(act == &ReLU)
    {
        dact = dReLU;
    }

    float * deltaPrev = (float*) malloc(sizeof(float) * net->outputLayer->count);

    for (int i = 0; i < net->outputLayer->count; i++) 
    {
        deltaPrev[i] = (net->outputLayer->neurons[i].value - correctAnswers[i]) * dact(net->outputLayer->neurons[i].value);
    }

    for (int i = net->count - 1; i >= 0; i--)
    {
        layer * curr = &net->hiddenLayers[i];
        layer * next;
        
        if (i == net->count - 1) 
        {
            next = net->outputLayer;
        } 
        else 
        {
            next = &net->hiddenLayers[i + 1];
        }

        float * deltaCurr = (float*) malloc(sizeof(float) * curr->count);

        for (int j = 0; j < curr->count; j++)
        {
            float deltaSum = 0;
            for (int k = 0; k < next->count; k++)
            {
                deltaSum += next->neurons[k].weight[j] * deltaPrev[k]; 
            }
            deltaCurr[j] = deltaSum * dact(curr->neurons[j].value);
        }

        for (int j = 0; j < next->count; j++) 
        {
            for (int k = 0; k < curr->count; k++) 
            {
                next->neurons[j].weight[k] -= learningRate * deltaCurr[k] * curr->neurons[k].value; 
            }
        }
        free(deltaPrev);
        deltaPrev = deltaCurr; 
    }
    free(deltaPrev);
}