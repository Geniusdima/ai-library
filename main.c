#include "aimath.h"

int main()
{   
    neuralNetwork * net = setupPerceptronNetwork(3,3,4,4);
    
    float arr[]={1,2,3};

    int i=0;

    while(1)
    {
        net->inputLayer->neurons[0].value=1;
        net->inputLayer->neurons[1].value=2;
        net->inputLayer->neurons[2].value=3;

        backpropagation(net,arr,0.01);

        float error = MSE(net, arr);

        if (i % 10000 == 0)
        {
            printf("%d %.3f\n",i,error);
        }
        i++;
    }
    forwardpropagation(net);

    clearNeuralNetwork(net);
}