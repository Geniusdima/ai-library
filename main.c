#include "aimath.h"

int main()
{   
    neuralNetwork * net = setupPerceptronNetwork(3,3,4,4);
    
    float arr[]={1,2,3};

    long int g=0;
    srand(time(NULL));
    while(g<50000)
    {
        net->inputLayer->neurons[0].value=1;
        net->inputLayer->neurons[1].value=2;
        net->inputLayer->neurons[2].value=3;

        backpropagation(net,arr,0.1);

        float error = MSE(net, arr);


        printf("%d %.3f\n",g,error);

        g++;
    }

    clearNeuralNetwork(net);
}