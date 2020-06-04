#include "iftTensorflow.h"
#include <time.h>

float mean(float* data, int n) {
    float s = 0.0;
    for(int i=0; i<n; ++i) {
        //printf("%f\n", data[i]);
        s += data[i] / n;
    }
    return s;
}

int main(const int argc, const char* argv) {

    printf("load model\n");

    iftTensorflowModel* model = NULL;
    model = iftLoadTensorflowModel("/home/peixinho/model_v4.pb");

    iftInitTensorflowSession(model);

    int inputDims[] = {5, 224, 224,3};
    int outputDims[] = {5, 2};

    model->input = iftTensorflowGetBinding(model, "input_1", 4, inputDims, TF_FLOAT);
    model->output = iftTensorflowGetBinding(model, "predictionsa_new/Softmax", 2, outputDims, TF_FLOAT);

    float* output = (float*) (model->output->val);
    float* input = (float*) (model->input->val);

    printf("%p\n", model->input->val);
    printf("%p\n", model->output->val);

    /*float data[] = { 0.        , 0.0257732 , 0.01030928, 0.1185567 , 0.20618557,
            0.28865979, 0.30412371, 0.33505155, 0.36597938, 0.42268041,
            0.43814433, 0.55670103, 0.73195876, 0.69587629, 0.77835052,
            0.85051546, 0.8814433 , 0.81443299, 0.92783505, 0.98969072,
            1. };*/

    for(int i=0; i<224*224*3; ++i) {
        input[i] = 0.0;
    }

    /*for(int i=0; i<1; ++i) {
        output[i] = 10.0;
    }*/

    output = (float*) TF_TensorData( model->output->tensor );

    printf("input: %f\n", mean(input, 224*224*3));
    printf("output: %f\n", mean(output, 2));
    iftTensorflowRun(model);

    clock_t t; 
    t = clock(); 
    
    for (int i=0; i<1000; ++i)
        iftTensorflowRun(model);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

    printf("Time spent: %lf\n", time_taken);

    output = (float*) (model->output->val);

    printf("input: %f\n", mean(input, 21));
    printf("output: %f\n", mean(output, 4*52*52*5));

    for(int i=0; i<1*10; ++i) {
        printf("output[%d] = %f\n", i, output[i]);
    }

    iftFinishTensorflowSession(model);
    printf("%p\n", input);
    printf("%p\n", output);

    iftDestroyTensorflowModel(&model);

    return 0;
}
