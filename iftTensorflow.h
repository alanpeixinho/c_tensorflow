
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>

/**
  * @author Peixinho
  * @date Jun, 2020
  */
typedef struct iftTensorflowBinding {
    int n;
    TF_Operation* op;
    TF_Tensor* tensor;
    
    //const char* name;
    //stores the data
    void* val;

} iftTensorflowBinding;

/**
 * @author Peixinho
 * @date Jun, 2020
 */
typedef struct iftTensorflowModel {
  TF_Graph* graph;
  TF_Buffer* graph_def;
  TF_Session* session;
  TF_Status* status;

  iftTensorflowBinding * input;
  iftTensorflowBinding* output;

} iftTensorflowModel;

void free_buffer(void* data, size_t length) { free(data); }

void deallocator(void* ptr, size_t len, void* arg) { free((void*)ptr); }

//from: http://iamsurya.com/inference-in-c-using-tensorflow/
TF_Buffer* tf_read_file(const char* file) {
	FILE* f = fopen(file, "rb");

    if (f==NULL) {
        return NULL;
    }

	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	void* data = malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer;
	return buf;
}

iftTensorflowBinding* iftTensorflowGetBinding(iftTensorflowModel* model, const char* nodeName, int ndims,  int dims[], TF_DataType dtype) {
    iftTensorflowBinding* binding = (iftTensorflowBinding*) calloc(1, sizeof(iftTensorflowBinding));

    printf("Input alloced .. %p\n", binding);

    printf("name: %s\n", nodeName);
    binding->op = TF_GraphOperationByName(model->graph, nodeName);
    printf("Operation got %p\n", binding->op);
    binding->n = TF_OperationNumOutputs(binding->op);
    printf("Operation num inputs got %d\n", binding->n);
    //binding->name = nodeName;

    printf("dtype: %d\n", dtype);

    //create tensor binding
    int dsize = dtype==TF_INT32 ? sizeof(int32_t) : sizeof(float);
    printf("INT32 = %d -> %d\n", TF_INT32, sizeof(int32_t));
    printf("FLOAT = %d -> %d\n", TF_FLOAT, sizeof(float));
    int64_t tfdims[ndims];
    int ndata = 1;
    for (int i=0; i<ndims; ++i) {
        tfdims[i] = dims[i];
        ndata *= dims[i];
    }

    //input->val = malloc(ndata * dsize);
    //printf("val: %p\n", input->val);
    //input->tensor = TF_NewTensor(dtype, tfdims, ndims, input->val, ndata*dsize, &deallocator, NULL);
    binding->tensor = TF_AllocateTensor(dtype, tfdims, ndims, ndata*dsize);
    binding->val = TF_TensorData( binding->tensor );

    printf("ndata = %d dsize = %d\n", ndata, dsize);

    return binding;
}

iftTensorflowModel* iftLoadTensorflowModel(const char* frozenModel) {

    printf("alloca model\n");
	iftTensorflowModel* model = (iftTensorflowModel*) calloc(1, sizeof(iftTensorflowModel));

	model->graph = TF_NewGraph();
    model->graph_def = tf_read_file(frozenModel);

    printf("loaded model\n");
    if (model->graph_def == NULL) {
        printf("Model %s not found.\n", frozenModel);
        return NULL;
    }

    model->status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(model->graph, model->graph_def, opts, model->status);
	TF_DeleteImportGraphDefOptions(opts);
	if (TF_GetCode(model->status) != TF_OK) {
		printf("ERROR: Unable to import graph %s\n", frozenModel);
		return NULL;
	}

    printf("return model\n");

	return model;
}

int iftInitTensorflowSession(iftTensorflowModel* model) {
    TF_SessionOptions* opt = TF_NewSessionOptions();
    model->session = TF_NewSession(model->graph, opt, model->status);
    TF_DeleteSessionOptions(opt);
    if (TF_GetCode(model->status) != TF_OK) {
        printf("Unable to Create Session \n");
        return 1;
    }
    return 0;
}

void iftFinishTensorflowSession(iftTensorflowModel* model) {
  TF_CloseSession(model->session, model->status);
}

//TODO:run valgrind to check if everything is being freed
void iftDestroyTensorflowModel(iftTensorflowModel** model) {

  iftTensorflowModel* m = *model;
  TF_DeleteSession(m->session, m->status);

  TF_DeleteStatus(m->status);
  TF_DeleteBuffer(m->graph_def);

  TF_DeleteGraph(m->graph);

  iftDestroyTensorflowBinding(&(m->input));
  iftDestroyTensorflowBinding(&(m->output));

  free(*model);
}

iftDestroyTensorflowBinding(iftTensorflowBinding** binding) {

    iftTensorflowBinding* b = *binding;

    TF_DeleteTensor(b->tensor);
    //free(b->op);
    free(b);
}

int iftTensorflowRun(iftTensorflowModel* model) {

    //Tensorflow Bindings, or how I remembered that simple things can get ugly when poorly done in C
    TF_Tensor* run_input_tensors [] = { model->input->tensor };
    //TF_Tensor* run_output_tensors [] = { model->output->tensor };

    TF_Tensor* run_output_tensors[1];

    TF_Output run_input = { .oper=model->input->op, .index=0 };
    TF_Output run_inputs [] = { run_input };

    TF_Output run_output = { .oper=model->output->op, .index=0 };
    TF_Output run_outputs [] = { run_output };

    //printf("input name: %s\n",  model->input->name);
    //printf("output name: %s\n", model->output->name);

    //printf("run_output = %p\n", run_output);
    //printf("run_input = %p\n", run_input);

    //printf("run_output_tensors[0] = %p\n", run_output_tensors[0]);
    //printf("run_output_tensors[0] = %p\n", model->output->tensor);


    TF_SessionRun(model->session,
                /* RunOptions */ NULL,
                /* Input tensors */ run_inputs, run_input_tensors, 1,
                /* Output tensors */ run_outputs, run_output_tensors, 1,
                /* Target operations */ NULL, 0,
                /* RunMetadata */ NULL,
                /* Output status */ model->status);
    if (TF_GetCode(model->status) != TF_OK) {
      fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message( model->status ));
      return 1;
    }

    //printf("TF Message: %s\n", TF_Message( model->status ));

    //printf("after run ...\n");
    //printf("input: %p\n", TF_TensorData( model->input->tensor ));
    //printf("output: %p\n", TF_TensorData( model->output->tensor ));

    //printf("run_output_tensors[0] = %p\n", run_output_tensors[0]);
    //printf("run_output_tensors[0] = %p\n", model->output->tensor);

    void * output_data = TF_TensorData(run_output_tensors[0]);

    //printf("output same: %p\n", output_data);
    //printf("output same data: %f\n", ((float*)output_data)[0]);

    model->output->tensor = run_output_tensors[0];
    model->output->val = output_data;

    return 0;
}
