// hader structure

#ifndef CNN_H
#define CNN_H

 // layer type

typedef enum _LayerType {
    Input = 0,
    Hidden,
    Output
} LayerType;


// layer structure

typedef struct _Layer{

    // Layer position
    int id;
    struct _Layer *lprev;
    struct _Layer *lnext;

    // shape
    int depth, height, width; 
    
    // Network Neuron
    int nNodes; // number of Neurons
    float *output; // output value (trained)
    float *Gradients; // Gradients value (trained)
    float *error; // error value (trained)
    
    // biases 
    int nBias; // number of bias 
    float *bias; // bias value (trained)
    float *u_bias // bias update value
    
    // weights
    int nWeights; // number of weights
    float *weights; // weights value (trained)
    float *u_weights; // weights update value
    
    LayerType ltype;
    
    union {
        // full connection dense
        struct {}full;
        
        // convolutional
        struct {
            int ksize; // kernel size
            int stride; // stride
            int padding; // padding
        }conv;

    }; 

} Layer;

// create input layer structure (depth, height, width)
Layer *createInputLayer(int depth, int height, int width);

// create hidden layer full (lprev, nNodes , std)
Layer *createHiddenLayer(Layer *lprev, int nNodes, float std);

// create convolutional layer (lprev,depth, height, width, ksize, stride, padding, std) 
Layer *createConvolutionalLayer(Layer *lprev, int depth, int height, int width, int ksize, int stride, int padding, float std); 

// Layer free Release
void freeLayer(Layer *l);

// Layer dump show debug output
void dumpLayer(Layer *l, FILE *fp);

// Layer set_inpute set input value

void setInput(Layer *l, float *input);

void Layer_getOutputs(const Layer* l, double* outputs);

/* Layer_getErrorTotal(self)
   Gets the error total.
*/
double Layer_getErrorTotal(const Layer* l);

/* Layer_learnOutputs(self, values)
   Learns the output values.
*/
void Layer_learnOutputs(Layer* l, const double* values);

/* Layer_update(self, rate)
   Updates the weights.
*/
void Layer_update(Layer* l, double rate);


#endif
