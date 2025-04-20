// cnn.h
#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>

// Activation function type
typedef enum {
    ACTIVATION_NONE = 0,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX
} ActivationType;

// Padding type
typedef enum {
    PADDING_VALID = 0, // No padding
    PADDING_SAME       // Zero-padding to maintain input size
} PaddingType;

// Layer type
typedef enum {
    LAYER_INPUT = 0,
    LAYER_CONVOLUTIONAL,
    LAYER_POOLING,
    LAYER_FULLY_CONNECTED,
    LAYER_BATCH_NORM,
    LAYER_DROPOUT,
    LAYER_OUTPUT
} LayerType;

// Pooling type (for pooling layers)
typedef enum {
    POOLING_MAX = 0,
    POOLING_AVERAGE
} PoolingType;

// Tensor shape (generalized for inputs, outputs, weights, etc.)
typedef struct {
    int batch;  // Batch size
    int depth;  // Number of channels (or filters for conv)
    int height; // Height of feature map
    int width;  // Width of feature map
} TensorShape;

// Tensor data (stores actual values)
typedef struct {
    TensorShape shape;
    double *data; // Flattened array of values
} Tensor;

// Layer-specific configuration (union for different layer types)
typedef union {
    // Convolutional layer
    struct {
        int kernel_size;  // Kernel size (square)
        int stride;       // Stride
        PaddingType padding; // Padding mode
        int num_filters;  // Number of output filters
    } conv;

    // Pooling layer
    struct {
        int kernel_size;  // Pooling window size
        int stride;       // Stride
        PoolingType pool_type; // Max or average pooling
    } pool;

    // Fully connected layer
    struct {
        int num_neurons; // Number of neurons in the layer
    } fc;

    // Batch normalization
    struct {
        double epsilon; // Small value for numerical stability
        double *gamma;   // Scale parameter
        double *beta;    // Shift parameter
        double *mean;    // Running mean
        double *variance; // Running variance
    } batch_norm;

    // Dropout
    struct {
        double dropout_rate; // Probability of dropping a neuron
    } dropout;
} LayerConfig;

// Forward declaration of Layer
typedef struct Layer Layer;

// Layer structure
struct Layer {
    // Layer metadata
    int id;              // Unique layer ID
    LayerType type;      // Type of layer
    const char *name;    // Optional layer name for debugging

    // Layer connectivity
    Layer *prev;         // Previous layer
    Layer *next;         // Next layer

    // Input and output tensors
    Tensor input;        // Input tensor
    Tensor output;       // Output tensor

    // Parameters (weights, biases, etc.)
    Tensor weights;      // Weights (if applicable)
    Tensor biases;       // Biases (if applicable)
    Tensor weight_grads; // Gradients for weights
    Tensor bias_grads;   // Gradients for biases

    // Activation function
    ActivationType activation;

    // Layer-specific configuration
    LayerConfig config;

    // Function pointers for layer operations
    void (*forward)(Layer *self);           // Forward pass
    void (*backward)(Layer *self);          // Backward pass
    void (*update)(Layer *self, double learning_rate); // Update parameters

    // Debugging and serialization
    void (*dump)(const Layer *self, FILE *fp); // Debug output
    void (*save)(const Layer *self, FILE *fp); // Save layer to file
    void (*load)(Layer *self, FILE *fp);       // Load layer from file
};

// Function declarations

// Create layers
Layer *create_input_layer(int batch, int depth, int height, int width);
Layer *create_convolutional_layer(const Layer *prev, int num_filters, int kernel_size, 
                                 int stride, PaddingType padding, ActivationType activation);
Layer *create_pooling_layer(const Layer *prev, int kernel_size, int stride, 
                           PoolingType pool_type);
Layer *create_fully_connected_layer(const Layer *prev, int num_neurons, 
                                   ActivationType activation);
Layer *create_batch_norm_layer(const Layer *prev, double epsilon);
Layer *create_dropout_layer(const Layer *prev, double dropout_rate);
Layer *create_output_layer(const Layer *prev, int num_classes, ActivationType activation);

// Free layer memory
void free_layer(Layer *layer);

// Set input data
void set_input(Layer *layer, const double *data);

// Get output data
void get_output(const Layer *layer, double *output);

// Forward pass for the entire network
void forward_network(Layer *input_layer);

// Backward pass for the entire network
void backward_network(Layer *output_layer, const double *targets);

// Update parameters for the entire network
void update_network(Layer *input_layer, double learning_rate);

// Compute total error
double get_total_error(const Layer *output_layer, const double *targets);

// Save and load network
void save_network(const Layer *input_layer, const char *filename);
void load_network(Layer *input_layer, const char *filename);

#endif
