import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.layers import Layer
from data_utils import MnistHandler, SortedNumberGenerator
import matplotlib.pyplot as plt

# Define the CPCLayer class
class CPCLayer(Layer):
    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

# Load the CPC model
cpc_model_path = '/home/paperspace/tinyfaces-cpc/models/64x64/cpc.h5'
cpc_model = load_model(cpc_model_path, custom_objects={'CPCLayer': CPCLayer})

# Initialize the MNIST handler
mnist_handler = MnistHandler()

# Parameters
batch_size = 32
terms = 4
predict_terms = 4  # Should match the trained model's expected terms for prediction
image_size = 64

# Get a batch of images for testing
sorted_number_generator = SortedNumberGenerator(batch_size=batch_size, subset='test', terms=terms, positive_samples=batch_size // 2, predict_terms=predict_terms, image_size=image_size, color=True, rescale=True)
(x_images, y_images), labels = next(sorted_number_generator)

# Make predictions using the CPC model
predictions = cpc_model.predict([x_images, y_images])

# Display the context images and predicted images
def plot_sequences(x, y, predictions, labels=None, output_path=None, max_subplots=256):
    ''' Draws a plot where sequences of numbers can be studied conveniently '''
    n_batches = x.shape[0]
    n_terms = x.shape[1] + y.shape[1]
    
    # Calculate total plots needed
    total_plots = n_batches * (n_terms + 1)
    
    # Determine the number of figures needed
    num_figures = int(np.ceil(total_plots / max_subplots))
    
    for fig_num in range(num_figures):
        plt.figure(figsize=(15, 15))
        start_idx = fig_num * (max_subplots // (n_terms + 1))
        end_idx = min(start_idx + (max_subplots // (n_terms + 1)), n_batches)
        
        counter = 1
        for n_b in range(start_idx, end_idx):
            for n_t in range(n_terms):
                plt.subplot(end_idx - start_idx, n_terms + 1, counter)
                if n_t < x.shape[1]:
                    plt.imshow(x[n_b, n_t, :, :, :])
                else:
                    plt.imshow(y[n_b, n_t - x.shape[1], :, :, :])
                plt.axis('off')
                counter += 1
            plt.subplot(end_idx - start_idx, n_terms + 1, counter)
            plt.text(0.5, 0.5, str(predictions[n_b][0]), fontsize=12, ha='center')
            plt.axis('off')
            counter += 1
        
        if output_path is not None:
            plt.savefig(f"{output_path}_part_{fig_num}.png", dpi=600)
        else:
            plt.show()

# Ensure the output directory exists
output_dir = '../predictions'
os.makedirs(output_dir, exist_ok=True)

# Plot the sequences with predictions and save them
output_path = os.path.join(output_dir, 'plot')
plot_sequences(x_images, y_images, predictions, labels, output_path)
