"""
SNN-PC MNIST Classification Demo
Real MNIST data with classification evaluation
1. AdEx neuron model
2. Synaptic transmission with exponential filters
3. Predictive coding hierarchy with positive/negative error units
4. Feedforward gist pathway
5. Hebbian learning mechanism
6. MNIST train
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import gzip
import os
import urllib.request
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def download_mnist():
    """Skip download since files are already available"""
    print("Using existing MNIST dataset files...")
    
def load_mnist_images(filename):
    """Load MNIST images from uncompressed file"""
    try:
        # Try compressed file first
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        else:
            # Try uncompressed file
            with open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    except:
        # Fallback to different filename format
        alt_filename = filename.replace('-idx3-ubyte.gz', '.idx3-ubyte').replace('mnist_data/', 'mnist_data/')
        with open(alt_filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    """Load MNIST labels from uncompressed file"""
    try:
        # Try compressed file first  
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            # Try uncompressed file
            with open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    except:
        # Fallback to different filename format
        alt_filename = filename.replace('-idx1-ubyte.gz', '.idx1-ubyte').replace('mnist_data/', 'mnist_data/')
        with open(alt_filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

@dataclass
class AdExParams:
    """Parameters for Adaptive Exponential Integrate-and-Fire neuron"""
    C_m: float = 281e-12  # Membrane capacitance (pF)
    g_L: float = 30e-9    # Leak conductance (nS) 
    E_L: float = -70.6e-3 # Leak reversal potential (mV)
    V_theta: float = -50.4e-3  # Spike threshold (mV)
    V_r: float = -70.6e-3      # Reset potential (mV)
    Delta_T: float = 2e-3      # Slope factor (mV)
    t_ref: float = 2e-3        # Refractory period (ms)
    c: float = 4e-9            # Adaptation coupling (nS)
    b: float = 0.0805e-9       # Adaptation increment (nA)
    tau_a: float = 144e-3      # Adaptation time constant (ms)
    tau_rise: float = 5e-3     # Synaptic rise time (ms)
    tau_decay: float = 50e-3   # Synaptic decay time (ms)
    dt: float = 1e-3           # Time step (ms)

class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire neuron model"""
    
    def __init__(self, params: AdExParams):
        self.params = params
        self.reset()
        
    def reset(self):
        """Reset neuron state"""
        self.V = self.params.E_L  # Membrane potential
        self.a = 0.0              # Adaptation variable
        self.I_syn = 0.0          # Synaptic current
        self.last_spike = -np.inf # Last spike time
        self.spikes = []          # Spike history
        self.X = 0.0              # Spike trace (for synaptic transmission)
        self.Y = 0.0              # Intermediate synaptic variable
        
    def update(self, I_ext: float, t: float) -> bool:
        """Update neuron state for one time step"""
        dt = self.params.dt
        
        # Check refractory period
        if t - self.last_spike < self.params.t_ref:
            return False
            
        # Update synaptic variables (Y -> X)
        dY_dt = -self.Y / self.params.tau_decay
        self.Y += dY_dt * dt
        
        dX_dt = self.Y / self.params.tau_rise - self.X / self.params.tau_decay
        self.X += dX_dt * dt
        
        # Membrane potential dynamics
        exp_term = self.params.g_L * self.params.Delta_T * np.exp(
            np.clip((self.V - self.params.V_theta) / self.params.Delta_T, -10, 10)
        )
        
        dV_dt = (
            -self.params.g_L * (self.V - self.params.E_L) +
            exp_term + I_ext + self.I_syn - self.a
        ) / self.params.C_m
        
        self.V += dV_dt * dt
        
        # Adaptation dynamics
        da_dt = (
            self.params.c * (self.V - self.params.E_L) - self.a
        ) / self.params.tau_a
        
        self.a += da_dt * dt
        
        # Check for spike
        spiked = False
        if self.V > self.params.V_theta:
            # Spike occurred
            self.V = self.params.V_r
            self.a += self.params.b
            self.Y = 1.0  # Trigger synaptic release
            self.last_spike = t
            self.spikes.append(t)
            spiked = True
            
        return spiked

class PredictiveCodingLayer:
    """Single layer in the predictive coding hierarchy"""
    
    def __init__(self, n_units: int, params: AdExParams, layer_idx: int = 0):
        self.n_units = n_units
        self.layer_idx = layer_idx
        self.params = params
        
        # Create neurons for each unit type
        self.R_neurons = [AdExNeuron(params) for _ in range(n_units)]  # Representation
        self.E_pos_neurons = [AdExNeuron(params) for _ in range(n_units)]  # Positive error
        self.E_neg_neurons = [AdExNeuron(params) for _ in range(n_units)]  # Negative error
        
        # Store activities for analysis
        self.R_activity = np.zeros(n_units)
        self.E_pos_activity = np.zeros(n_units)
        self.E_neg_activity = np.zeros(n_units)
        
    def update(self, bottom_up_input: np.ndarray, top_down_pred: np.ndarray, 
               gist_input: np.ndarray, t: float):
        """Update all neurons in the layer"""
        
        # Reset synaptic currents
        for neuron in self.R_neurons + self.E_pos_neurons + self.E_neg_neurons:
            neuron.I_syn = 0.0
        
        # Update representation neurons
        for i, neuron in enumerate(self.R_neurons):
            I_ext = 0.0
            
            # Bottom-up error signals
            if hasattr(self, 'prev_layer') and self.prev_layer is not None:
                if i < len(self.prev_layer.E_pos_activity):
                    I_ext += (self.prev_layer.E_pos_activity[i] - self.prev_layer.E_neg_activity[i]) * 1e-9
            
            # Top-down error signals (lateral connections)
            I_ext -= (self.E_pos_activity[i] - self.E_neg_activity[i]) * 1e-9
            
            # Gist input
            if gist_input is not None and i < len(gist_input):
                I_ext += gist_input[i] * 1e-9
                
            neuron.update(I_ext, t)
            
        # Update error neurons
        for i in range(self.n_units):
            # Positive error: input > prediction
            pos_input = bottom_up_input[i] if i < len(bottom_up_input) else 0.0
            pred_input = top_down_pred[i] if i < len(top_down_pred) else 0.0
            
            pos_error = max(0, pos_input - pred_input)
            neg_error = max(0, pred_input - pos_input)
                
            self.E_pos_neurons[i].update(pos_error * 1e-9, t)
            self.E_neg_neurons[i].update(neg_error * 1e-9, t)
            
        # Update activity traces
        self.R_activity = np.array([n.X for n in self.R_neurons])
        self.E_pos_activity = np.array([n.X for n in self.E_pos_neurons])
        self.E_neg_activity = np.array([n.X for n in self.E_neg_neurons])

class FeedforwardGistPathway:
    """Fast feedforward gist pathway"""
    
    def __init__(self, input_size: int, gist_size: int = 16, connection_prob: float = 0.05):
        self.input_size = input_size
        self.gist_size = gist_size
        
        # Create sparse random connections
        self.weights = np.random.normal(0, 0.1, (gist_size, input_size))
        mask = np.random.random((gist_size, input_size)) < connection_prob
        self.weights *= mask
        
        # Gist neurons
        self.gist_neurons = [AdExNeuron(AdExParams()) for _ in range(gist_size)]
        self.gist_activity = np.zeros(gist_size)
        
        # Projections to each layer
        self.layer_projections = {}
        
    def add_layer_projection(self, layer_idx: int, layer_size: int):
        """Add projection to a specific layer"""
        self.layer_projections[layer_idx] = np.random.normal(
            0, 0.1, (layer_size, self.gist_size)
        )
        
    def compute_gist(self, input_signal: np.ndarray, t: float) -> np.ndarray:
        """Compute gist representation of input"""
        gist_input = np.dot(self.weights, input_signal)
        
        # Reset synaptic currents
        for neuron in self.gist_neurons:
            neuron.I_syn = 0.0
        
        # Update gist neurons
        for i, neuron in enumerate(self.gist_neurons):
            neuron.update(gist_input[i] * 1e-9, t)
            
        self.gist_activity = np.array([n.X for n in self.gist_neurons])
        return self.gist_activity
        
    def project_to_layer(self, layer_idx: int) -> np.ndarray:
        """Project gist to specific layer"""
        if layer_idx in self.layer_projections:
            return np.dot(self.layer_projections[layer_idx], self.gist_activity)
        return None

class SNNPCNetwork:
    """Complete SNN-PC Network"""
    
    def __init__(self, layer_sizes: List[int], use_gist: bool = True):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.params = AdExParams()
        self.use_gist = use_gist
        
        # Create layers
        self.layers = []
        for i, size in enumerate(layer_sizes):
            layer = PredictiveCodingLayer(size, self.params, i)
            self.layers.append(layer)
            
        # Link layers
        for i in range(1, self.n_layers):
            self.layers[i].prev_layer = self.layers[i-1]
            
        # Create feedforward weights (for predictions)
        self.ff_weights = []
        for i in range(self.n_layers - 1):
            weights = np.random.normal(0, 0.1, (layer_sizes[i], layer_sizes[i+1]))
            weights = np.abs(weights)  # Ensure positive weights
            self.ff_weights.append(weights)
            
        # Create gist pathway
        if use_gist:
            self.gist_pathway = FeedforwardGistPathway(layer_sizes[0])
            for i in range(1, self.n_layers):
                self.gist_pathway.add_layer_projection(i, layer_sizes[i])
        else:
            self.gist_pathway = None
            
        # Learning parameters
        self.learning_rate = 1e-5
        self.regularization = 1e-6
        
        # History for analysis
        self.prediction_errors = [[] for _ in range(self.n_layers-1)]
        
    def forward(self, input_signal: np.ndarray, t: float):
        """Forward pass through the network"""
        
        # Compute gist if enabled
        gist_inputs = [None] * self.n_layers
        if self.gist_pathway is not None:
            self.gist_pathway.compute_gist(input_signal, t)
            for i in range(1, self.n_layers):
                gist_inputs[i] = self.gist_pathway.project_to_layer(i)
                
        # Forward pass through layers
        for i in range(self.n_layers):
            if i == 0:
                # Input layer
                bottom_up = input_signal
                if i < self.n_layers - 1:
                    top_down = np.dot(self.ff_weights[i], self.layers[i+1].R_activity)
                    # Ensure same size
                    if len(top_down) > len(bottom_up):
                        top_down = top_down[:len(bottom_up)]
                    elif len(top_down) < len(bottom_up):
                        top_down = np.pad(top_down, (0, len(bottom_up) - len(top_down)))
                else:
                    top_down = np.zeros_like(bottom_up)
            else:
                # Hidden layers
                bottom_up = self.layers[i-1].R_activity
                if i < self.n_layers - 1:
                    top_down = np.dot(self.ff_weights[i], self.layers[i+1].R_activity)
                    # Ensure same size
                    if len(top_down) > len(bottom_up):
                        top_down = top_down[:len(bottom_up)]
                    elif len(top_down) < len(bottom_up):
                        top_down = np.pad(top_down, (0, len(bottom_up) - len(top_down)))
                else:
                    top_down = np.zeros_like(bottom_up)
                          
            self.layers[i].update(bottom_up, top_down, gist_inputs[i], t)
            
        # Compute prediction errors
        for i in range(self.n_layers - 1):
            bottom_up = self.layers[i].R_activity
            prediction = np.dot(self.ff_weights[i], self.layers[i+1].R_activity)
            if len(prediction) > len(bottom_up):
                prediction = prediction[:len(bottom_up)]
            elif len(prediction) < len(bottom_up):
                prediction = np.pad(prediction, (0, len(bottom_up) - len(prediction)))
            error = np.mean((bottom_up - prediction) ** 2)
            self.prediction_errors[i].append(error)
            
    def train_step(self):
        """Perform one training step using Hebbian learning"""
        
        for i in range(self.n_layers - 1):
            # Get activities
            pre_activity = self.layers[i].E_pos_activity - self.layers[i].E_neg_activity
            post_activity = self.layers[i+1].R_activity
            
            # Ensure compatible sizes for outer product
            min_size = min(len(pre_activity), self.ff_weights[i].shape[0])
            max_size = min(len(post_activity), self.ff_weights[i].shape[1])
            
            if min_size > 0 and max_size > 0:
                # Hebbian update
                delta_w = self.learning_rate * np.outer(
                    pre_activity[:min_size], 
                    post_activity[:max_size]
                )
                
                # Ensure delta_w matches weight matrix size
                delta_w = delta_w[:self.ff_weights[i].shape[0], :self.ff_weights[i].shape[1]]
                
                # L1 regularization
                reg_term = self.regularization * np.sign(self.ff_weights[i])
                
                # Update weights
                self.ff_weights[i] += delta_w - reg_term
                self.ff_weights[i] = np.maximum(0, self.ff_weights[i])  # Keep positive
            
    def get_representation(self, layer_idx: int = 1) -> np.ndarray:
        """Get internal representation from specified layer"""
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].R_activity
        return np.array([])

#CLASSIFICATION MIGHT NOT BE TRUE
def classify_representation(representations: np.ndarray, labels: np.ndarray, 
                          test_representations: np.ndarray, test_labels: np.ndarray):
    """Simple classification using nearest centroid"""
    
    # Compute class centroids
    n_classes = 10
    centroids = np.zeros((n_classes, representations.shape[1]))
    
    for c in range(n_classes):
        class_mask = labels == c
        if np.sum(class_mask) > 0:
            centroids[c] = np.mean(representations[class_mask], axis=0)
    
    # Classify test samples
    predictions = []
    for test_rep in test_representations:
        # Compute distances to centroids
        distances = np.linalg.norm(centroids - test_rep, axis=1)
        predictions.append(np.argmin(distances))
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == test_labels)
    
    return predictions, accuracy, centroids

def run_mnist_classification_demo():
    """Run MNIST classification demonstration"""
    
    print("SNN-PC MNIST Classification Demo")
    print("=" * 40)
    
    # Download and load MNIST
    try:
        download_mnist()
        
        train_images = load_mnist_images('mnist_data/train-images.idx3-ubyte')
        train_labels = load_mnist_labels('mnist_data/train-labels.idx1-ubyte')
        test_images = load_mnist_images('mnist_data/t10k-images.idx3-ubyte')
        test_labels = load_mnist_labels('mnist_data/t10k-labels.idx1-ubyte')
        
        print(f"Loaded MNIST: {train_images.shape[0]} train, {test_images.shape[0]} test samples")
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Using synthetic data instead...")
        
        # Fallback to synthetic data
        train_images = np.random.random((1000, 28, 28))
        train_labels = np.random.randint(0, 10, 1000)
        test_images = np.random.random((200, 28, 28))
        test_labels = np.random.randint(0, 10, 200)
    
    # Use subset for faster demo
    n_train = min(1000, len(train_images))
    n_test = min(200, len(test_images))
    
    train_images = train_images[:n_train].reshape(n_train, -1)
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test].reshape(n_test, -1)
    test_labels = test_labels[:n_test]
    
    print(f"Using {n_train} training and {n_test} test samples")
    
    # Create networks
    layer_sizes = [784, 128, 64, 32]
    network_with_gist = SNNPCNetwork(layer_sizes, use_gist=True)
    network_without_gist = SNNPCNetwork(layer_sizes, use_gist=False)
    
    networks = {
        'With Gist': network_with_gist,
        'Without Gist': network_without_gist
    }
    
    # Training parameters
    T = 0.1  # Even shorter simulation time for speed
    dt = 1e-3  # Larger time step
    time_steps = int(T / dt)
    n_epochs = 10
    
    results = {}
    
    for network_name, network in networks.items():
        print(f"\nTraining {network_name} network...")
        
        # Training loop
        for epoch in range(n_epochs):
            print(f"  Epoch {epoch + 1}/{n_epochs}")
            
            # Shuffle training data
            indices = np.random.permutation(n_train)
            
            for i, idx in enumerate(indices[:50]):  # Use smaller subset for speed
                input_sample = train_images[idx]
                
                # Run simulation
                for step in range(time_steps):
                    t = step * dt
                    network.forward(input_sample, t)
                
                # Learning update
                network.train_step()
                
                if i % 25 == 0:
                    errors = [pe[-1] if pe else 0 for pe in network.prediction_errors]
                    print(f"    Sample {i}: Errors = {[f'{e:.2f}' for e in errors]}")
        
        # Extract representations
        print(f"  Extracting representations...")
        train_representations = []
        test_representations = []
        
        # Get training representations
        for i in range(n_train):
            input_sample = train_images[i]
            
            # Run simulation
            for step in range(time_steps):
                t = step * dt
                network.forward(input_sample, t)
            
            rep = network.get_representation(1)  # Use layer 1
            train_representations.append(rep)
        
        # Get test representations
        for i in range(n_test):
            input_sample = test_images[i]
            
            # Run simulation
            for step in range(time_steps):
                t = step * dt
                network.forward(input_sample, t)
            
            rep = network.get_representation(1)
            test_representations.append(rep)
        
        train_representations = np.array(train_representations)
        test_representations = np.array(test_representations)
        
        # Classification
        print(f"  Performing classification...")
        predictions, accuracy, centroids = classify_representation(
            train_representations, train_labels,
            test_representations, test_labels
        )
        
        results[network_name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'train_representations': train_representations,
            'test_representations': test_representations,
            'prediction_errors': network.prediction_errors
        }
        
        print(f"  Classification accuracy: {accuracy:.3f}")
    
    # Visualization
    create_classification_visualization(results, test_labels, test_images)
    
    return results

def create_classification_visualization(results, test_labels, test_images):
    """Create comprehensive visualization of classification results"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Classification accuracies
    ax1 = plt.subplot(3, 4, 1)
    networks = list(results.keys())
    accuracies = [results[net]['accuracy'] for net in networks]
    
    bars = ax1.bar(networks, accuracies, alpha=0.8)
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Classification Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Confusion matrices
    for i, (network_name, result) in enumerate(results.items()):
        ax = plt.subplot(3, 4, 2 + i)
        
        # Compute confusion matrix
        predictions = result['predictions']
        n_classes = 10
        confusion = np.zeros((n_classes, n_classes))
        
        for true_label, pred_label in zip(test_labels, predictions):
            confusion[true_label, pred_label] += 1
        
        # Normalize
        confusion = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax.imshow(confusion, cmap='Blues')
        ax.set_title(f'Confusion Matrix\n{network_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, f'{confusion[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    # 3. Prediction errors over training
    ax4 = plt.subplot(3, 4, 4)
    for network_name, result in results.items():
        for layer_idx, errors in enumerate(result['prediction_errors']):
            if len(errors) > 0:
                ax4.plot(errors, label=f'{network_name} L{layer_idx}', alpha=0.7)
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Prediction Error')
    ax4.set_title('Learning Dynamics')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 4. Sample reconstructions/classifications
    ax5 = plt.subplot(3, 4, 5)
    
    # Show first few test images with predictions
    n_samples = min(5, len(test_images))
    sample_images = test_images[:n_samples].reshape(n_samples, 28, 28)
    
    # Create a grid of sample images
    grid_img = np.zeros((28, 28 * n_samples))
    for i in range(n_samples):
        grid_img[:, i*28:(i+1)*28] = sample_images[i]
    
    ax5.imshow(grid_img, cmap='gray')
    ax5.set_title('Test Samples')
    ax5.axis('off')
    
    # Add prediction labels
    for i in range(n_samples):
        true_label = test_labels[i]
        with_gist_pred = results['With Gist']['predictions'][i]
        without_gist_pred = results['Without Gist']['predictions'][i]
        
        ax5.text(i*28 + 14, -2, f'True: {true_label}', ha='center', fontsize=8)
        ax5.text(i*28 + 14, 30, f'W/Gist: {with_gist_pred}', ha='center', fontsize=8, color='blue')
        ax5.text(i*28 + 14, 32, f'W/O Gist: {without_gist_pred}', ha='center', fontsize=8, color='red')
    
    # 5. Representation similarity (t-SNE-like visualization)
    ax6 = plt.subplot(3, 4, 6)
    
    # Simple 2D projection using PCA-like approach
    test_reps = results['With Gist']['test_representations']
    if test_reps.shape[1] >= 2:
        # Take first two principal components (simplified)
        mean_rep = np.mean(test_reps, axis=0)
        centered_reps = test_reps - mean_rep
        
        # SVD for dimensionality reduction
        if centered_reps.shape[0] > 2:
            U, s, Vt = np.linalg.svd(centered_reps, full_matrices=False)
            projections = U[:, :2] * s[:2]
        else:
            projections = centered_reps[:, :2]
        
        # Color by true labels
        scatter = ax6.scatter(projections[:, 0], projections[:, 1], 
                            c=test_labels, cmap='tab10', alpha=0.7)
        ax6.set_title('Representation Space\n(With Gist)')
        ax6.set_xlabel('PC1')
        ax6.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax6)
    
    # 6. Layer activities comparison
    ax7 = plt.subplot(3, 4, 7)
    
    # Compare average layer activities
    layer_activities_with = []
    layer_activities_without = []
    
    # Get average activities across test samples
    for layer_idx in range(len(results['With Gist']['test_representations'][0])):
        if layer_idx < 4:  # Limit to first 4 components for visualization
            with_gist_act = np.mean([rep[layer_idx] for rep in results['With Gist']['test_representations'] 
                                   if len(rep) > layer_idx])
            without_gist_act = np.mean([rep[layer_idx] for rep in results['Without Gist']['test_representations']
                                      if len(rep) > layer_idx])
            layer_activities_with.append(with_gist_act)
            layer_activities_without.append(without_gist_act)
    
    x = np.arange(len(layer_activities_with))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, layer_activities_with, width, label='With Gist', alpha=0.8)
    bars2 = ax7.bar(x + width/2, layer_activities_without, width, label='Without Gist', alpha=0.8)
    
    ax7.set_xlabel('Layer Components')
    ax7.set_ylabel('Average Activity')
    ax7.set_title('Layer Activities')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 7. Performance by digit class
    ax8 = plt.subplot(3, 4, 8)
    
    # Compute per-class accuracy
    class_accuracies_with = np.zeros(10)
    class_accuracies_without = np.zeros(10)
    
    for digit in range(10):
        digit_mask = test_labels == digit
        if np.sum(digit_mask) > 0:
            with_gist_preds = results['With Gist']['predictions'][digit_mask]
            without_gist_preds = results['Without Gist']['predictions'][digit_mask]
            
            class_accuracies_with[digit] = np.mean(with_gist_preds == digit)
            class_accuracies_without[digit] = np.mean(without_gist_preds == digit)
    
    x = np.arange(10)
    width = 0.35
    
    ax8.bar(x - width/2, class_accuracies_with, width, label='With Gist', alpha=0.8)
    ax8.bar(x + width/2, class_accuracies_without, width, label='Without Gist', alpha=0.8)
    
    ax8.set_xlabel('Digit Class')
    ax8.set_ylabel('Classification Accuracy')
    ax8.set_title('Per-Class Performance')
    ax8.set_xticks(x)
    ax8.set_xticklabels([str(i) for i in range(10)])
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snn_pc_mnist_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Classification results saved as 'snn_pc_mnist_classification.png'")

if __name__ == "__main__":
    print("SNN-PC MNIST Classification Demo")
    print("Loading real MNIST data and testing classification performance...")
    
    results = run_mnist_classification_demo()
    
    print("\nDemo completed!")
    print("\nSummary:")
    for network_name, result in results.items():
        print(f"  {network_name}: {result['accuracy']:.3f} accuracy") 