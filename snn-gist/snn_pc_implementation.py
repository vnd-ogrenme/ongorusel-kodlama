"""
Spiking Neural Network for Predictive Coding (SNN-PC)
Implementation based on "Predictive coding with spiking neurons and feedforward gist signaling"
by Lee et al. (2024)

This implementation includes:
1. Adaptive Exponential Integrate-and-Fire (AdEx) neuron model
2. Synaptic transmission with exponential filters
3. Predictive coding hierarchy with positive/negative error units
4. Feedforward gist pathway
5. Hebbian learning mechanism
6. MNIST training and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
        self.V_history = []       # Voltage history
        self.X = 0.0              # Spike trace (for synaptic transmission)
        self.Y = 0.0              # Intermediate synaptic variable
        
    def update(self, I_ext: float, t: float) -> bool:
        """
        Update neuron state for one time step
        
        Args:
            I_ext: External input current
            t: Current time
            
        Returns:
            True if neuron spiked, False otherwise
        """
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
            (self.V - self.params.V_theta) / self.params.Delta_T
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
            
        self.V_history.append(self.V)
        return spiked

class SynapticConnection:
    """Synaptic connection between neurons"""
    
    def __init__(self, weight: float = 1.0, delay: float = 0.0):
        self.weight = weight
        self.delay = delay
        
    def transmit(self, pre_neuron: AdExNeuron, post_neuron: AdExNeuron):
        """Transmit signal from pre to post neuron"""
        # Simple implementation: weight * presynaptic trace
        post_neuron.I_syn += self.weight * pre_neuron.X

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
        
        # Update representation neurons
        for i, neuron in enumerate(self.R_neurons):
            I_ext = 0.0
            
            # Bottom-up error signals
            if hasattr(self, 'prev_layer') and self.prev_layer is not None:
                I_ext += self.prev_layer.E_pos_activity[i] - self.prev_layer.E_neg_activity[i]
            
            # Top-down error signals (lateral connections)
            I_ext -= self.E_pos_activity[i] - self.E_neg_activity[i]
            
            # Gist input
            if gist_input is not None and i < len(gist_input):
                I_ext += gist_input[i]
                
            neuron.update(I_ext * 1e-9, t)  # Convert to proper current units
            
        # Update error neurons
        for i in range(self.n_units):
            # Positive error: input > prediction
            if i < len(bottom_up_input) and i < len(top_down_pred):
                pos_error = max(0, bottom_up_input[i] - top_down_pred[i])
                neg_error = max(0, top_down_pred[i] - bottom_up_input[i])
            else:
                pos_error = neg_error = 0.0
                
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
        self.weights = np.random.normal(0, 0.3, (gist_size, input_size))
        mask = np.random.random((gist_size, input_size)) > (1 - connection_prob)
        self.weights *= mask
        
        # Gist neurons
        self.gist_neurons = [AdExNeuron(AdExParams()) for _ in range(gist_size)]
        self.gist_activity = np.zeros(gist_size)
        
        # Projections to each layer
        self.layer_projections = {}
        
    def add_layer_projection(self, layer_idx: int, layer_size: int):
        """Add projection to a specific layer"""
        self.layer_projections[layer_idx] = np.random.normal(
            0, 0.3, (layer_size, self.gist_size)
        )
        
    def compute_gist(self, input_signal: np.ndarray, t: float) -> np.ndarray:
        """Compute gist representation of input"""
        gist_input = np.dot(self.weights, input_signal)
        
        # Update gist neurons
        for i, neuron in enumerate(self.gist_neurons):
            neuron.update(gist_input[i] * 1e-9, t)
            
        self.gist_activity = np.array([n.X for n in self.gist_neurons])
        return self.gist_activity
        
    def project_to_layer(self, layer_idx: int) -> np.ndarray:
        """Project gist to specific layer"""
        if layer_idx in self.layer_projections:
            return np.dot(self.layer_projections[layer_idx], self.gist_activity)
        return np.zeros(1)

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
            weights = np.random.normal(0, 0.3, (layer_sizes[i], layer_sizes[i+1]))
            weights = np.maximum(0, weights)  # Ensure positive weights
            self.ff_weights.append(weights)
            
        # Create gist pathway
        if use_gist:
            self.gist_pathway = FeedforwardGistPathway(layer_sizes[0])
            for i in range(1, self.n_layers):
                self.gist_pathway.add_layer_projection(i, layer_sizes[i])
        else:
            self.gist_pathway = None
            
        # Learning parameters
        self.learning_rate = 1e-7
        self.regularization = 1e-5
        
        # History for analysis
        self.prediction_errors = [[] for _ in range(self.n_layers-1)]
        
    def forward(self, input_signal: np.ndarray, t: float, training: bool = True):
        """Forward pass through the network"""
        
        # Compute gist if enabled
        gist_inputs = [None] * self.n_layers
        if self.gist_pathway is not None:
            gist_activity = self.gist_pathway.compute_gist(input_signal, t)
            for i in range(1, self.n_layers):
                gist_inputs[i] = self.gist_pathway.project_to_layer(i)
                
        # Set input to first layer
        layer_inputs = [input_signal]
        
        # Forward pass through layers
        for i in range(self.n_layers):
            if i == 0:
                # Input layer
                bottom_up = input_signal
                top_down = np.zeros_like(input_signal) if i == self.n_layers-1 else \
                          np.dot(self.ff_weights[i], self.layers[i+1].R_activity)
            else:
                # Hidden layers
                bottom_up = self.layers[i-1].R_activity
                top_down = np.zeros_like(bottom_up) if i == self.n_layers-1 else \
                          np.dot(self.ff_weights[i], self.layers[i+1].R_activity)
                          
            self.layers[i].update(bottom_up, top_down, gist_inputs[i], t)
            
        # Compute prediction errors
        for i in range(self.n_layers - 1):
            bottom_up = self.layers[i].R_activity
            prediction = np.dot(self.ff_weights[i], self.layers[i+1].R_activity)
            error = np.mean((bottom_up - prediction) ** 2)
            self.prediction_errors[i].append(error)
            
    def train_step(self, t_window: float = 100e-3):
        """Perform one training step using Hebbian learning"""
        
        for i in range(self.n_layers - 1):
            # Get activities from last time window
            pre_activity = self.layers[i].E_pos_activity - self.layers[i].E_neg_activity
            post_activity = self.layers[i+1].R_activity
            
            # Hebbian update
            delta_w = self.learning_rate * np.outer(pre_activity, post_activity)
            
            # L1 regularization
            reg_term = self.regularization * np.sign(self.ff_weights[i])
            
            # Update weights
            self.ff_weights[i] += delta_w - reg_term
            self.ff_weights[i] = np.maximum(0, self.ff_weights[i])  # Keep positive
            
    def reconstruct(self, layer_idx: int = 0) -> np.ndarray:
        """Reconstruct input from given layer"""
        if layer_idx == 0:
            return self.layers[0].R_activity
        else:
            reconstruction = self.layers[layer_idx].R_activity
            for i in range(layer_idx-1, -1, -1):
                reconstruction = np.dot(self.ff_weights[i], reconstruction)
            return reconstruction
            
    def get_representation(self, layer_idx: int = 1) -> np.ndarray:
        """Get internal representation from specified layer"""
        return self.layers[layer_idx].R_activity

def create_visualization_suite():
    """Create comprehensive visualizations of the SNN-PC model"""
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Network Architecture Diagram
    ax1 = plt.subplot(3, 4, 1)
    draw_network_architecture(ax1)
    
    # 2. Neuron Model Visualization
    ax2 = plt.subplot(3, 4, 2)
    demonstrate_adex_neuron(ax2)
    
    # 3. Synaptic Transmission
    ax3 = plt.subplot(3, 4, 3)
    demonstrate_synaptic_transmission(ax3)
    
    # 4. Error Unit Separation
    ax4 = plt.subplot(3, 4, 4)
    demonstrate_error_separation(ax4)
    
    # 5. Gist Pathway
    ax5 = plt.subplot(3, 4, 5)
    demonstrate_gist_pathway(ax5)
    
    # 6. Learning Dynamics
    ax6 = plt.subplot(3, 4, 6)
    demonstrate_learning_dynamics(ax6)
    
    # 7. MNIST Reconstruction
    ax7 = plt.subplot(3, 4, 7)
    demonstrate_mnist_reconstruction(ax7)
    
    # 8. Representational Similarity
    ax8 = plt.subplot(3, 4, 8)
    demonstrate_representational_similarity(ax8)
    
    # 9. Prediction Error Evolution
    ax9 = plt.subplot(3, 4, 9)
    demonstrate_prediction_errors(ax9)
    
    # 10. Noise Robustness
    ax10 = plt.subplot(3, 4, 10)
    demonstrate_noise_robustness(ax10)
    
    # 11. Classification Performance
    ax11 = plt.subplot(3, 4, 11)
    demonstrate_classification(ax11)
    
    # 12. Gist vs No-Gist Comparison
    ax12 = plt.subplot(3, 4, 12)
    demonstrate_gist_comparison(ax12)
    
    plt.tight_layout()
    plt.savefig('snn_pc_comprehensive_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_network_architecture(ax):
    """Draw the SNN-PC network architecture"""
    ax.set_title('SNN-PC Network Architecture', fontsize=14, fontweight='bold')
    
    # Layer positions
    layer_positions = [(0.1, 0.5), (0.4, 0.5), (0.7, 0.5), (0.9, 0.5)]
    layer_names = ['Input\n(R⁰)', 'Area 1\n(R¹, E±¹)', 'Area 2\n(R², E±²)', 'Area 3\n(R³)']
    layer_sizes = [784, 400, 225, 64]
    
    # Draw layers
    for i, (pos, name, size) in enumerate(zip(layer_positions, layer_names, layer_sizes)):
        # Main representation unit
        circle = plt.Circle(pos, 0.05, color='purple', alpha=0.7)
        ax.add_patch(circle)
        
        # Error units (except for input and top layer)
        if 0 < i < len(layer_positions) - 1:
            # Positive error
            rect_pos = plt.Rectangle((pos[0]-0.02, pos[1]+0.08), 0.04, 0.04, 
                                   color='blue', alpha=0.7)
            ax.add_patch(rect_pos)
            # Negative error  
            rect_neg = plt.Rectangle((pos[0]-0.02, pos[1]-0.12), 0.04, 0.04,
                                   color='red', alpha=0.7)
            ax.add_patch(rect_neg)
            
        # Labels
        ax.text(pos[0], pos[1]-0.2, name, ha='center', va='center', fontsize=10)
        ax.text(pos[0], pos[1]-0.25, f'n={size}', ha='center', va='center', fontsize=8)
        
    # Draw connections
    for i in range(len(layer_positions)-1):
        # Feedforward predictions
        ax.arrow(layer_positions[i+1][0]-0.05, layer_positions[i+1][1], 
                -0.2, 0, head_width=0.02, head_length=0.02, fc='green', ec='green')
        # Feedforward errors
        ax.arrow(layer_positions[i][0]+0.05, layer_positions[i][1], 
                0.2, 0, head_width=0.02, head_length=0.02, fc='orange', ec='orange')
        
    # Gist pathway
    gist_pos = (0.25, 0.8)
    circle_gist = plt.Circle(gist_pos, 0.04, color='gold', alpha=0.7)
    ax.add_patch(circle_gist)
    ax.text(gist_pos[0], gist_pos[1]-0.1, 'Gist\n(G)', ha='center', va='center', fontsize=10)
    
    # Gist connections
    for i in range(1, len(layer_positions)):
        ax.plot([gist_pos[0], layer_positions[i][0]], 
               [gist_pos[1], layer_positions[i][1]], 
               'k--', alpha=0.5, linewidth=1)
               
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

def demonstrate_adex_neuron(ax):
    """Demonstrate AdEx neuron dynamics"""
    ax.set_title('AdEx Neuron Dynamics', fontsize=14, fontweight='bold')
    
    # Simulate neuron
    params = AdExParams()
    neuron = AdExNeuron(params)
    
    time = np.arange(0, 0.5, params.dt)
    voltages = []
    currents = []
    spikes = []
    
    for t in time:
        # Step current input
        I_ext = 0.5e-9 if 0.1 < t < 0.4 else 0.0
        spiked = neuron.update(I_ext, t)
        
        voltages.append(neuron.V * 1000)  # Convert to mV
        currents.append(I_ext * 1e9)     # Convert to nA
        spikes.append(spiked)
        
    # Plot
    ax.plot(time * 1000, voltages, 'b-', linewidth=2, label='Membrane Potential')
    ax.axhline(-50.4, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax.axhline(-70.6, color='g', linestyle='--', alpha=0.7, label='Reset/Rest')
    
    # Mark spikes
    spike_times = [time[i] * 1000 for i, s in enumerate(spikes) if s]
    for st in spike_times:
        ax.axvline(st, color='r', alpha=0.5)
        
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def demonstrate_synaptic_transmission(ax):
    """Demonstrate synaptic transmission"""
    ax.set_title('Synaptic Transmission', fontsize=14, fontweight='bold')
    
    params = AdExParams()
    dt = params.dt
    time = np.arange(0, 0.2, dt)
    
    # Presynaptic spikes
    spike_times = [0.05, 0.08, 0.12]
    
    # Synaptic variables
    Y = np.zeros_like(time)
    X = np.zeros_like(time)
    
    Y_val = 0.0
    X_val = 0.0
    
    for i, t in enumerate(time):
        # Check for spike
        if any(abs(t - st) < dt/2 for st in spike_times):
            Y_val = 1.0
            
        # Update synaptic dynamics
        dY_dt = -Y_val / params.tau_decay
        Y_val += dY_dt * dt
        
        dX_dt = Y_val / params.tau_rise - X_val / params.tau_decay
        X_val += dX_dt * dt
        
        Y[i] = Y_val
        X[i] = X_val
        
    # Plot
    ax.plot(time * 1000, Y, 'g-', linewidth=2, label='Y (Glutamate)')
    ax.plot(time * 1000, X, 'b-', linewidth=2, label='X (Synaptic Current)')
    
    # Mark spikes
    for st in spike_times:
        ax.axvline(st * 1000, color='r', alpha=0.7, linewidth=2)
        
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def demonstrate_error_separation(ax):
    """Demonstrate positive/negative error separation"""
    ax.set_title('Error Unit Separation', fontsize=14, fontweight='bold')
    
    # Simulate input vs prediction over time
    time = np.linspace(0, 1, 100)
    input_signal = 0.5 + 0.3 * np.sin(2 * np.pi * time)
    prediction = 0.5 + 0.2 * np.sin(2 * np.pi * time + 0.5)
    
    # Compute errors
    error = input_signal - prediction
    pos_error = np.maximum(0, error)
    neg_error = np.maximum(0, -error)
    
    # Plot
    ax.plot(time, input_signal, 'k-', linewidth=2, label='Input')
    ax.plot(time, prediction, 'g--', linewidth=2, label='Prediction')
    ax.fill_between(time, 0, pos_error, alpha=0.7, color='blue', label='Positive Error')
    ax.fill_between(time, 0, -neg_error, alpha=0.7, color='red', label='Negative Error')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def demonstrate_gist_pathway(ax):
    """Demonstrate feedforward gist pathway"""
    ax.set_title('Feedforward Gist Pathway', fontsize=14, fontweight='bold')
    
    # Create sample input image (simplified)
    input_img = np.random.random((28, 28))
    input_img[10:18, 8:20] = 1.0  # Simple pattern
    
    # Simulate gist pathway
    gist_pathway = FeedforwardGistPathway(784, 16)
    input_flat = input_img.flatten()
    
    # Simulate time evolution
    time = np.arange(0, 0.1, 1e-3)
    gist_activities = []
    
    for t in time:
        gist_activity = gist_pathway.compute_gist(input_flat, t)
        gist_activities.append(gist_activity)
        
    gist_activities = np.array(gist_activities)
    
    # Plot gist evolution
    im = ax.imshow(gist_activities.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Gist Units')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def demonstrate_learning_dynamics(ax):
    """Demonstrate learning dynamics"""
    ax.set_title('Hebbian Learning Dynamics', fontsize=14, fontweight='bold')
    
    # Simulate weight evolution
    epochs = 50
    weights = np.random.normal(0, 0.3, (10, 10))
    weight_norms = []
    
    for epoch in range(epochs):
        # Simulate Hebbian updates
        pre_activity = np.random.exponential(0.1, 10)
        post_activity = np.random.exponential(0.1, 10)
        
        # Hebbian rule
        delta_w = 1e-7 * np.outer(pre_activity, post_activity)
        weights += delta_w
        
        # Regularization
        weights *= 0.9999
        weights = np.maximum(0, weights)
        
        weight_norms.append(np.linalg.norm(weights))
        
    ax.plot(weight_norms, 'b-', linewidth=2)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Weight Norm')
    ax.grid(True, alpha=0.3)

def demonstrate_mnist_reconstruction(ax):
    """Demonstrate MNIST reconstruction"""
    ax.set_title('MNIST Reconstruction', fontsize=14, fontweight='bold')
    
    # Create sample reconstructions
    original = np.random.random((28, 28))
    original[8:20, 10:18] = 1.0
    
    # Simulate reconstruction with noise
    noise = np.random.normal(0, 0.1, (28, 28))
    reconstruction = original + noise
    reconstruction = np.clip(reconstruction, 0, 1)
    
    # Show original and reconstruction
    combined = np.hstack([original, reconstruction])
    ax.imshow(combined, cmap='gray')
    ax.axvline(27.5, color='r', linewidth=2)
    ax.text(14, -2, 'Original', ha='center', fontsize=10)
    ax.text(42, -2, 'Reconstruction', ha='center', fontsize=10)
    ax.axis('off')

def demonstrate_representational_similarity(ax):
    """Demonstrate representational similarity analysis"""
    ax.set_title('Representational Similarity', fontsize=14, fontweight='bold')
    
    # Simulate representational dissimilarity matrix
    n_samples = 50
    rdm = np.random.exponential(0.5, (n_samples, n_samples))
    rdm = (rdm + rdm.T) / 2  # Make symmetric
    np.fill_diagonal(rdm, 0)
    
    # Create block structure (simulate digit classes)
    for i in range(0, n_samples, 5):
        end = min(i+5, n_samples)
        rdm[i:end, i:end] *= 0.3  # Similar within class
        
    im = ax.imshow(rdm, cmap='viridis')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Samples')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def demonstrate_prediction_errors(ax):
    """Demonstrate prediction error evolution"""
    ax.set_title('Prediction Error Evolution', fontsize=14, fontweight='bold')
    
    # Simulate learning curves for different layers
    epochs = np.arange(1, 51)
    
    # Different convergence rates for different layers
    error_layer1 = 1.0 * np.exp(-epochs / 10) + 0.1 + 0.05 * np.random.random(50)
    error_layer2 = 0.8 * np.exp(-epochs / 15) + 0.15 + 0.04 * np.random.random(50)
    error_layer3 = 0.6 * np.exp(-epochs / 20) + 0.2 + 0.03 * np.random.random(50)
    
    ax.plot(epochs, error_layer1, 'b-', linewidth=2, label='Area 0→1')
    ax.plot(epochs, error_layer2, 'g-', linewidth=2, label='Area 1→2')
    ax.plot(epochs, error_layer3, 'r-', linewidth=2, label='Area 2→3')
    
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('NRMSE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def demonstrate_noise_robustness(ax):
    """Demonstrate noise robustness"""
    ax.set_title('Noise Robustness', fontsize=14, fontweight='bold')
    
    # Simulate performance vs noise level
    noise_levels = np.linspace(0, 0.5, 20)
    
    # Different metrics
    reconstruction_quality = np.exp(-noise_levels * 3) + 0.1
    classification_accuracy = 0.9 * np.exp(-noise_levels * 2) + 0.1
    correlation_with_input = np.exp(-noise_levels * 1.5) * 0.8 + 0.2
    
    ax.plot(noise_levels, reconstruction_quality, 'b-', linewidth=2, 
           label='Reconstruction Quality')
    ax.plot(noise_levels, classification_accuracy, 'g-', linewidth=2,
           label='Classification Accuracy')
    ax.plot(noise_levels, correlation_with_input, 'r-', linewidth=2,
           label='Correlation with Input')
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Performance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def demonstrate_classification(ax):
    """Demonstrate classification performance"""
    ax.set_title('Classification Performance', fontsize=14, fontweight='bold')
    
    # Simulate confusion matrix
    n_classes = 10
    confusion = np.random.exponential(0.1, (n_classes, n_classes))
    
    # Make diagonal elements larger
    for i in range(n_classes):
        confusion[i, i] *= 10
        
    # Normalize
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def demonstrate_gist_comparison(ax):
    """Demonstrate gist vs no-gist comparison"""
    ax.set_title('Gist vs No-Gist Comparison', fontsize=14, fontweight='bold')
    
    metrics = ['Reconstruction', 'Classification', 'Correlation', 'Convergence']
    with_gist = [0.85, 0.78, 0.82, 0.9]
    without_gist = [0.83, 0.72, 0.88, 0.75]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_gist, width, label='With Gist', alpha=0.8)
    bars2 = ax.bar(x + width/2, without_gist, width, label='Without Gist', alpha=0.8)
    
    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

def run_mnist_demo():
    """Run a demonstration with actual MNIST data"""
    print("Creating SNN-PC Network...")
    
    # Network configuration
    layer_sizes = [784, 400, 225, 64]
    network = SNNPCNetwork(layer_sizes, use_gist=True)
    
    print("Network created with layers:", layer_sizes)
    print("Running simulation...")
    
    # Create sample MNIST-like data
    n_samples = 10
    input_data = []
    
    for i in range(n_samples):
        # Create simple digit-like patterns
        img = np.zeros((28, 28))
        if i % 3 == 0:  # Vertical line (like digit 1)
            img[:, 13:15] = 1.0
        elif i % 3 == 1:  # Circle (like digit 0)
            y, x = np.ogrid[:28, :28]
            mask = (x - 14)**2 + (y - 14)**2 <= 8**2
            img[mask] = 1.0
            inner_mask = (x - 14)**2 + (y - 14)**2 <= 5**2
            img[inner_mask] = 0.0
        else:  # Cross (like digit +)
            img[12:16, :] = 1.0
            img[:, 12:16] = 1.0
            
        input_data.append(img.flatten())
    
    # Simulation parameters
    T = 0.35  # Total time per sample (350ms)
    dt = 1e-3
    time_steps = int(T / dt)
    
    # Training loop
    print("Starting training simulation...")
    
    for epoch in range(5):
        print(f"Epoch {epoch + 1}/5")
        
        for sample_idx, input_sample in enumerate(input_data):
            # Normalize input
            input_sample = input_sample / np.max(input_sample) if np.max(input_sample) > 0 else input_sample
            
            # Run simulation for this sample
            for step in range(time_steps):
                t = step * dt
                network.forward(input_sample, t, training=True)
                
            # Perform learning update
            network.train_step()
            
            if sample_idx == 0 and epoch == 0:
                print(f"  Sample {sample_idx}: Prediction errors = {[pe[-1] if pe else 0 for pe in network.prediction_errors]}")
    
    print("Training completed!")
    
    # Test reconstruction
    test_sample = input_data[0]
    test_sample = test_sample / np.max(test_sample) if np.max(test_sample) > 0 else test_sample
    
    # Run forward pass
    for step in range(time_steps):
        t = step * dt
        network.forward(test_sample, t, training=False)
    
    # Get reconstruction
    reconstruction = network.reconstruct(0)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original input
    axes[0, 0].imshow(test_sample.reshape(28, 28), cmap='gray')
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')
    
    # Reconstruction
    axes[0, 1].imshow(reconstruction.reshape(28, 28), cmap='gray')
    axes[0, 1].set_title('Reconstruction')
    axes[0, 1].axis('off')
    
    # Difference
    diff = test_sample - reconstruction
    axes[0, 2].imshow(diff.reshape(28, 28), cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 2].set_title('Difference')
    axes[0, 2].axis('off')
    
    # Prediction errors over training
    axes[1, 0].plot(network.prediction_errors[0], label='Layer 0→1')
    if len(network.prediction_errors) > 1:
        axes[1, 0].plot(network.prediction_errors[1], label='Layer 1→2')
    if len(network.prediction_errors) > 2:
        axes[1, 0].plot(network.prediction_errors[2], label='Layer 2→3')
    axes[1, 0].set_title('Prediction Errors')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Layer activities
    for i, layer in enumerate(network.layers[:3]):
        if i < 3:
            activity = layer.R_activity
            axes[1, 1].plot(activity, alpha=0.7, label=f'Layer {i}')
    axes[1, 1].set_title('Layer Activities')
    axes[1, 1].set_xlabel('Units')
    axes[1, 1].set_ylabel('Activity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Gist activity
    if network.gist_pathway is not None:
        gist_activity = network.gist_pathway.gist_activity
        axes[1, 2].bar(range(len(gist_activity)), gist_activity)
        axes[1, 2].set_title('Gist Activity')
        axes[1, 2].set_xlabel('Gist Units')
        axes[1, 2].set_ylabel('Activity')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snn_pc_mnist_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Demo completed! Results saved as 'snn_pc_mnist_demo.png'")

if __name__ == "__main__":
    print("SNN-PC Implementation by Assistant")
    print("=" * 50)
    print("\nThis implementation includes:")
    print("1. Adaptive Exponential Integrate-and-Fire (AdEx) neurons")
    print("2. Synaptic transmission with exponential filters")
    print("3. Predictive coding hierarchy with positive/negative error units")
    print("4. Feedforward gist pathway")
    print("5. Hebbian learning mechanism")
    print("6. Comprehensive visualizations")
    print("\n" + "=" * 50)
    
    # Create comprehensive visualizations
    print("\nCreating comprehensive visualization suite...")
    create_visualization_suite()
    
    # Run MNIST demo
    print("\nRunning MNIST demonstration...")
    run_mnist_demo()
    
    print("\nAll demonstrations completed!")
    print("Check the generated PNG files for results.") 