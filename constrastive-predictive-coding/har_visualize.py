import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Load the saved numpy file
file_path = '/home/paperspace/tinyfaces-cpc/processed_data/Walking While Reading Book.npy'
data = np.load(file_path, allow_pickle=True)

# Function to save frames from a sequence as an image file
def save_frame_sequence(frame_sequence, sequence_type, output_dir):
    num_frames = len(frame_sequence)
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    for i in range(num_frames):
        axes[i].imshow(frame_sequence[i])
        axes[i].axis('off')
    output_path = os.path.join(output_dir, f'sequence_{sequence_type}.png')
    plt.savefig(output_path)
    plt.close(fig)

# Output directory for the images
output_dir = '/home/paperspace/tinyfaces-cpc/visualizations'
os.makedirs(output_dir, exist_ok=True)

# Select one positive and one negative sample
positive_sample = data[0]
negative_sample = data[1].copy()
np.random.shuffle(negative_sample[4:])  # Shuffle the output frames to create a negative sample

# Save the positive sample
print("Saving positive sample")
save_frame_sequence(positive_sample, 'positive', output_dir)

# Save the negative sample
print("Saving negative sample")
save_frame_sequence(negative_sample, 'negative', output_dir)
