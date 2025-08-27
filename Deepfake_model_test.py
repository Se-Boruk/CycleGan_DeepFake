import torch
import numpy as np
import Training_assets
import Architectures
import json
###################################################################
# ( 1 ) Hardware setup
###################################################################
print("\nSearching for cuda device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
###################################################################
# ( 2 ) Loading data
###################################################################
print("\nLoading data...")
Source_data = np.load("Processed_data/Source_Data.npy")
Target_data = np.load("Processed_data/Target_Data.npy")  
_, img_h, img_w, in_channels = Source_data.shape


#Remove heatmap (landmarks)
Source_data = np.delete(Source_data, 4, axis=-1)
Target_data = np.delete(Target_data, 4, axis=-1)



###################################################################
# ( 3 ) Loading models (pretrained)
###################################################################
model_path = "models/cyclegan_60.pth"
#Hyperparemetrs
print("Loading image config...")
with open('models/model_input_shape.json') as f:
    input_config = json.load(f)

input_channels = input_config['in_channels']
output_channels = input_config['out_channels']
base_filters = input_config['base_filters']
n_residual_blocks = input_config['residual_blocks']

assert img_h == input_config['height'], "Height of data should be the same size as height input of model"
assert img_w == input_config['width'], "Width of data should be the same size as width input of model"

#Models
G_AB = Architectures.Generator(
    input_channels=input_channels,
    output_channels=output_channels,
    n_residual=n_residual_blocks,
    base_filters=base_filters
)

G_BA = Architectures.Generator(
    input_channels=input_channels,
    output_channels=output_channels,
    n_residual=n_residual_blocks,
    base_filters=base_filters
)

#Load weights from checkpoint and move to device
checkpoint = torch.load(model_path, map_location=device)
G_AB.load_state_dict(checkpoint['G_AB'])
G_BA.load_state_dict(checkpoint['G_BA'])

G_AB.to(device)
G_BA.to(device)


###################################################################
# ( 4 ) Preparing tensor format for img
###################################################################
idx = 300
#Comaprision image
source_tensor = Source_data[idx][:, :, :input_channels]
source_tensor = torch.from_numpy(source_tensor).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

target_tensor = Target_data[idx][:, :, :input_channels]
target_tensor = torch.from_numpy(target_tensor).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

source_tensor = source_tensor.to(device)
target_tensor = target_tensor.to(device)


###################################################################
# ( 5 ) Showing model stages on sample 
###################################################################

#(Blend may look worse than final due to difference in video/tensor processing)
print("\nTesting face replacement with comparison...")
Training_assets.visualize_deepfakes(source_sample_tensor = source_tensor,
                                    target_sample_tensor = target_tensor,
                                    G_AB=G_AB,
                                    G_BA=G_BA,
                                    device=device,
                                    show_plot=True,
                                    save_plot = True,
                                    folder = "Model_results"
                                )
