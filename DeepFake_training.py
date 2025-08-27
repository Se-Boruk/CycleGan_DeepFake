import torch
import numpy as np
import Training_assets
import Architectures
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os
import gc
import Functions
import torch.nn.functional as F
from torch.optim import AdamW
from pytorch_msssim import ms_ssim
import lpips
import torch.nn as nn

###################################################################
# ( 1 ) Hardware setup
###################################################################

print("\nSearching for cuda device...")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print available GPUs
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    

###################################################################
# ( 2 ) Loading data
###################################################################
print("\nLoading data...")   

Source_directory = "Processed_data/Source_Data.npy"
Target_directory = "Processed_data/Target_Data.npy"

Source_data = np.load(Source_directory)
Target_data = np.load(Target_directory)

#remove landmarks (heatmpap) from data
Source_data = np.delete(Source_data, 4, axis=-1)
Target_data = np.delete(Target_data, 4, axis=-1)

print("Done!")
###################################################################
# ( 3 ) Setting parameters
###################################################################
_, img_h, img_w, in_channels = Source_data.shape
reference_img_idx = 478

#Parameters (model training)
epochs = 150
batch_size = 6
disc_update_interval = 1
#Losses (main)
lambda_cycle = 11
lambda_identity = 1.5

#loss partials
lambda_ssim = 1
lambda_l1 = 1
lambda_lpips = 0.3

#Architecture parameters
input_channels = 4
output_channels = 4
base_filters = 64
n_residual_blocks = 9

#Saving model input params
Functions.save_model_input_shape(input_h = img_h,
                                 input_w = img_w,
                                 input_channels = input_channels,
                                 output_channels = output_channels,
                                 base_filters = base_filters,
                                 residual_blocks = n_residual_blocks,
                                 save_dir = 'models',
                                 filename = 'model_input_shape.json')


###################################################################
# ( 4 ) Model creation, dataloader preparation
###################################################################
print("\nPreparing models...")
#Generators
G_AB = Architectures.Generator(
    input_channels=input_channels,
    output_channels=output_channels,
    n_residual=n_residual_blocks,
    base_filters=base_filters
).to(device)

G_BA = Architectures.Generator(
    input_channels=input_channels,
    output_channels=output_channels,
    n_residual=n_residual_blocks,
    base_filters=base_filters
).to(device)
#Discriminators
D_A = Architectures.Discriminator(
    input_channels=input_channels,
    base_filters=base_filters
).to(device)

D_B = Architectures.Discriminator(
    input_channels=input_channels,
    base_filters=base_filters
).to(device)

#Optimizers for models
opt_G = AdamW(
    list(G_AB.parameters()) + list(G_BA.parameters()),
    lr=2e-4,
    betas=(0.5, 0.999),
    weight_decay=1e-5 
)

opt_D = torch.optim.Adam(
    list(D_A.parameters()) + list(D_B.parameters()),
    lr=2e-4,
    betas=(0.5, 0.999),
    weight_decay=0.0
)
print("Done!")

#Creating dataloader and saving data format
dataset = Training_assets.EpochBalancedFaceDataset(Source_data, Target_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

###################################################################
# ( 5 ) Showing / saving architecture scheme [ Optional ]
###################################################################
source_shape = Source_data[0,:,:,:input_channels].shape
g_input = torch.randn(source_shape).permute(2,0,1)


Training_assets.Show_architecture(model = G_AB,
                                  input_tensor = g_input,
                                  save_png = True,
                                  folder = "Architecture_pngs",
                                  name = "Generator_1"
                                  )

d_input = G_AB(g_input.unsqueeze(0)).squeeze(0)


Training_assets.Show_architecture(model = D_B,
                                  input_tensor = d_input,
                                  save_png = True,
                                  folder = "Architecture_pngs",
                                  name = "Discriminator_1"
                                  )


del g_input, d_input
torch.cuda.empty_cache()

###################################################################
# ( 6 ) Preparation of input tensors (for reference in training)
###################################################################
print("\nPreparing reference images...")
#Comaprision image (in tensor format)
source_tensor = Source_data[reference_img_idx][:, :, :input_channels]
source_tensor = torch.from_numpy(source_tensor).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

target_tensor = Target_data[reference_img_idx][:, :, :input_channels]
target_tensor = torch.from_numpy(target_tensor).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
#Moving into gpu
source_sample_tensor_gpu = source_tensor.to(device).to(torch.float32)
target_sample_tensor_gpu = target_tensor.to(device).to(torch.float32)
print("Done!")

###################################################################
# ( 7 ) Final model moving just in case
###################################################################
print("\nMoving models to GPU...")
G_AB = G_AB.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)
D_B = D_B.to(device)
print("Done!")


###################################################################
# ( 8 ) Training loop
###################################################################

#Short mask rename for clarity (mask instead of l1)
l1 = torch.nn.L1Loss()
def mask_loss(pred, target):
    return l1(pred, target)

bce_loss = nn.BCEWithLogitsLoss()

#Preparing perceptual lpips loss
perceptual = lpips.LPIPS(net='vgg').to(device)

#Scaler for halfprecision
scaler = torch.amp.GradScaler()


#===================== Training loop ========================
start_epoch = 0
"""
###
model_path = "models/cyclegan_90.pth"

checkpoint = torch.load(model_path, map_location=device)
G_AB.load_state_dict(checkpoint['G_AB'])
G_BA.load_state_dict(checkpoint['G_BA'])

D_A.load_state_dict(checkpoint['D_A'])
D_B.load_state_dict(checkpoint['D_B'])

opt_G.load_state_dict(checkpoint['opt_G'])
opt_D.load_state_dict(checkpoint['opt_D'])

scaler.load_state_dict(checkpoint['scaler'])

start_epoch = 91
###
"""

for epoch in range(start_epoch, epochs):
    #Train mode for models
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    #Loss across epoch
    total_loss_G = 0.0
    total_loss_D_A = 0.0
    total_loss_D_B = 0.0
    num_batches = len(loader)

    #Prefetching (asynchronous batch loading for performance improvement)
    dataset.on_epoch_start()
    prefetcher = Training_assets.DataPrefetcherWithAugmentation(loader, device, Training_assets.color_augmentation)
    batch = prefetcher.next()

    pbar = tqdm(range(num_batches), desc=f"Epoch [{epoch}/{epochs-1}]", leave=False)

    for _ in pbar:
        batch = prefetcher.next()
        if batch is None or batch[0] is None:
            break
        _, _, real_A_masked, real_B_masked = batch

        #=== Generator update ===
        opt_G.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            #Fake faces
            fake_B_masked = G_AB(real_A_masked)
            fake_A_masked = G_BA(real_B_masked)
            #Reconstructed faces
            rec_A_masked = G_BA(fake_B_masked)
            rec_B_masked = G_AB(fake_A_masked)

            # Calculating Discriminator output for adversarial loss
            D_A_real_logits = D_A(real_A_masked)
            D_A_fake_logits = D_A(fake_A_masked)
            D_B_real_logits = D_B(real_B_masked)
            D_B_fake_logits = D_B(fake_B_masked)
            
            # ----- Adversarial loss -----
            # From logits
            valid_logits_B = torch.ones_like(D_B_fake_logits)
            loss_GAN_AB = bce_loss(D_B_fake_logits, valid_logits_B)
            
            valid_logits_A = torch.ones_like(D_A_fake_logits)
            loss_GAN_BA = bce_loss(D_A_fake_logits, valid_logits_A)
            
            
            
            #Identity loss (simple l1 loss to compare 2 images absolute difference)
            id_A_gen = G_BA(real_A_masked)
            id_B_gen = G_AB(real_B_masked)
            
            identity_loss = (
                F.l1_loss(id_A_gen, real_A_masked) +
                F.l1_loss(id_B_gen, real_B_masked)
            )

            #Cycle loss (Partial of l1 loss ms-ssim loss and lpips loss)
            cycle_loss_ssim = (1 - ms_ssim(rec_A_masked, real_A_masked) + 1 - ms_ssim(rec_B_masked, real_B_masked)).mean()
            cycle_loss_l1 = (F.l1_loss(rec_A_masked, real_A_masked) + F.l1_loss(rec_B_masked, real_B_masked)).mean()
            cycle_loss_lpips = (perceptual(rec_A_masked[:, :3], real_A_masked[:, :3]).view(-1).mean() + perceptual(rec_B_masked[:, :3], real_B_masked[:, :3]).view(-1).mean())
            
            cycle_loss = (
                lambda_ssim * cycle_loss_ssim+
                lambda_l1 * cycle_loss_l1 +
                lambda_lpips * cycle_loss_lpips
            )
            
            #Mask preservation loss
            loss_mask = (
                mask_loss(fake_B_masked[:, 3:], real_B_masked[:, 3:]) +
                mask_loss(fake_A_masked[:, 3:], real_A_masked[:, 3:]) +
                mask_loss(rec_A_masked[:, 3:], real_A_masked[:, 3:]) +
                mask_loss(rec_B_masked[:, 3:], real_B_masked[:, 3:])
            )
            
            #Tv loss (for smoothness and overall "no artifacts" promotion )
            loss_tv = (
                Training_assets.tv_loss(fake_B_masked[:, :3]) +
                Training_assets.tv_loss(fake_A_masked[:, :3]) +
                Training_assets.tv_loss(rec_B_masked[:, :3]) +
                Training_assets.tv_loss(rec_A_masked[:, :3])
            )
            
            #Final calculation
            loss_G = (
                (loss_GAN_AB + loss_GAN_BA) +
                lambda_cycle * cycle_loss +
                lambda_identity * identity_loss +
                0.001 * loss_tv +
                0.1 * loss_mask
            )
        
        #Gradients update with exception for explosion (funny addon, model would collapse anyway though)
        if torch.isfinite(loss_G):
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()
        else:
            print(f"[Warning] Skipped Generator step due to non-finite loss: {loss_G.item()}")

        # === Discriminator update ===
        if epoch % disc_update_interval == 0:
            opt_D.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Get logits and predictions from both discriminators
                D_A_real_logits = D_A(real_A_masked.detach())
                D_A_fake_logits = D_A(fake_A_masked.detach())
                D_B_real_logits = D_B(real_B_masked.detach())
                D_B_fake_logits = D_B(fake_B_masked.detach())
            
                # Labels
                real_labels_logits = torch.ones_like(D_A_real_logits)
                fake_labels_logits = torch.zeros_like(D_A_fake_logits)

            
                # ----- D_A Loss -----
                # From logits
                loss_D_A_real_logits = bce_loss(D_A_real_logits, real_labels_logits)
                loss_D_A_fake_logits = bce_loss(D_A_fake_logits, fake_labels_logits)
                
                loss_D_A = 0.5 * (loss_D_A_real_logits + loss_D_A_fake_logits)
            

                # ----- D_B Loss -----
                # From logits
                loss_D_B_real_logits = bce_loss(D_B_real_logits, torch.ones_like(D_B_real_logits))
                loss_D_B_fake_logits = bce_loss(D_B_fake_logits, torch.zeros_like(D_B_fake_logits))
                
                loss_D_B = 0.5 * (loss_D_B_real_logits + loss_D_B_fake_logits)
            

                # ----- Total Loss -----
                loss_D = loss_D_A + loss_D_B
                
            #Gradients update with exception for explosion (funny addon, model would collapse anyway though)
            if torch.isfinite(loss_D):
                scaler.scale(loss_D).backward()
                scaler.step(opt_D)
                scaler.update()
            else:
                print(f"[Warning] Skipped Discriminator step due to non-finite loss: {loss_D.item()}")
        
        #Batch loss update dynamically
        total_loss_G += loss_G.item()
        total_loss_D_A += loss_D_A.item()
        total_loss_D_B += loss_D_B.item()

        pbar.set_postfix({
            "G_loss": f"{loss_G.item():.4f}",
            "D_A": f"{loss_D_A.item():.4f}",
            "D_B": f"{loss_D_B.item():.4f}"
        })

    #===Epoch end===
    avg_loss_G = total_loss_G / num_batches
    avg_loss_D_A = total_loss_D_A / num_batches
    avg_loss_D_B = total_loss_D_B / num_batches

    print(f"\nEpoch [{epoch}/{epochs-1}] | G: {avg_loss_G:.4f} | D_A: {avg_loss_D_A:.4f} | D_B: {avg_loss_D_B:.4f}")
    print("=====================================================================")

    #Save logs
    with open("training_log.csv", mode='a', newline='') as f:
        csv.writer(f).writerow([epoch, avg_loss_G, avg_loss_D_A, avg_loss_D_B])

    os.makedirs("models", exist_ok=True)
    
    #If epoch is divisible by 2 save model, also save preview from every epoch
    if (epoch) % 2 == 0:
        torch.save({
            'G_AB': G_AB.state_dict(),
            'G_BA': G_BA.state_dict(),
            'D_A': D_A.state_dict(),
            'D_B': D_B.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'scaler': scaler.state_dict()
        }, f"models/cyclegan_{epoch}.pth")

    if (epoch) % 1 == 0:
        Training_assets.visualize_deepfakes(
            source_sample_tensor=source_sample_tensor_gpu,
            target_sample_tensor=target_sample_tensor_gpu,
            G_AB=G_AB,
            G_BA=G_BA,
            device=device,
            show_plot=False,
            save_plot=True,
            folder="Model_results",
            name=f"Epoch_{epoch}.png"
        )

    #Clean garbage from memory (PyTorch I love you (Tensorflow GPU memory issues looking at you))
    gc.collect()
    torch.cuda.empty_cache()





