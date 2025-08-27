# Training_assets.py
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

import os
from torchsummary import summary
from torchviz import make_dot
import gc



def tv_loss(img):
    """
    Penalizes high-frequency noise/artifacts.
    """
    loss_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return loss_h + loss_w


#Show architecture of given model
def Show_architecture(model, input_tensor, save_png=False, folder="Model_graphs", name="Model"):
    print("Preparing model...")
    # Wrapped model with preset parameter
    
    model.to("cpu")
    model.eval()
    input_tensor = input_tensor.to("cpu")  # Ensure correct tensor assignment
    
    input_shape = input_tensor.shape

    summary(model, input_shape, device="cpu")  # Print summary


    if save_png:
        # Ensure folder exists
        os.makedirs(folder, exist_ok=True)
        
        # Output tensor of network in given state
        output_tensor = model(input_tensor.unsqueeze(0))

        # Generate computation graph
        graph = make_dot(output_tensor, params=dict(model.named_parameters()))

        # Save graph
        graph.format = "png"
        save_path = os.path.join(folder, name)  # Ensure correct naming
        graph.render(save_path)
        print(f"Model architecture saved as '{save_path}.png'")
        
        del output_tensor,graph
        
    del input_tensor
    del model
    torch.cuda.empty_cache()



def color_augmentation(img_batch, mask_batch=None):
    """
    Apply color augmentation only within the masked region (if provided).

    Args:
        img_batch (Tensor): Input images (B, 3, H, W), range [-1,1] or [0,1]
        mask_batch (Tensor or None): (B, 1, H, W), values in [-1,1] or [0,1]
    Returns:
        Augmented batch with masked regions altered.
    """

    is_normalized = (img_batch.min() < 0)
    if is_normalized:
        img_batch = (img_batch + 1.0) / 2.0  # to [0,1]

    if mask_batch is not None:
        if mask_batch.min() < 0:
            mask_batch = (mask_batch + 1.0) / 2.0  # to [0,1]
        mask_batch = mask_batch.clamp(0, 1)

    B = img_batch.size(0)
    aug_imgs = []

    for i in range(B):
        img = img_batch[i]
        mask = mask_batch[i] if mask_batch is not None else None

        # Random color params
        brightness = random.uniform(0.9, 1.1)
        contrast   = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        hue        = random.uniform(-0.05, 0.05)
        gamma      = random.uniform(0.95, 1.05)

        aug_img = TF.adjust_brightness(img, brightness)
        aug_img = TF.adjust_contrast(aug_img, contrast)
        aug_img = TF.adjust_saturation(aug_img, saturation)
        aug_img = TF.adjust_hue(aug_img, hue)
        aug_img = TF.adjust_gamma(aug_img, gamma)

        if mask is not None:
            # Blend using mask: M * aug + (1-M) * original
            aug_img = aug_img * mask + img * (1 - mask)

        aug_imgs.append(aug_img)

    out_batch = torch.stack(aug_imgs)

    if is_normalized:
        out_batch = out_batch * 2.0 - 1.0  # back to [-1,1]

    return out_batch


def full_to_masked(full):
    
    real_rgb, real_mask = full[:, :3], full[:, 3:]

    real_mask_bin = ((real_mask + 1) / 2).round()

    masked_rgb = real_rgb * real_mask_bin + (-1.0) * (1 - real_mask_bin)
    
    full_masked = torch.cat([masked_rgb, real_mask], dim=1)

    return full_masked


def visualize_deepfakes(source_sample_tensor, target_sample_tensor,
                        G_AB, G_BA, device='cuda',
                        show_plot=True, save_plot=False, folder=None, name="Plot.png"):

    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        # Assume inputs already on device and half precision outside this function

        #Convert full rgb to masked rgb
        full_A_masked = full_to_masked(source_sample_tensor)
        full_B_masked = full_to_masked(target_sample_tensor)
        
        
        fake_source_org = G_BA(full_B_masked)
        fake_target_org = G_AB(full_A_masked)
        

        #Histogram matching
        fake_source_org_hist = histogram_match_with_mask(source_tensor = fake_source_org[:,:3,:,:],
                                                target_tensor = full_B_masked[:,:3,:,:],
                                                source_mask = fake_source_org[:,3:,:,:],
                                                target_mask = full_B_masked[:,3:,:,:]
                                                )
        
        fake_target_org_hist = histogram_match_with_mask(source_tensor = fake_target_org[:,:3,:,:],
                                                target_tensor = full_A_masked[:,:3,:,:],
                                                source_mask = fake_target_org[:,3:,:,:],
                                                target_mask = full_A_masked[:,3:,:,:]
                                                )
        
        ########################
        #Blend part
        source_eroded_mask = ((fake_source_org_hist[:,3:] + 1) / 2).round()
        target_eroded_mask = ((fake_target_org_hist[:,3:] + 1) / 2).round()
        
        
        source_blended = (1 - source_eroded_mask) * target_sample_tensor[:, :3] + fake_source_org_hist[:, :3] * source_eroded_mask
        target_blended = (1 - target_eroded_mask) * source_sample_tensor[:, :3] + fake_target_org_hist[:, :3] * target_eroded_mask
        

        def to_numpy(tensor):
            img = tensor.squeeze(0).detach().to(torch.float32).cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            return np.clip(img, 0, 1)


        s_raw = to_numpy(source_sample_tensor)
        t_raw = to_numpy(target_sample_tensor)

        s_real = to_numpy(full_A_masked)
        t_real = to_numpy(full_B_masked)
        
        s_fake_org = to_numpy(fake_source_org)
        t_fake_org = to_numpy(fake_target_org)
        
        s_fake_org_hist = to_numpy(fake_source_org_hist)
        t_fake_org_hist = to_numpy(fake_target_org_hist)
        
        s_tensor_blend = to_numpy(source_blended)
        t_tensor_blend = to_numpy(target_blended)
        


        fig, axs = plt.subplots(2, 6, figsize=(8, 6))
        
        axs[0, 0].imshow(s_raw[:,:,:3]); axs[0, 0].set_title("Raw image")
        axs[0, 1].imshow(s_real[:,:,:3]); axs[0, 1].set_title("Gen Input")
        axs[0, 2].imshow(t_fake_org[:,:,:3]); axs[0, 2].set_title("Deepfake")
        axs[0, 3].imshow(t_fake_org[:,:,:]); axs[0, 3].set_title("Masked")
        axs[0, 4].imshow(t_fake_org_hist[:,:,:]); axs[0, 4].set_title("Hist adjusted")
        axs[0, 5].imshow(t_tensor_blend[:,:,:]); axs[0, 5].set_title("Blended tensor")

        
        
        
        axs[1, 0].imshow(t_raw[:,:,:3]); axs[1, 0].set_title("Raw image")
        axs[1, 1].imshow(t_real[:,:,:3]); axs[1, 1].set_title("Gen Input")
        axs[1, 2].imshow(s_fake_org[:,:,:3]); axs[1, 2].set_title("Deepfake")
        axs[1, 3].imshow(s_fake_org[:,:,:]); axs[1, 3].set_title("Masked")
        axs[1, 4].imshow(s_fake_org_hist[:,:,:]); axs[1, 4].set_title("Hist adjusted")
        axs[1, 5].imshow(s_tensor_blend[:,:,:]); axs[1, 5].set_title("Blended tensor")

        
        for ax in axs.flatten():
            ax.axis("off")
        plt.tight_layout()

        if save_plot and folder:
            os.makedirs(folder, exist_ok=True)
            fig.savefig(os.path.join(folder, name))
            print(f"Plot saved to {os.path.join(folder, name)}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            plt.close()


        
    gc.collect()
    torch.cuda.empty_cache()


class DataPrefetcherWithAugmentation:
    def __init__(self, loader, device, augmenter):
        self.loader = iter(loader)
        self.device = device
        self.augmenter = augmenter
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            full_A, full_B = next(self.loader)
        except StopIteration:
            self.next_real_A_aug = None
            self.next_real_B_aug = None
            self.next_mask_A = None
            self.next_mask_B = None
            return

        with torch.cuda.stream(self.stream):
            full_A = full_A.to(self.device, non_blocking=True)
            full_B = full_B.to(self.device, non_blocking=True)

            real_A = full_A[:, :3, :, :]
            real_B = full_B[:, :3, :, :]
            mask_A = full_A[:, 3:, :, :]
            mask_B = full_B[:, 3:, :, :]

            real_A_aug = self.augmenter(real_A)
            real_B_aug = self.augmenter(real_B)

            self.next_real_A_aug = real_A_aug
            self.next_real_B_aug = real_B_aug
            self.next_mask_A = mask_A
            self.next_mask_B = mask_B

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
    
        real_A_aug = self.next_real_A_aug  # [B, 3, H, W]
        real_B_aug = self.next_real_B_aug  # [B, 3, H, W]
        mask_A = self.next_mask_A          # [B, 1, H, W]
        mask_B = self.next_mask_B          # [B, 1, H, W]
    
        if real_A_aug is not None:
            for t in [real_A_aug, real_B_aug, mask_A, mask_B]:
                t.record_stream(torch.cuda.current_stream())
    
            # === Concatenate RGB + Mask along channel dimension ===
            full_A_aug = torch.cat([real_A_aug, mask_A], dim=1)  # [B, 4, H, W]
            full_B_aug = torch.cat([real_B_aug, mask_B], dim=1)  # [B, 4, H, W]
            
            full_A_aug_masked = full_to_masked(full_A_aug)
            full_B_aug_masked = full_to_masked(full_B_aug)
            
        else:
            full_A_aug = full_B_aug = full_A_aug_masked = full_B_aug_masked= None
    
        self.preload()
        return full_A_aug, full_B_aug, full_A_aug_masked, full_B_aug_masked


def relativistic_gan_loss_D(D_real, D_fake):
    """
    Relativistic average discriminator loss (for discriminator training).
    No label smoothing. Mean is computed across batch and spatial dims.
    """
    # Flatten across batch and spatial dimensions
    D_real_flat = D_real.view(D_real.size(0), -1)
    D_fake_flat = D_fake.view(D_fake.size(0), -1)

    # Compute means across all samples and spatial positions
    D_fake_mean = D_fake_flat.mean()
    D_real_mean = D_real_flat.mean()

    # Compute BCE losses (no smoothing)
    loss_real = F.binary_cross_entropy_with_logits(D_real - D_fake_mean, torch.ones_like(D_real))
    loss_fake = F.binary_cross_entropy_with_logits(D_fake - D_real_mean, torch.zeros_like(D_fake))
    
    return (loss_real + loss_fake) / 2


def relativistic_gan_loss_G(D_real, D_fake):
    """
    Relativistic average generator loss (for generator training).
    No label smoothing. Mean is computed across batch and spatial dims.
    """
    D_real_flat = D_real.view(D_real.size(0), -1)
    D_fake_flat = D_fake.view(D_fake.size(0), -1)

    D_fake_mean = D_fake_flat.mean()
    D_real_mean = D_real_flat.mean()

    loss_real = F.binary_cross_entropy_with_logits(D_real - D_fake_mean, torch.zeros_like(D_real))
    loss_fake = F.binary_cross_entropy_with_logits(D_fake - D_real_mean, torch.ones_like(D_fake))
    
    return (loss_real + loss_fake) / 2


class EpochBalancedFaceDataset(Dataset):
    def __init__(self, source_array, target_array):
        assert source_array.shape[1:] == target_array.shape[1:], "Mismatch in face dims"
        
        self.source = source_array
        self.target = target_array
        self.is_source_smaller = len(source_array) <= len(target_array)
        self.min_len = min(len(source_array), len(target_array))
        self.max_len = max(len(source_array), len(target_array))

        self.source_is_large = len(source_array) > len(target_array)
        self.large_dataset = source_array if self.source_is_large else target_array
        self.small_dataset = target_array if self.source_is_large else source_array

        self.large_indices = self._sample_large_indices()

    def _sample_large_indices(self):
        """Randomly sample indices from the larger dataset to match the smaller one's length."""
        return random.sample(range(self.max_len), self.min_len)

    def on_epoch_start(self):
        """Call at the start of each epoch to reshuffle the large dataset sampling."""
        self.large_indices = self._sample_large_indices()

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        idx_small = idx
        idx_large = self.large_indices[idx]

        small_img = self.small_dataset[idx_small].transpose(2, 0, 1)
        large_img = self.large_dataset[idx_large].transpose(2, 0, 1)

        source_tensor = torch.from_numpy(small_img).float()
        target_tensor = torch.from_numpy(large_img).float()

        if self.source_is_large:
            return target_tensor, source_tensor  # source was larger, so return (small, sampled large)
        else:
            return source_tensor, target_tensor
        
        
        
        
def histogram_match_with_mask(source_tensor, target_tensor, source_mask, target_mask, alpha=0.85):
    """
    GPU-based differentiable histogram matching in masked regions.
    Returns a 4-channel tensor: RGB (matched) + original source mask.
    Ensures output RGB channels are clamped to [-1, 1].
    """
    assert source_tensor.shape == target_tensor.shape
    assert source_tensor.shape[1] == 3  # RGB
    assert source_mask.shape[1] == 1
    assert target_mask.shape[1] == 1

    # Normalize masks from [-1, 1] -> [0, 1], soft edge preserved
    s_mask = (source_mask + 1) / 2  # (B, 1, H, W)
    t_mask = (target_mask + 1) / 2  # (B, 1, H, W)

    eps = 1e-6
    B, C, H, W = source_tensor.shape

    def compute_mean_std(tensor, mask):
        masked = tensor * mask  # (B, C, H, W)
        mean = masked.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + eps)  # (B, C)
        var = ((masked - mean.unsqueeze(-1).unsqueeze(-1))**2 * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + eps)
        std = torch.sqrt(var + eps)  # (B, C)
        return mean, std

    # Compute per-sample channel stats
    s_mean, s_std = compute_mean_std(source_tensor, s_mask)
    t_mean, t_std = compute_mean_std(target_tensor, t_mask)

    # Adjust source -> target
    normed = (source_tensor - s_mean.unsqueeze(-1).unsqueeze(-1)) / (s_std.unsqueeze(-1).unsqueeze(-1) + eps)
    matched = normed * t_std.unsqueeze(-1).unsqueeze(-1) + t_mean.unsqueeze(-1).unsqueeze(-1)

    # Blend the transformation only in the masked region
    s_mask_full = s_mask.expand(-1, 3, -1, -1)  # (B, 3, H, W)
    blended_rgb = alpha * s_mask_full * matched + (1 - alpha * s_mask_full) * source_tensor

    # Clamp RGB channels to [-1, 1]
    blended_rgb = torch.clamp(blended_rgb, -1.0, 1.0)

    # Concatenate the source mask back to the RGB result
    output = torch.cat([blended_rgb, source_mask], dim=1)  # (B, 4, H, W)
    return output  
        




def generate_deepfakes_from_numpy(target_array, 
                                 G_BA,
                                 BlendNet = None,
                                 batch_size=12,
                                 device='cuda',
                                 dtype=torch.float32,
                                 memmap_folder="temp",
                                 memmap_filename="deepfake_output.dat",
                                 use_memmap=True):
    """
    Generates deepfakes from a numpy array with optional disk-backed memmap storage.
    """

    N, H, W, C = target_array.shape
    assert C == 4, "Expected 4 channels (RGB + mask)"

    # Setup memmap file path and folder
    if use_memmap:
        if not os.path.exists(memmap_folder):
            os.makedirs(memmap_folder)
        memmap_path = os.path.join(memmap_folder, memmap_filename)
        output_array = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(N, H, W, 4))
    else:
        output_array = np.empty((N, H, W, 4), dtype=np.float32)

    # Model setup
    G_BA = G_BA.to(device=device, dtype=dtype)
    G_BA.eval()
    if BlendNet is not None:
        BlendNet = BlendNet.to(device = device, dtype = dtype)
        BlendNet.eval()

    def to_tensor(np_array):
        return torch.from_numpy(np_array).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)

    def to_numpy(tensor):
        tensor = tensor.permute(0, 2, 3, 1).contiguous().float().cpu()
        return ((tensor + 1) / 2).numpy()

    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="Generating deepfakes"):
            batch = target_array[i:i + batch_size]
            B = batch.shape[0]

            t_full = to_tensor(batch)  # (B, 4, H, W)
            
            t_rgb, t_mask = t_full[:, :3], t_full[:, 3:]

            
            t_mask_bin = ((t_mask + 1) / 2).round()
            
            t_rgb_masked = (t_rgb * t_mask_bin) + ((-1.0) * (1 - t_mask_bin))
            #plt.imshow(to_numpy(t_rgb_masked)[0])
            
            t_full_masked = torch.cat([t_rgb_masked, t_mask], dim=1)
            

            # Forward pass (RGB)
            fake_s = G_BA(t_full_masked)  # (B, 4, H, W)


            # Histogram matching
            fake_s_hist = histogram_match_with_mask(
                source_tensor = fake_s[:, :3],
                target_tensor= t_full_masked[:, :3],
                source_mask = fake_s[:, 3:],
                target_mask = t_full_masked[:, 3:]
            )
            
            if BlendNet is not None:
                fake_s_hist = BlendNet(fake_s_hist[:,:3])
                fake_s_hist = torch.cat([fake_s_hist,fake_s[:,3:]], dim = 1)
                


            # Store directly into the preallocated array (memmap or RAM)
            output_array[i:i+B] = to_numpy(fake_s_hist)

    if use_memmap:
        output_array.flush()

    return output_array








