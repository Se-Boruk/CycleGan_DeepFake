import Functions 
import numpy as np
import os
import json
import torch
import Architectures
from tqdm import tqdm
import Training_assets
import gc
import cv2
import mediapipe as mp
from scipy.spatial.distance import cosine

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

######################################################################
# ( 1 ) Setting parameters
######################################################################
video_path = r"Original_Data\Portfolio_video\Avatar.mp4"
output_path = "Deepfake/Deepfake.mp4"
model_weights_path = "models/cyclegan_110.pth"

clear_memory_at_end = True
batch_size = 750        #Batch size of vide processing (500-1500 is good for keeping image statistic stable)
batch_size_model = 6    #Batch size fo model processing (best to keep same as in training, but can be as high as you want theoratically)
frame_skip = 1          #Skip every n frames in video (if you want transform vide keep 1. For testing or experimenting with model can apply higher value)

#Model hyperparameters
with open('models/model_input_shape.json') as f:
    input_config = json.load(f)
    
input_channels = input_config['in_channels']
output_channels = input_config['out_channels']
base_filters = input_config['base_filters']
n_residual_blocks = input_config['residual_blocks']

deepfake_h = input_config["height"]
deepfake_w = input_config['width']

######################################################################
# ( 2 ) Loading and preparing model
######################################################################
print("Loading Deepfake model...")
print("searching for cuda device...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model...")
G_BA = Architectures.Generator(input_channels=input_channels,
                                output_channels=output_channels,
                                n_residual=n_residual_blocks,
                                base_filters=base_filters
                                )

print("Loading weights")
checkpoint = torch.load(model_weights_path, map_location=device)
G_BA.load_state_dict(checkpoint['G_BA'])

del checkpoint
gc.collect()
print("Done!")
print("-----------------------------------------\n")


######################################################################
# ( 3 ) Setting up video deepfake 
######################################################################

#If no video, quit
if not os.path.exists(video_path):
    print("No video found...")
    exit()

#Prepare video cap reader with its global parameters
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#total_frames = 500
#Prepare video writer to save new video while processing
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

#Tracking of processed frames and batches
frame_counter = 0
batch_counter = 0

#Creating mediapipe face detector
mp_face_mesh = mp.solutions.face_mesh


######################################################################
# ( 4 ) Video ---> Deepfake processing loop
######################################################################

#Comment Initialize face‐mesh detector and progress bar
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as face_mesh, tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:

    while cap.isOpened():
        #Reset per‐batch containers for raw frames, mappings, face crops, coords, and histograms
        rgb_frames = []
        batch_frame_map = []
        frame_idxs = []
        faces = []
        coords = []
        hist_samples = []

        #Stop loop if processed all frames
        if frame_counter >= total_frames:
            print(f"\nReached artificial limit of {total_frames} frames. Stopping.")
            break

        #Load up to batch_size frames (skipping as per frame_skip) and convert to RGB
        while len(rgb_frames) < batch_size and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % frame_skip != 0:
                frame_counter += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb)
            batch_frame_map.append(frame_counter)
            frame_counter += 1
            pbar.update(1)

        if not rgb_frames:
            break

        print(f"\n--- Batch {batch_counter} | {len(rgb_frames)} frames loaded ---")

        #Detect face landmarks, rotate to neutral orientation, extract face masks, compute histograms
        for i, rgb_frame in enumerate(rgb_frames):
            results = face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                continue

            h, w, _ = rgb_frame.shape
            for face_landmarks in results.multi_face_landmarks:
                #build landmark array and convex‐hull mask
                landmarks = np.array([
                    (int(round(l.x * w)), int(round(l.y * h)))
                    for l in face_landmarks.landmark
                ])
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, cv2.convexHull(landmarks), 255)

                #correct head rotation, reproject landmarks adn masks
                angle = Functions.detect_face_rotation(landmarks)
                rgb_rot, mask_rot = Functions.rotate_image(rgb_frame, mask, angle)
                landmarks_rot = Functions.rotate_coordinates(landmarks, angle, w, h)

                #crop face ROI and compute  histogram channel wise
                x0, x1 = landmarks_rot[:,0].min(), landmarks_rot[:,0].max()
                y0, y1 = landmarks_rot[:,1].min(), landmarks_rot[:,1].max()
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                try:
                    rgba[...,:3] = rgb_rot
                    rgba[..., 3]   = mask_rot
                    face_crop = rgba[y0:y1, x0:x1]
                    hist = Functions.compute_image_histogram(face_crop)
    
                    frame_idxs.append(batch_frame_map[i])
                    faces.append(face_crop)
                    coords.append(((y0, x0), (y1, x1)))
                    hist_samples.append(hist)
                except:
                    pass
        print(f"Processing batch {batch_counter}...")

        #Filter out inconsistent faces by comparing per‐frame histograms to composite histogram (reference one), so different face should be filtered if occur on video
        composite_hist = Functions.compute_composite_histogram_from_list(hist_samples)
        frame_idxs_filtered, faces_filtered, coords_filtered = [], [], []

        for idx in range(len(faces)):
            #compute mean RGB similarity
            sim = np.mean([
                1 - cosine(composite_hist[c], hist_samples[idx][c])
                for c in range(3)
            ])
            if sim >= 0.8:
                frame_idxs_filtered.append(frame_idxs[idx])
                coords_filtered.append(coords[idx])
                #resize + normalize for generator input
                resized = cv2.resize(faces[idx], (deepfake_w, deepfake_h), interpolation=cv2.INTER_LANCZOS4)
                faces_filtered.append(resized.astype(np.float32) / 255.0)

        #remap filtered indices into batch array positions
        frame_idxs_local = [ batch_frame_map.index(fi) for fi in frame_idxs_filtered ]
        faces_np = np.array(faces_filtered, dtype=np.float32)
        masks_np = faces_np[..., 3:4].copy()
        inputs = (faces_np - 0.5) * 2  # scale to [-1, 1] for CycleGAN

        #Run generator network to produce deepfake face crops
        Target_deepfakes = Training_assets.generate_deepfakes_from_numpy(
            inputs, G_BA, BlendNet=None,
            batch_size=batch_size_model, device=device,
            memmap_folder="temp", memmap_filename="deepfake_output.dat",
            use_memmap=False
        )

        #For each deepfake crop: resize back, erode mask, blend into original frames
        for i, local_idx in enumerate(frame_idxs_local):
            deepfake = Target_deepfakes[i]
            (y0, x0), (y1, x1) = coords_filtered[i]
            h_crop, w_crop = y1 - y0, x1 - x0

            #resize generator output and isolate alpha
            df_resized = cv2.resize(deepfake, (w_crop, h_crop), interpolation=cv2.INTER_LANCZOS4)
            df_resized = np.clip(df_resized,0,1)
            
            mask = df_resized[..., 3]
            df_rgb = df_resized[..., :3]
            eroded = Functions.erode_and_soften_mask(mask, bin_thresh=0.1, max_dist_ratio=0.07, edge_power=1.5, decay_strength = 4.5)

            #blend into normalized ROI, then denormalize and write back
            roi = rgb_frames[local_idx][y0:y1, x0:x1].astype(np.float32) / 255.0
            blended = Functions.blend_with_color_bleed(roi, df_rgb, eroded, bleed_px=15)
            rgb_frames[local_idx][y0:y1, x0:x1] = (np.clip(blended * 255, 0, 255)).astype(np.uint8)

        #Write out all frames in this batch to the output video
        for frame in rgb_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"Batch {batch_counter} written to video.")
        batch_counter += 1

        #Comment Release per‐batch memory
        del rgb_frames, frame_idxs, faces, coords, hist_samples, 
        #del Target_deepfakes

#Finalize video streams (end it)
cap.release()
out.release()
print("\nFinished all batches.")


print("Adding sound...")
#Both functions work for full lenght video, however if video is both using n_frames and is part of original video any of this wont work
"""
#! Use if video used skip frames so its fastened (function will adjust audio speed)
Functions.merge_audio_adjust_speed_ffmpeg(video_no_audio_path = "Deepfake\Deepfake.mp4",
                                   video_with_audio_path = video_path,
                                   output_path = "Deepfake\Deepfake_audio.mp4"
                                   )
"""
#! Use if video is only part of original so f.e. first minute
Functions.merge_audio_partial_match(video_no_audio_path = "Deepfake\Deepfake.mp4",
                                   original_video_with_audio = video_path,
                                   output_path = "Deepfake\Deepfake_audio.mp4"
                                   )


gc.collect()
torch.cuda.empty_cache()    



