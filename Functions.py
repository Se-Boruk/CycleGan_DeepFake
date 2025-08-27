import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import pickle
from scipy.spatial.distance import cosine
import json
import glob
import tempfile
import subprocess
import shutil
import gc



def merge_audio_partial_match(video_no_audio_path, original_video_with_audio, output_path):
    def get_ffmpeg_executables():
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path is None or ffprobe_path is None:
            raise FileNotFoundError("ffmpeg or ffprobe not found in PATH.")
        return ffmpeg_path, ffprobe_path

    ffmpeg_path, ffprobe_path = get_ffmpeg_executables()

    def get_duration(path):
        cmd = [
            ffprobe_path, '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    # Get duration of processed silent video
    duration_video = get_duration(video_no_audio_path)

    # Temp file to store the clipped audio
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    # Extract only N seconds of audio to match the processed video
    cmd_extract_audio = [
        ffmpeg_path,
        "-y",
        "-i", original_video_with_audio,
        "-t", str(duration_video),
        "-vn",
        "-acodec", "aac",
        temp_audio_path
    ]
    subprocess.run(cmd_extract_audio, check=True)

    # Combine audio with processed video
    cmd_merge = [
        ffmpeg_path,
        "-y",
        "-i", video_no_audio_path,
        "-i", temp_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]
    subprocess.run(cmd_merge, check=True)

    os.remove(temp_audio_path)


def merge_audio_adjust_speed_ffmpeg(video_no_audio_path, video_with_audio_path, output_path):
    # Helper: get ffmpeg paths
    def get_ffmpeg_executables():
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path is None or ffprobe_path is None:
            raise FileNotFoundError("ffmpeg or ffprobe not found in system PATH.")
        return ffmpeg_path, ffprobe_path

    ffmpeg_path, ffprobe_path = get_ffmpeg_executables()

    def get_duration(path):
        cmd = [
            ffprobe_path, '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    # Durations
    duration_video = get_duration(video_no_audio_path)
    duration_audio = get_duration(video_with_audio_path)

    speed_factor = duration_audio / duration_video

    # Break down speed factor into valid chain of atempo filters
    def atempo_chain(factor):
        filters = []
        while factor > 2.0:
            filters.append("atempo=2.0")
            factor /= 2.0
        while factor < 0.5:
            filters.append("atempo=0.5")
            factor *= 2.0
        filters.append(f"atempo={factor:.6f}")
        return ",".join(filters)

    audio_filter = atempo_chain(speed_factor)

    # Temp audio file
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    # Adjust audio speed
    cmd_speed_audio = [
        ffmpeg_path, '-y', '-i', video_with_audio_path,
        '-filter:a', audio_filter,
        '-vn', '-acodec', 'aac',
        temp_audio_path
    ]
    subprocess.run(cmd_speed_audio, check=True)

    # Combine with no-audio video
    cmd_combine = [
        ffmpeg_path, '-y', '-i', video_no_audio_path,
        '-i', temp_audio_path,
        '-c:v', 'copy', '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(cmd_combine, check=True)

    os.remove(temp_audio_path)
    

def rotate_image(img, mask, angle):
    """ Rotates image and mask by given angle in openCV """
    if angle == 0:
        return img, mask
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img, mask

def Extract_faces_from_video(video_path, save_path=None, frame_skip=5, batch_size=100):

    if not os.path.exists(video_path):
        print("No video found...")
        return

    if save_path is not None:
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_skip + 1

    frame_counter = 0
    saved_faces = 0
    hist_samples = []
    shapes = []

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_faces_dir = os.path.join(temp_dir, "temp_faces")
    os.makedirs(temp_faces_dir, exist_ok=True)

    metadata = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh, tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % frame_skip != 0:
                frame_counter += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                h, w, _ = rgb_frame.shape
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array([(int(round(l.x * w)), int(round(l.y * h))) for l in face_landmarks.landmark])
                    mask = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(landmarks), 255)

                    angle = detect_face_rotation(landmarks)
                    rgb_rot, mask_rot = rotate_image(rgb_frame, mask, angle)
                    landmarks_rot = rotate_coordinates(landmarks, angle, w, h)

                    x0, x1 = np.min(landmarks_rot[:, 0]), np.max(landmarks_rot[:, 0])
                    y0, y1 = np.min(landmarks_rot[:, 1]), np.max(landmarks_rot[:, 1])

                    #segmented_face = cv2.bitwise_and(rgb_rot, rgb_rot, mask=mask_rot)
                    segmented_face = rgb_rot
                    #plt.imshow(segmented_face)


                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    try:
                        rgba[..., :3] = segmented_face
                        rgba[..., 3] = mask_rot
    
                        #x_off = int((x1-x0)*0.1)
                        #y_off = int((y1-y0)*0.1)  
                        
                        #x0 = x0 - x_off
                        #x1 = x1 + x_off
                        
                        #y0 = y0 - y_off
                        #y1 = y1 + y_off
                        
                        #x0 = max(x0,0)
                        #y0 = max(y0,0)
                        
                        #x1 = min(x1,w)
                        #y1 = min(y1,h)
                        
    
                        face_crop = rgba[y0:y1, x0:x1]
                        landmark_crop = landmarks_rot - [x0, y0]
                        
    
                        face_hist = compute_image_histogram(face_crop)
                        hist_samples.append(face_hist)
                        shapes.append(face_crop.shape[:2])
    
                        # Temporarily save face and landmarks
                        np.savez_compressed(os.path.join(temp_faces_dir, f"{saved_faces:06d}.npz"),
                                            face=face_crop, landmark=landmark_crop, frame_idx=frame_counter,
                                            coords=((y0, x0), (y1, x1)))
                        saved_faces += 1
                    except:
                        print("Rotation error, skipping...")

            frame_counter += 1
            pbar.update(1)

    cap.release()

    # === MEDIAN HISTOGRAM & SHAPE ===
    print("Computing global histogram and median shape...")
    composite_hist = compute_composite_histogram_from_list(hist_samples)
    median_h = int(np.median([s[0] for s in shapes]))
    median_w = int(np.median([s[1] for s in shapes]))

    resized_faces_path = os.path.join(temp_dir, "resized_faces.dat")
    resized_landmarks_path = os.path.join(temp_dir, "resized_landmarks.dat")
    resized_faces = np.memmap(resized_faces_path, dtype=np.uint8, mode='w+', shape=(saved_faces, median_h, median_w, 4))
    
    resized_landmarks = np.memmap(resized_landmarks_path, dtype=np.float32, mode='w+', shape=(saved_faces, 468, 2))

    print("Filtering and resizing...")
    kept_idx = []
    score_sum, score_all = 0.0, 0.0
    kept = 0

    for i in tqdm(range(saved_faces), desc="Outlier filtering + resize"):
        data = np.load(os.path.join(temp_faces_dir, f"{i:06d}.npz"))
        face = data["face"]
        landmark = data["landmark"]
        frame_idx = data["frame_idx"]
        coords = data["coords"]

        hist = compute_image_histogram(face)
        sim = np.mean([1 - cosine(composite_hist[c], hist[c]) for c in range(3)])
        score_all += sim

        if sim >= 0.8:
            resized_img, resized_lm = resize_image_with_landmarks(face, landmark, target_size=(median_h, median_w))
            resized_faces[kept] = resized_img
            resized_landmarks[kept] = resized_lm
            kept_idx.append(frame_idx)
            metadata.append(coords)
            score_sum += sim
            kept += 1

    # Trim memmap to actual kept count
    resized_faces.flush()
    resized_landmarks.flush()
    resized_faces = np.memmap(resized_faces_path, dtype=np.uint8, mode='r+', shape=(kept, median_h, median_w, 4))
    resized_landmarks = np.memmap(resized_landmarks_path, dtype=np.float32, mode='r+', shape=(kept, 468, 2))

    print("\nSaved faces:", kept)
    print("Avg similarity (kept):", round(score_sum / kept, 4) if kept > 0 else 0)
    print("Avg similarity (all):", round(score_all / saved_faces, 4) if saved_faces > 0 else 0)

    r_data = {
        "Frame_idx": kept_idx,
        "Faces": resized_faces,
        "Float_landmarks": resized_landmarks,
        "Original_coords": metadata,
    }

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(r_data, f)
    
    del r_data, data        
    
    # Clean up memmap references
    resized_faces.flush()
    resized_landmarks.flush()
    del resized_faces
    del resized_landmarks
    gc.collect()  # Force release of file handles   

    clear_folder("temp/temp_faces")
    clear_folder("temp")


def display_and_save_video(image_array, save_path = None, fps = 15):
    """
    Displays an OpenCV window showing a sequence of RGBA images as a video 
    and saves it as "integrity_test.mp4" in the given directory.

    Args:
        image_array (np.ndarray): NumPy array of shape (num_frames, height, width, 4) in RGBA format.
        save_dir (str): Directory where the video should be saved.
        fps (int, optional): Frames per second for the saved video. Defaults to 30.
    """
    if len(image_array.shape) != 4 or image_array.shape[-1] != 4:
        raise ValueError("Input array must have shape (num_frames, height, width, 4) for RGBA images.")

    num_frames, height, width, _ = image_array.shape
    
    if save_path is not None:
        pass
        
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # Display video
    for frame in image_array:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR for OpenCV display
        cv2.imshow("Integrity Test Video", bgr_frame)  
        video_writer.write(bgr_frame )  # Write frame to video file

        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit early
            break

    # Cleanup
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video saved at: {save_path}")

def detect_face_rotation(landmarks):
    """
    Determines the rotation angle (0, 90, 180, 270 degrees) by landmarks position.
    """
    left_eye, right_eye, nose, chin = get_key_landmarks(landmarks)

    # Calculate face height (distance between chin and forehead/nose)
    face_height = abs(chin[1] - nose[1])  # You could use forehead or nose if it's available

    # Calculate threshold as 25% of face height
    threshold = face_height * 0.4

    # Chin should always be the lowest point
    if chin[1] < min(left_eye[1], right_eye[1]):  # Chin above eyes = 180° rotation
        return 180

    # Check if eyes are misaligned based on a percentage of face height
    eye_slope = abs(left_eye[1] - right_eye[1])

    if eye_slope > threshold:  # If eyes are NOT horizontally aligned based on threshold
        if left_eye[1] > right_eye[1]:  
            return 90  # Left eye is lower → 90° rotation
        else:  
            return 270  # Right eye is lower → 270° rotation

    return 0  # No rotation needed


def get_key_landmarks(landmarks):
    """ Extracts relevant landmarks from MediaPipe detection """
    LEFT_EYE = [33, 133]  # Outer and inner corner
    RIGHT_EYE = [362, 263]
    NOSE_TIP = 1
    CHIN = 152

    left_eye = np.mean([landmarks[LEFT_EYE[0]], landmarks[LEFT_EYE[1]]], axis=0)
    right_eye = np.mean([landmarks[RIGHT_EYE[0]], landmarks[RIGHT_EYE[1]]], axis=0)
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]

    return left_eye, right_eye, nose, chin




def rotate_coordinates(coords, angle, img_width, img_height):
    """
    Rotate the given coordinates by the specified angle (90, 180, 270 degrees).
    :param coords: List of (x, y) tuples.
    :param angle: Rotation angle (should be 90, 180, or 270 degrees).
    :param img_width: Width of the image (for reference).
    :param img_height: Height of the image (for reference).
    :return: List of rotated (x, y) tuples.
    """
    rotated_coords = []
    if angle == 0:
        return coords
    else:
        for x, y in coords:
            if angle == 90:
                # Rotate 90 degrees clockwise
                new_x = img_height - y
                new_y = x
            elif angle == 180:
                # Rotate 180 degrees (flip both axes)
                new_x = img_width - x
                new_y = img_height - y
            elif angle == 270:
                # Rotate 270 degrees clockwise (or 90 degrees counter-clockwise)
                new_x = y
                new_y = img_width - x
            else:
                raise ValueError("Rotation angle must be one of 90, 180, or 270 degrees.")
    
            rotated_coords.append([new_x, new_y])
    
        return np.array(rotated_coords)





def resize_image_with_landmarks(image, landmarks, target_size):
    """
    Resizez image to target size alongside with its landmarks 
    (crucial coordinates in the image), so they match after resizing
    """
    h, w = image.shape[:2]
    # Compute scaling factor to maintain aspect ratio
    scale = min(target_size[0] / h, target_size[1] / w)
    
    h_scale = target_size[0] / h
    w_scale = target_size[1] / w


    # Use LANCZOS for best upscaling and AREA for best downscaling
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA


    target_h = target_size[0]
    target_w = target_size[1]
    opencv_size = (target_w,target_h)
    
    resized_landmarks = landmarks*[w_scale,h_scale]
    resized_image = cv2.resize(image, opencv_size, interpolation=interpolation)
    
    return resized_image, resized_landmarks



def compute_composite_histogram(faces_list):
    histograms = []

    for face in faces_list:
        if face is None or face.size == 0:
            continue  # Skip empty or invalid images
        
        # Convert to RGB (OpenCV loads in BGR)

        
        # Compute histograms for each channel (normalize to sum=1)
        hist_r = cv2.calcHist([face], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([face], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([face], [2], None, [256], [0, 256]).flatten()

        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        histograms.append(np.stack([hist_r, hist_g, hist_b], axis=0))  # Shape: (3, 256)

    if not histograms:
        return None  # No valid histograms found

    histograms = np.array(histograms)  # Shape: (num_faces, 3, 256)

    # Compute the median histogram across all faces
    median_histogram = np.median(histograms, axis=0)  # Shape: (3, 256)

    return median_histogram  # Returns a (3, 256) array for R, G, B


def compute_composite_histogram_from_list(hist_list):
    """
    Computes the global median histogram from a list of histograms.
    Each item in hist_list should be an array of shape (3, 256).
    Returns median histogram: shape (3, 256)
    """
    if not hist_list:
        return None

    hist_stack = np.stack(hist_list, axis=0)  # Shape: (N, 3, 256)
    median_hist = np.median(hist_stack, axis=0)  # Shape: (3, 256)
    return median_hist


def compute_image_histogram(image):
    """
    Compute the normalized histogram for each color channel (R, G, B)
    within the masked (foreground) area only.

    Args:
        image (np.ndarray): Input image with shape (H, W, 4), where the 4th channel is the binary mask (0 or 255).
        
    Returns:
        hist (list of np.ndarray): List containing histograms for R, G, and B channels.
    """
    if image is None or image.shape[2] != 4:
        print("Invalid image. Must be 4-channel (RGB + mask).")
        return None

    rgb = image[:, :, :3]
    mask = image[:, :, 3]

    histograms = []
    for i in range(3):  # R, G, B channels
        hist = cv2.calcHist([rgb], [i], mask, [256], [0, 256])
        hist = hist / hist.sum() if hist.sum() > 0 else hist  # Avoid division by zero
        histograms.append(hist.flatten())

    return histograms


def clear_memmap_cache(folder="temp", extension="*.dat"):
    files = glob.glob(os.path.join(folder, extension))
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")


def Uniform_Data(Source_faces, Target_faces, new_h = None, new_w = None):
    if new_h is None or new_w is None:
        Source_face_shape = Source_faces.get("Faces").shape
        Target_face_shape = Target_faces.get("Faces").shape
        
        
        
        Source_aspect_ratio = Source_face_shape[2] / Source_face_shape[1]
        Target_aspect_ratio = Target_face_shape[2] / Target_face_shape[1]
        
        Aspect_ratio = round((Source_aspect_ratio + Target_aspect_ratio) / 2, 4)
        
        
        h_diff = Source_face_shape[1] / Target_face_shape[1]
        w_diff = Source_face_shape[2] / Target_face_shape[2]
        
        
        if h_diff<=0.5 or h_diff>=2:
            print("One of the faces height if twice as big as the other one. It may be source of noise in final deepfake!")
        if w_diff<=0.5 or w_diff>=2:
            print("One of the faces width if twice as big as the other one. It may be source of noise in final deepfake!")
        
        
        Source_face_area = Source_face_shape[1]*Source_face_shape[2]
        Target_face_area = Target_face_shape[1]*Target_face_shape[2]
        
        if Source_face_area < Target_face_area:
            h = Source_face_shape[1]
            w = Source_face_shape[2]
        else:
            h = Target_face_shape[1]
            w = Target_face_shape[2]
        
        
        new_h, new_w = resize_to_aspect(h, w, Aspect_ratio)

    
    
    #############
    #Resizing Source faces
    Original_S_Faces = Source_faces.get("Faces")
    Original_S_Landmarks = Source_faces.get("Float_landmarks")
    S_Faces_array = []
    S_Landmarks_array = []
    
    
    for i in tqdm(range(Original_S_Faces.shape[0]),desc = "Resizing Source faces..."):
        
        face, landmark = resize_image_with_landmarks(image = Original_S_Faces[i],
                                                     landmarks = Original_S_Landmarks[i],
                                                     target_size = (new_h, new_w)
                                                     )
        S_Faces_array.append(face)
        S_Landmarks_array.append(landmark)
        
    S_Faces_array = np.asarray(S_Faces_array)
    S_Landmarks_array = np.asarray(S_Landmarks_array)
    
    
    #############
    #Resizing Target faces
    Original_T_Faces = Target_faces.get("Faces")
    Original_T_Landmarks = Target_faces.get("Float_landmarks")
    T_Faces_array = []
    T_Landmarks_array = []
    
    
    for i in tqdm(range(Original_T_Faces.shape[0]),desc = "Resizing Target faces..."):
        
        face, landmark = resize_image_with_landmarks(image = Original_T_Faces[i],
                                                     landmarks = Original_T_Landmarks[i],
                                                     target_size = (new_h, new_w)
                                                     )
        T_Faces_array.append(face)
        T_Landmarks_array.append(landmark)
        
    T_Faces_array = np.asarray(T_Faces_array)
    T_Landmarks_array = np.asarray(T_Landmarks_array)
    
    
    #Replace resized arrays into the dictionary
    Source_faces["Faces"] = S_Faces_array
    Source_faces["Float_landmarks"] = S_Landmarks_array
    
    Target_faces["Faces"] = T_Faces_array
    Target_faces["Float_landmarks"] = T_Landmarks_array
    
    
    Source_face_shape = Source_faces.get("Faces").shape
    Target_face_shape = Target_faces.get("Faces").shape
    
    print("Source shape: ", Source_face_shape)
    print("Target shape: ", Target_face_shape)
    
    return Source_faces, Target_faces


def save_model_input_shape(input_h,
                           input_w,
                           input_channels,
                           output_channels,
                           base_filters,
                           residual_blocks,
                           save_dir,
                           filename='model_input_shape.json'
                           ):

    shape_dict = {
        "height": int(input_h),
        "width": int(input_w),
        "in_channels": int(input_channels),
        "out_channels": int(output_channels),
        "base_filters": int(base_filters),
        "residual_blocks": int(residual_blocks)   
        }

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Define full path
    save_path = os.path.join(save_dir, filename)
    
    # Save JSON
    with open(save_path, 'w') as f:
        json.dump(shape_dict, f, indent=4)
    
    print(f"Model input shape saved to {save_path}")


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')



def blend_with_color_bleed(roi, deepfake_resized, mask, bleed_px=8):
    """
    Blend with edge color bleeding to improve seamlessness.
    
    Args:
        roi (np.ndarray): (H,W,3) background RGB, float32 in [0,1]
        deepfake_resized (np.ndarray): (H,W,3) target RGB, float32 in [0,1]
        mask (np.ndarray): (H,W,1) float32 mask in [0,1], soft
        bleed_px (int): feather radius to borrow colors from outside mask
    
    Returns:
        np.ndarray: blended RGB image
    """
    # 1. Ensure shapes
    assert roi.shape == deepfake_resized.shape
    assert mask.ndim == 3 and mask.shape[2] == 1

    H, W = mask.shape[:2]
    
    # 2. Create reverse mask (for outside)
    inv_mask = 1.0 - mask

    # 3. Feather outside region color into the deepfake
    # Use a blur on the inverse mask to smear background color inward
    bleed_mask = cv2.GaussianBlur(inv_mask, (2*bleed_px+1, 2*bleed_px+1), 0)
    if bleed_mask.ndim == 2:
        bleed_mask = bleed_mask[..., np.newaxis]

    # Optional: normalize bleed_mask to stay within [0,1]
    bleed_mask = np.clip(bleed_mask, 0, 1)

    # 4. Mix some of the background (roi) into deepfake at edges
    softened_deepfake = deepfake_resized * (1 - bleed_mask) + roi * bleed_mask

    # 5. Standard alpha blend
    blended = roi * (1 - mask) + softened_deepfake * mask

    return np.clip(blended, 0, 1)


def erode_and_soften_mask(mask, bin_thresh=0.1, max_dist_ratio=0.08, edge_power=2.5, decay_strength=4.0):
    """
    Smooths a mask with extended transition zone using distance transform + exponential decay.

    Args:
        mask (np.ndarray): (H,W) float32 mask in [0,1]
        bin_thresh (float): threshold for binary mask
        max_dist_ratio (float): relative width of transition (0.05–0.2)
        edge_power (float): exponent controlling curve sharpness (higher = sharper edge)
        decay_strength (float): affects how steep decay is

    Returns:
        np.ndarray: (H,W,1) float32 mask with smooth falloff
    """
    assert mask.ndim == 2 and mask.dtype == np.float32
    H, W = mask.shape

    pad = int(max(H, W) * 0.05) + 10  # enough padding for erosion/dist
    padded = np.pad(mask, pad, mode='constant', constant_values=0)

    H_pad, W_pad = padded.shape
    max_dist = int(max(H_pad, W_pad) * max_dist_ratio)
    if max_dist < 1:
        max_dist = 1

    # Binarize
    bin_mask = (padded > bin_thresh).astype(np.uint8)

    # Distance transforms
    dist_out = cv2.distanceTransform(1 - bin_mask, cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)

    # Smooth mask base
    smooth_mask = np.ones_like(padded, dtype=np.float32)

    # Outer falloff
    outside = dist_out <= max_dist
    d = dist_out[outside] / max_dist
    smooth_mask[outside] = np.exp(- (d ** edge_power) * decay_strength)

    # Optional soft inner edge
    inside = dist_in <= max_dist
    d2 = dist_in[inside] / max_dist
    inner_soft = 1 - np.exp(- (d2 ** edge_power) * decay_strength)
    smooth_mask[inside] = np.minimum(smooth_mask[inside], inner_soft)

    # Crop back to original size
    final = smooth_mask[pad:pad+H, pad:pad+W]
    return np.clip(final, 0, 1)[..., np.newaxis].astype(np.float32)


def generate_face_heatmaps(image_shape, landmarks, sigma=4):
    """
    Generates a batch of grayscale heatmaps with subpixel accuracy for landmarks.

    Parameters:
        image_shape (tuple): Shape of the output heatmaps (height, width).
        landmarks (np.array): Array of shape (batch_size, num_landmarks, 2) 
                              containing (x, y) coordinates for facial landmarks.
        sigma (int): Standard deviation for Gaussian blur.

    Returns:
        heatmaps (np.array): Batch of heatmaps with shape (batch_size, height, width, 1).
    """
    batch_size = landmarks.shape[0]
    heatmaps = np.zeros((batch_size, image_shape[0], image_shape[1], 1), dtype=np.float32)  # Create batch array

    for i in tqdm(range(batch_size), desc = "Generating heatmaps..."):
        heatmap = np.zeros(image_shape[:2], dtype=np.float32)  # Grayscale heatmap
        
        for x, y in landmarks[i]:  # Iterate over landmarks
            x+=1
            y+=1
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                # Integer & fractional parts
                x_int, y_int = int(x), int(y)
                x_frac, y_frac = x - x_int, y - y_int

                # Bilinear interpolation: distribute intensity to 4 neighboring pixels
                if x_int + 1 < image_shape[1] and y_int + 1 < image_shape[0]:  # Ensure inside bounds
                    heatmap[y_int, x_int]     += (1 - x_frac) * (1 - y_frac)
                    heatmap[y_int, x_int + 1] += x_frac * (1 - y_frac)
                    heatmap[y_int + 1, x_int] += (1 - x_frac) * y_frac
                    heatmap[y_int + 1, x_int + 1] += x_frac * y_frac

        # Apply Gaussian blur for smooth heatmap regions
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Normalize to 0-1 range
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmaps[i, :, :, 0] = heatmap  # Store in batch array

    # Remove very small values
    heatmaps[np.abs(heatmaps) < 1e-6] = 0

    return heatmaps



def Recalculate_landmarks(Face_dict):
    #Open numpy array with faces:
    Faces = Face_dict.get("Faces")
    
    # Initialize FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
                            static_image_mode=True,  # Single image or video tracking
                            max_num_faces=1,  # Max number of faces to detect
                            refine_landmarks=True,  # High-precision lips/eye tracking
                            min_detection_confidence=0.6,  # Increase for more reliable detection
                            min_tracking_confidence=0.6  # More stable tracking
                            ) as face_mesh:
        landmark_list = []
        face_list = []
        for face in tqdm(Faces, desc="Recalculating landmarks...", unit="face"):
            
            # Convert to RGB (drop alpha channel)
            rgb_face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)
            # Detect face
            results = face_mesh.process(rgb_face)
    
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = rgb_face.shape
                    landmarks = np.array([(l.x * w, l.y * h) for l in face_landmarks.landmark])
                        
                    landmark_list.append(landmarks)
                    face_list.append(face)

    Face_dict["Float_landmarks"] = np.asarray(landmark_list)
    Face_dict["Faces"] = np.asarray(face_list)
    return Face_dict



def resize_to_aspect(h, w, target_aspect):
    """
    Adjust height and width to achieve the target aspect ratio with minimal distortion,
    prioritizing one dimension (either height or width) to remain closest to its original value.
    
    Parameters:
        h (int): Original height of the image
        w (int): Original width of the image
        target_aspect (float): Desired aspect ratio (width / height)
    
    Returns:
        (new_h, new_w): Optimized height and width
    """
    # Compute current aspect ratio
    current_aspect = w / h
    
    # Prioritize resizing based on the difference between target and current aspect ratios
    if target_aspect > current_aspect:
        # Target is wider, adjust width, maintain height as constant
        new_w = int(h * target_aspect)
        new_h = h
    else:
        # Target is taller, adjust height, maintain width as constant
        new_h = int(w / target_aspect)
        new_w = w
    
    return new_h, new_w




    















