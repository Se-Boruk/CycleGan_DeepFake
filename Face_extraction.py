import Functions 
import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

frame_skip = 4


#Creating source faces (face to insert into video)
source_video_path = r"Original_Data\Sebastian_bright\Sebo_training.mp4"
source_faces_path = "Extracted_Faces\Sebastian_Faces.pickle"
      
if not os.path.exists(source_faces_path):
    _ = Functions.Extract_faces_from_video(video_path = source_video_path,
                                                      save_path = source_faces_path,
                                                      frame_skip = frame_skip)  

with open(source_faces_path, "rb") as f:
    Source_faces = pickle.load(f)





#Creating target face (face to deepfake - remove from the video)
target_video_path = r"Original_Data\Portfolio_video\Avatar_training.mp4"
target_faces_path = "Extracted_Faces\Avatar_Faces.pickle"

if not os.path.exists(target_faces_path):
     _ = Functions.Extract_faces_from_video(video_path = target_video_path,
                                                      save_path = target_faces_path,
                                                      frame_skip = frame_skip)  

with open(target_faces_path, "rb") as f:
    Target_faces = pickle.load(f)
    

#Optional - to view or save video from faces. For debugging, checking if loaded correctly
Functions.display_and_save_video(image_array = Source_faces.get("Faces"),
                                 save_path = None)



