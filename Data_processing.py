import Functions 
import os
import pickle
import numpy as np
######################################################
#1
#Load Faces

#Dirs for extracted faces
source_faces_path = "Extracted_Faces\Sebastian_Faces.pickle"
target_faces_path = "Extracted_Faces\Avatar_Faces.pickle"


with open(source_faces_path, "rb") as f:
    Source_faces = pickle.load(f)
    
with open(target_faces_path, "rb") as f:
    Target_faces = pickle.load(f)
    

######################################################
#2
#Uniform sizes of source and target face
Source_faces, Target_faces = Functions.Uniform_Data(Source_faces = Source_faces,
                                                    Target_faces = Target_faces,
                                                    new_h = 320,
                                                    new_w = 256
                                                    )


######################################################
#3
# Recalculating landmarks after size uniform
Source_faces = Functions.Recalculate_landmarks(Face_dict = Source_faces)
Target_faces = Functions.Recalculate_landmarks(Face_dict = Target_faces)



######################################################
#4
#Extracting faces
S_faces = Source_faces.get("Faces")
T_faces = Target_faces.get("Faces")


######################################################
#5
#Preparing heatmaps from landmakrs
sigma = 3

#Source faces
S_heatmaps = Functions.generate_face_heatmaps(image_shape = S_faces.shape[1:3],
                                             landmarks = Source_faces.get("Float_landmarks"),
                                             sigma=sigma
                                             )
#Target faces
T_heatmaps = Functions.generate_face_heatmaps(image_shape = T_faces.shape[1:3],
                                             landmarks = Target_faces.get("Float_landmarks"),
                                             sigma=sigma
                                             )
######################################################
#6
#Cut % edge of each data (outliers and abominations may be at beginning and end)




S_edge = int(len(S_faces)*0.02)
T_edge = int(len(T_faces)*0.02)

S_faces = S_faces[S_edge:len(S_faces)-S_edge]
S_heatmaps = S_heatmaps[S_edge:len(S_heatmaps) - S_edge]

T_faces = T_faces[T_edge:len(T_faces)-T_edge]
T_heatmaps = T_heatmaps[T_edge:len(T_heatmaps) - T_edge]


######################################################
#7
#Clearing other variables
del Source_faces, source_faces_path, Target_faces, target_faces_path, sigma, f, S_edge, T_edge


######################################################
#8
#Data normalization (-1,1)
S_faces = (S_faces.astype(np.float32) / 127.5) - 1
S_heatmaps = S_heatmaps*2 -1

T_faces = (T_faces.astype(np.float32) / 127.5) - 1
T_heatmaps = T_heatmaps*2 -1

######################################################
#9
#Stack data
Source_data = np.concatenate([S_faces, S_heatmaps], axis=-1)
Target_data = np.concatenate([T_faces, T_heatmaps], axis=-1)

del S_faces, S_heatmaps, T_faces, T_heatmaps


######################################################
#10
#Save
Source_data_path = "Processed_data/Source_Data.npy"
if not os.path.exists(Source_data_path):
    
    directory = os.path.dirname(Source_data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    np.save(Source_data_path, Source_data)
    

Target_data_path = "Processed_data/Target_Data.npy"
if not os.path.exists(Target_data_path):
    
    directory = os.path.dirname(Target_data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    np.save(Target_data_path, Target_data)
    









