# CycleGan DeepFake
![alt text](https://github.com/Se-Boruk/CycleGan_DeepFake/blob/master/gifs/Monk.gif?raw=true)

**Left**: Original

**Right**: Deepfake

This project aims to utilize CycleGan architecture and create targeted DeepFake model.

It is targeted deepfake which means that you need to train it for specific 2 faces.

### Project contains essential scripts for:
- Extracting faces from video                      (Face_extraction.py)
- Processing extracted faces                       (Data_processing.py)
- Training deepfake                                (DeepFake_training.py)
- Generating deepfake                              (Video_deepfake_pararell.py)

And others (libs / testing scripts)

### Due to the file size, and obvious reason project does not contains:
- Training datasets
- models allowing to recreate my face.

## Explanation of concept
### Face extraction
Faces are extracted from the 2 videos containing people we want to swap. Faces are filtered and slightly processed, so obtained dataset does not contain outliers or artifacts.
### Face processing
After extraction, faces are uniformed in size. If size is specified (more stable and repeatable), they will be resized to given size and if not size will be calculated automatically, which can fit better into exact face scenario, but may harm experimenting and overall stability of the process.
### Training DeepFake
Training model for face swapping. Parameters can be customized, depending of the video size, face "difficulty" etc. Default parameters works well for me and my case.
### Generating DeepFake
Applying swapped face to selected video. Face adjusted based on histogram statistics of original face and blended into into the original face, reducing edges effect.

## Full video of my face-swap on YouTube:
![alt text](https://github.com/Se-Boruk/CycleGan_DeepFake/blob/master/gifs/Avatar.gif?raw=true)

Youtube video: https://youtu.be/hHH5KBjyaK0