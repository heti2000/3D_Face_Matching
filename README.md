# 3d_Scanning
3D Scanning &amp; Motion Capture project repository

## Installation dependencies

### Dependencies

The following libraries need to be installed:
- Ceres
- OpenCV
- HDF5
- glog
- GLEW
- GLFW
- GCC >=13.0

#### Linux
```
sudo apt update
sudo apt install libhdf5-dev libopencv-dev python3-opencv libceres-dev libeigen3-dev
```

### Installation

```
cd 3D_scanning
mkdir build
cmake ..
make -j8
```

  
## Project structure
```
- datasets
- lib
- models
    - BFM
        - model2017-1_face12_nomouth.h5
        - Landmarks68_BFM.anl
    - PIPNet
        - pipnet_resnet101_10x68x32x256_300w.pth
- src
- tools
- main.cpp
- expression_transfer.cpp
- video_analyze.cpp
```
### Data preparation
There is some restrinction of file on each folder in order to be able run correctly the programme.
#### Face model
The Basel model you can find it on this [web](https://faces.dmi.unibas.ch/bfm/bfm2017.html). Save with the name that we showed above.

#### PIPNet model
Download `pipnet_resnet101_10x68x32x256_300w.pth` and put it in  the correct directory [web](https://github.com/jhb86253817/PIPNet).

#### Source Images
To be able to run everything, the landmarks need to be extracted using python and PIPNet.

For single images, copy your jpg file into the datasets folder and run the following command from the project directory:
````
python3 ./tools/landmark_detector.py ../datasets ../datasets/ImageName.jpg
````
This will create the files `input_ImageName.anl` and `output_ImageName.jpg`. Confirm with the `output_ImageName.jpg` file that your keypoints were successfully identified.

#### Source Videos
For videos, we require every image to be in the format `image_%04d.jpg` (i.e. `image_0001.jpg`, `image_0002.jpg`, ...).
You can extract the images yourself and call the command for the source images or use our provided script.
For this script ffmpeg needs to be installed.
Run the following commands from the project directory:
```
mkdir datasets/clipName
./tools/prepare_video.sh path/to/video.mp4 ./datasets/clipName
```

## Executables
We provide 3 experiments:
- `main`: Normal face matching for a single image
- `video_analyze`: Face matching for a video sequence (dense for the first image, sparse for the others)
- `expression_transfer`: Expression transfer

Those executables all provide a usage info and an example command, when calling them without arguments.
