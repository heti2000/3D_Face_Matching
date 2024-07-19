import torchlm
import cv2
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import sys

# Define a function to format and save landmarks
def save_landmarks_as_anl(landmarks, filename):
    with open(filename, 'w') as file:
        for landmark in landmarks:
            # Format each landmark as 'x, y'
            line = f"{landmark[0]} {landmark[1]}\n"
            file.write(line)


# get filename from command line
folder_path = sys.argv[1]
file_name = sys.argv[2].split("/")[-1].split(".")[0]

img_path = f"{folder_path}/{file_name}.jpg"
save_path = f"{folder_path}/output_{file_name}.jpg"
checkpoint = "./models/PIPNet/pipnet_resnet101_10x68x32x256_300w.pth"
device = "cpu"
image = cv2.imread(img_path)
# set map_location="cuda" if you want to run with CUDA
torchlm.runtime.bind(faceboxesv2())
torchlm.runtime.bind(
        pipnet(
            backbone="resnet101",
            pretrained=True,
            num_nb=10,
            num_lms=68,
            net_stride=32,
            input_size=256,
            meanface_type="300w",
            backbone_pretrained=False,
            map_location=device,
            checkpoint=checkpoint
        )
    )
 # will auto download pretrained weights from latest release if pretrained=True
landmarks, bboxes = torchlm.runtime.forward(image)
# save landmarks and bboxes

save_landmarks_as_anl(landmarks[0], f"{folder_path}/input_{file_name}.anl")

print(landmarks)
image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
image = torchlm.utils.draw_landmarks(image, landmarks=landmarks, circle=5)

cv2.imwrite(save_path, image)

