import os
import cv2
import torch
import numpy as np
from tqdm import tqdm  # For the progress bar
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet  # Import RRDBNet architecture

# Configuration of the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Path to the pre-trained model
model_path = 'weights/RealESRGAN_x4plus.pth'  # Updated model

# Define the model architecture
model = RRDBNet(
    num_in_ch=3,       # Number of input channels (RGB)
    num_out_ch=3,      # Number of output channels (RGB)
    num_feat=64,       # Number of features
    num_block=23,      # Number of RRDB blocks
    num_grow_ch=32,    # Number of growth channels
    scale=4            # Scale factor (x4)
)

# Load the model weights onto the device (GPU)
loadnet = torch.load(model_path, map_location=device)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
model.load_state_dict(loadnet[keyname], strict=True)
model.eval()
model = model.to(device)

# Instantiate the RealESRGANer model
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    dni_weight=None,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,         # Set to False for full precision
    device=device
)

# Paths to input and output videos
input_video_path = 'input.mp4'
output_video_path = 'input_1440p.mp4'

# Capture the input video
cap = cv2.VideoCapture(input_video_path)

# Get properties of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = input_width / input_height

# Desired output height (1440p)
output_height = 1440
output_width = int(aspect_ratio * output_height)

# Create the VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# Process each frame with a progress bar
with tqdm(total=frame_count, desc='Processing video', unit='frame') as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply Real-ESRNet
        try:
            output, _ = upsampler.enhance(frame_rgb, outscale=output_height / input_height)
        except Exception as e:
            print(f'Error processing frame: {e}')
            continue

        # Convert the result to BGR for OpenCV
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Resize to the exact output size if necessary
        output_bgr = cv2.resize(output_bgr, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

        # Write the frame to the output video
        out.write(output_bgr)

        # Update the progress bar
        pbar.update(1)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
