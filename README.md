# Super Resolution for Low-Quality Videos

This project applies super-resolution to low-quality videos, enhancing their resolution using deep learning models. It uses the Real-ESRGAN architecture to upscale videos to a higher resolution (e.g., 1440p), improving their visual quality.

## Features

- **Super Resolution**: Enhances video frames from lower resolutions to 1440p using the Real-ESRGAN model.
- **Deep Learning Model**: Utilizes a pre-trained Real-ESRGAN model built on the RRDBNet architecture.
- **GPU Support**: Optimized to run on GPUs (CUDA) if available, with a fallback to CPU.
- **Progress Tracking**: Displays a progress bar for each frame processed, providing real-time updates on the video enhancement process.
- **Input/Output Support**: Processes input videos in formats like MP4 and outputs the enhanced video in higher resolution.

## How It Works

1. **Model Architecture**: The model uses the RRDBNet architecture, which is loaded with pre-trained weights from the Real-ESRGAN repository.
2. **Frame Processing**: Each video frame is converted to RGB, passed through the upscaling model, and then written to the output video at the desired resolution (1440p).
3. **Video Enhancement**: The video is processed frame by frame, applying super-resolution, and then saving the results in a higher-quality output file.

## Technologies Used

- **Deep Learning Model**: Real-ESRGAN based on RRDBNet.
- **Computer Vision**: OpenCV for reading and writing video frames.
- **GPU Acceleration**: Supports CUDA for faster video processing.
- **Python Libraries**: Torch, TQDM, NumPy, OpenCV.

## Setup and Installation

1. Clone the repository and install the required dependencies:
   ```bash
   pip install -r requirements.txt
