# Driver Drowsiness and Inattention Detection

A student mini-project with:
- CNN training on a folder-based Yawn-Eye image dataset
- Live webcam monitoring using MediaPipe Face Mesh landmarks
- Eye-closure, yawning, and approximate head-direction indicators
- On-screen status and optional beep alert
- Training accuracy/loss plots and a classification report

## Important dataset note
NTHU-DDD is a video dataset and Yawn-Eye is an image dataset. Their folder structures can vary by download/version.
The CNN trainer expects class folders such as:
    dataset/yawn_eye/open_eyes/
    dataset/yawn_eye/closed_eyes/
    dataset/yawn_eye/yawn/
    dataset/yawn_eye/no_yawn/

Rename/copy the existing class folders to these names if necessary. The trainer also accepts other class folder names, but each class must have its own folder. Do not mix unrelated crop types into a class.

## Setup (Windows)
1. Install Python 3.10 or 3.11 (recommended for compatibility with TensorFlow and MediaPipe).
2. Open a terminal in this project folder.
3. Create a virtual environment:
   `py -3.10 -m venv .venv`
   `.venv\\Scripts\\activate`
4. Install packages:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`

If installation fails, install packages individually and check that your Python version is supported by TensorFlow and MediaPipe.

## Train the CNN
Place the Yawn-Eye image classes under `dataset/yawn_eye/`, then run:
    python train_model.py --data dataset/yawn_eye --epochs 15

The script saves `models/yawn_eye_cnn.keras`, class names, plots, and a classification report under `outputs/`.

## Start real-time webcam detection
    python webcam_monitor.py

Press Q to quit. If the webcam is not camera 0, try:
    python webcam_monitor.py --camera 1

The webcam module uses MediaPipe facial landmarks for live eye/mouth/head-pose estimates. It does not claim to use the CNN model directly, because the Yawn-Eye images may be cropped eye/mouth images rather than full face images. This avoids applying a classifier to a different type of input than it was trained on.

## Test an existing video file
    python webcam_monitor.py --video path/to/video.mp4

## Suggested project structure
    dataset/yawn_eye/<class folders>/
    models/
    outputs/
    train_model.py
    webcam_monitor.py
    requirements.txt

## How the live indicators work
- Eye Aspect Ratio (EAR): low values for a sustained period indicate possible eye closure.
- Mouth Aspect Ratio (MAR): high values for a sustained period indicate possible yawning.
- Head direction: estimates face orientation from facial landmarks; this is approximate, not true gaze tracking.
- A warning is raised when an indicator persists beyond a configurable threshold.

Thresholds vary by camera position, lighting, and individual face. Adjust the values in `webcam_monitor.py` after testing. Normal mirror checks can trigger false positives. This is an educational prototype, not a certified driver-safety system. Test while stationary; never operate the code or adjust it while driving.
