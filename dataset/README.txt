Put the extracted Yawn-Eye dataset here.

Expected:
dataset/train
    closed_eyes/
        image1.jpg
    open_eyes/
        image2.jpg
    yawn/
        image3.jpg
    no_yawn/
        image4.jpg

Folder names can differ, but each class must be in its own immediate subfolder. If your dataset already has train/test folders, combine or reorganize class folders carefully and avoid duplicate images across splits. Do not put NTHU video files into this image training directory; use videos for testing with webcam_monitor.py --video.
