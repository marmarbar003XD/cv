"""HW09: detect and track one moving animal using optical flow.

Submit only this file. Use the course cv Conda environment.
Allowed flow methods: cv2.calcOpticalFlowPyrLK,
cv2.calcOpticalFlowFarneback, or DISOpticalFlow_create(...).calc(...).
You may add helpers here. Keep experiments and file I/O out of the function.
"""
import cv2
import numpy as np


def track_animal(frames):
    """Return one [x, y, width, height] box for each input frame.

    frames: tuple of BGR uint8 arrays, each shaped (685, 1280, 3),
            in chronological order at 10 frames per second.
            Do not modify the supplied arrays; copy them if needed.
    returns: real numeric NumPy array of shape (len(frames), 4).
             Coordinates refer to the supplied image, not a resized image.
             Use [nan, nan, nan, nan] for a missing detection.

    No initial box is supplied. Use one algorithm and parameter policy for
    every clip. You may keep an estimate when the animal briefly stops.
    """
    boxes = np.full((len(frames), 4), np.nan, dtype=float)

    # first frame
    intial_frame = frames[0]
    prev_frame_g = cv2.cvtColor(intial_frame, cv2.COLOR_BGR2GRAY)
    mag_threshold = 0.5

    # go through frames
    for f in range(1, len(frames)):
    
        curr_frame = frames[f]
        curr_frame_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
        # apply farneback
        farneback = cv2.calcOpticalFlowFarneback(prev_frame_g, curr_frame_g, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
    
        # get mag | ang no
        mag, _ = cv2.cartToPolar(farneback[..., 0], farneback[..., 1])
    
        # mask w/ pix vals + morph
        m = (mag > mag_threshold).astype(np.uint8) * 255
    
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_o = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
        mask_c = cv2.morphologyEx(mask_o, cv2.MORPH_CLOSE, kern)
    
        # find shapes
        contour, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # get largest blob
    
        large_c = None
        large_a = 100
    
    
        for c in contour:
            if cv2.contourArea(c) > large_a:
                large_c = c
                large_a = cv2.contourArea(c)
    
        # create box
        if large_c is not None:
            x, y, w, h = cv2.boundingRect(large_c)
            boxes[f] = [x, y, w, h]
        else:
            if not np.isnan(boxes[f-1]).any():
                boxes[f] = boxes[f-1]
            else:
                boxes[f] = [np.nan, np.nan, np.nan, np.nan]
    
        prev_frame_g = curr_frame_g
            
    return boxes