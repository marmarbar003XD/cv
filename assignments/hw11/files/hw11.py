"""HW11: 3D reconstruction from two views. Due Monday, November 23, 2026.

Complete the three functions below. Submit hw11.py, your two photos, and calibration.json.
Run:  python evaluate.py
      python evaluate.py --photos photos
"""
import cv2
import numpy as np


def reconstruct_sparse(image1, image2, K, dist):
    """Relative pose and sparse 3D points from two photos.

    image1, image2: BGR uint8 photos of the same size.
    K: (3, 3) intrinsic matrix.  dist: lens distortion coefficients.

    Return pts1, pts2, R, t, X:
        pts1, pts2: (N, 2) matched pixels in image1 and image2 (as given, distorted)
        R: (3, 3) and t: (3, 1) pose of camera 2 relative to camera 1, X_2 = R X_1 + t
        X: (N, 3) triangulated points in camera 1's frame; row i belongs to match i
    """
    # TODO: match features, undistort the matches, cv2.findEssentialMat (RANSAC),
    #       cv2.recoverPose, cv2.triangulatePoints.

    # image to grey
    img1_g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # sift 
    sift = cv2.SIFT_create()

    keyp_1, descrip1 = sift.detectAndCompute(img1_g, None)
    keyp_2, descrip2 = sift.detectAndCompute(img2_g, None)

    # Bf
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descrip1, descrip2, k = 2)

    # lowe ratio
    best_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            best_matches.append(m)


    coord1 = []
    coord2 = []

    for m in best_matches:

        pt1 = coord1.append(keyp_1[m.queryIdx].pt)
        pt2 = coord2.append(keyp_2[m.trainIdx].pt)

        coord1.append(pt1)
        coord2.append(pt2)


    pts1 = np.array(coord1)
    pts2 = np.array(coord2)

    # remove lens dist
    pts1_c = cv2.undistortPoints(pts1, K, dist)
    pts2_c = cv2.undistortPoints(pts2, K, dist)

    # Essential matrix
    E, mask_ess = cv2.findEssentialMat(pts1_c, pts2_c, K, cv2.Ransac)

    # relative pose
    _, R, t, mask = cv2.recoverPose(E, pts1_c, pts2_c, K, mask_ess)

    # triangulate
    P1 = K @np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    X4 = cv2.triangulatePoints(P1, P2, pts1_c.T, pts2_c.T)
    X = (X4[:3]/ X4[3]).T

    return pts1, pts2, R, t, X


def rectify_pair(image1, image2, K, dist, R, t):
    """Rectify both photos using the pose R, t of camera 2 relative to camera 1.

    Return rect1, rect2, R1, R2, P1, P2, Q:
        rect1, rect2: rectified BGR uint8 images, the same size as the inputs
        R1, R2, P1, P2, Q: the outputs of cv2.stereoRectify
    """
    # TODO: cv2.stereoRectify, cv2.initUndistortRectifyMap, cv2.remap.

    w, h = image1.shape[1], image1.shape[0]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, (w, h), R, t)

    # wrap 
    map1 = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w,h))
    rect1 = cv2.remap(image1, map1, cv2.INTER_LINEAR)

    map2 = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w,h))
    rect2 = cv2.remap(image2, map2, cv2.INTER_LINEAR)

    return rect1, rect2, R1, R2, P1, P2, Q


    


def reconstruct_dense(rect1, rect2, Q):
    """Disparity and dense 3D points from a rectified pair.

    Return disparity, valid, points:
        disparity: (H, W) float32 disparity in pixels
        valid: (H, W) Boolean, True where the 3D point is usable (finite, positive depth)
        points: (H, W, 3) 3D points in rectified camera 1's frame
    """
    # TODO: cv2.StereoBM or cv2.StereoSGBM, divide by 16, cv2.reprojectImageTo3D.
    raise NotImplementedError("Complete reconstruct_dense")
