"""HW10: Camera simulation and calibration. Due Wednesday, November 11, 2026.

Set SQUARE_SIZE_MM and complete the four functions below.
Submit hw10.py and your photos.
Run:  python evaluate.py projection
      python evaluate.py calibration photos
"""
import cv2
import numpy as np

# TODO: side length of one square of YOUR printed or displayed checkerboard, in mm.
SQUARE_SIZE_MM = 23


def project_points(X_w, K, R_c_w, t_c_w):
    """Return the pixel coordinates (u, v) of N world points, shape (N, 2).

    X_w:   (N, 3) array; row i is the world point ^wX_i.
    K:     (3, 3) intrinsic matrix.
    R_c_w: (3, 3) rotation ^cR_w from the world frame to the camera frame.
    t_c_w: (3, 1) translation ^ct_w from the world frame to the camera frame.

    Use NumPy. Do not call cv2.projectPoints in this function.
    """
    # TODO: world frame -> camera frame -> pixels (divide by the depth Z_c).

    # cam mat
    x_c = R_c_w @ X_w.T + t_c_w

    # cam coords
    cam_x = x_c[0, :]
    cam_y = x_c[1, :]
    cam_z = x_c[2, :]

    # normalize
    u_norm = cam_x/ cam_z
    v_norm = cam_y/ cam_z

    # intrinsic vals
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # pixel coords
    u = fx * u_norm + cx
    v = fy * v_norm + cy

    pix_coords = np.vstack((u, v))

    return pix_coords.T



def board_points(pattern_size, square_size):
    """Return the board points ^bX_i = (X, Y, 0), float32, shape (columns*rows, 3).

    pattern_size = (columns, rows) counts INNER corners, e.g., (8, 6).
    Start at (0, 0, 0) and go row by row: X increases fastest, then Y.
    Neighboring corners are square_size apart.
    """
    points = []
    for row in range(pattern_size[1]):
        for col in range(pattern_size[0]):
            points.append([col * square_size, row * square_size, 0])
    return np.array(points, dtype = np.float32)
            

def detect_board(image, pattern_size):
    """Return the refined corners from cv2.cornerSubPix, or None if not found.

    image: BGR uint8 photo. Do not modify it.
    Use cv2.findChessboardCorners, then refine with cv2.cornerSubPix.
    Return the refined corners as OpenCV returns them: float32, one (x, y)
    per corner, columns*rows corners in OpenCV's order.
    """

    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    state, corners = cv2.findChessboardCorners(image_g, pattern_size)

    if not state:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    ref_corner = cv2.cornerSubPix(image_g, corners, (11, 11), (-1, -1), criteria)

    return ref_corner




def calibrate_camera(images, pattern_size, square_size):
    """Calibrate the camera from checkerboard photos.

    images: a list of BGR uint8 photos, all the same size.
            Loop over it with `for image in images:`.
    Skip photos in which detect_board returns None.

    Return the five outputs of cv2.calibrateCamera, unchanged:
        rms, K, dist, rvecs, tvecs
    """
    image_size = (images[0].shape[1], images[0].shape[0])

    brd_point = board_points(pattern_size, square_size)
    obj_points = []
    img_points = []

    for image in images:
        corn_state = detect_board(image, pattern_size)
        if corn_state is None:
            continue
        obj_points.append(brd_point)
        img_points.append(corn_state)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size,  None, None)

    return rms, K, dist, rvecs, tvecs
    
