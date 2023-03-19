import cv2 as cv
import numpy as np


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return [i, x.index(v)]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def interpolation(interp_corner):
    a = interp_corner

    top_left = interp_corner[0]
    bottom_left = interp_corner[1]
    bottom_right = interp_corner[2]
    top_right = interp_corner[3]

    # if abs(bottom_right[0]-bottom_left[0])<abs(top_left[1]-bottom_left[1]): # cambiar dependiendo camara
    x1 = np.linspace(top_right[0], top_left[0], 6)
    y1 = np.linspace(top_right[1], top_left[1], 6)

    x2 = np.linspace(bottom_left[0], top_left[0], 8)
    y2 = np.linspace(bottom_left[1], top_left[1], 8)

    x3 = np.linspace(bottom_right[0], top_right[0], 8)
    y3 = np.linspace(bottom_right[1], top_right[1], 8)

    x4 = np.linspace(bottom_right[0], bottom_left[0], 6)
    y4 = np.linspace(bottom_right[1], bottom_left[1], 6)

    corners_int_f = []
    corners_int_print = []
    for j in range(0, 8):
        for i in range(0, 6):
            line1 = [[x1[i], y1[i]], [x4[i], y4[i]]]
            line2 = [[x2[j], y2[j]], [x3[j], y3[j]]]
            x_int, y_int = line_intersection(line1, line2)
            corners_int_f.append([[x_int, y_int]])
            x_int = int(x_int)
            y_int = int(y_int)
            corners_int_print.append((x_int, y_int))

    return corners_int_f, corners_int_print


def Camera_Calibration(path):

    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            top_left_corner.append([x, y])
            four_manual_corners.append(top_left_corner)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, str(x) + ',' +
                       str(y), (x, y), font,
                       1, (255, 0, 0), 2)
            cv.imshow('img', img)
        if event == cv.EVENT_RBUTTONDOWN:
            font = cv.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv.putText(img, str(b) + ',' +
                       str(g) + ',' + str(r),
                       (x, y), font, 1,
                       (255, 255, 0), 2)
            cv.imshow('img', img)

    cap = cv.VideoCapture(path + '/intrinsics.avi')
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # determine characteristics of the chessboard
    size1 = 8
    size2 = 6
    square_size = 115  # mm

    length_vector = np.linspace(1, length, num=length)
    frames_used_vector = np.linspace(length_vector[0], length_vector[-1], num=55, dtype='int')
    count = 1

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, square_size, 0.001)

    # create an array of the real world points in mm starting from the top left
    objp = np.zeros((size1 * size2, 3), np.float32)
    objp[:, :2] = np.mgrid[0:size1, 0:size2].T.reshape(-1, 2)
    objp[:, :2] = objp[:, :2] * square_size  # in mm

    # arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space (mm)
    imgpoints = []  # 2d points in image plane (pixels).
    top_left_corner = []

    # Randomly select 25 frames
    frameIds = cap.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=70)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        # for each frame in video obtain cube and axis
        # convert image to gray
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        find, corners = cv.findChessboardCorners(gray, (size1, size2), None)

        c = 0
        if find:
            objpoints.append(objp)
            # Detect corners location in subpixels
            corners2 = cv.cornerSubPix(gray, corners, (size1, size2), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(frame, (size1, size2), corners2, ret)
            c += 1
        # Display median frame
        cv.imshow('img', frame)
        cv.waitKey(2)

    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
        errors.append(error)
    print("The total error: {}".format(mean_error / len(objpoints)))

    # Task 1.2. : extrinsic parameters

    cap = cv.VideoCapture(path + '/checkerboard.avi')
    ret, frame = cap.read()

    # Convert the area

    img = frame

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 115, 0.001)  # REVISAR EL 115 ANTES ERA 30
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)
    objp[:, :2] = objp[:, :2] * 115  # in mm
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    four_manual_corners = []
    top_left_corner = []
    objpoints.append(objp)
    cv.imshow("img", img)
    cv.setMouseCallback('img', click_event)
    k = 0
    # Close the window when key q is pressed
    while k != 113:
        # Display the image
        cv.imshow("img", img)
        k = cv.waitKey(0)
    coordinates_input = top_left_corner

    cv.destroyAllWindows()

    img_perspective = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cap = cv.VideoCapture(path + '/checkerboard.avi')
    ret, frame = cap.read()

    cv.destroyAllWindows()

    img_perspective = frame
    src = np.array(
        [[coordinates_input[0][0], coordinates_input[0][1]], [coordinates_input[1][0], coordinates_input[1][1]],
         [coordinates_input[2][0], coordinates_input[2][1]], [coordinates_input[3][0], coordinates_input[3][1]]],
        np.float32)
    dst = np.array([[0, 0], [0, 500], [500, 500], [500, 0]], np.float32)
    transform_mat = cv.getPerspectiveTransform(src, dst)
    out = cv.warpPerspective(img_perspective, transform_mat, (500, 500), flags=cv.INTER_LINEAR)

    # Doing the interpolation

    imgpoints = []  # 2d points in image plane.
    top_left_corner = []
    img = out
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    count = []
    four_manual_corners = []
    objpoints.append(objp)
    cv.imshow('img', img)
    cv.setMouseCallback('img', click_event)
    k = 0

    four_corners = top_left_corner

    cv.imshow("img", img)
    k = cv.waitKey(0)
    cv.destroyAllWindows()

    corners_four_manual = []
    imgpoints = []
    for i in range(0, 4):
        pts = np.float32(np.array([[[four_corners[i][0], four_corners[i][1]]]]))
        warped_pt = cv.perspectiveTransform(pts, np.linalg.inv(transform_mat))[0]
        corners_four_manual.append([warped_pt[0][0], warped_pt[0][1]])
    corners_int_f, corners_int_print = interpolation(corners_four_manual)

    cap = cv.VideoCapture(path + '/checkerboard.avi')
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    color = (0, 0, 255)  # in BGR format
    radius = 1
    # Draw the dots on the image
    for dot in corners_int_print:
        cv.circle(frame, dot, radius, color, -1)
        cv.imshow('circles', frame)
        cv.waitKey(40)
    # Show the image with the dots

    cv.destroyAllWindows()
    cv.imshow('circles', frame)
    k = cv.waitKey(0)
    cv.destroyAllWindows()

    cv.destroyAllWindows()

    corners_int = np.array(corners_int_f)
    ret, rvecs, tvecs = cv.solvePnP(objp, corners_int, mtx, dist)

    points = np.float32([[5, 0, 0], [0, 5, 0], [0, 0, -5], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv.projectPoints(points * 115, rvecs, tvecs, mtx, dist)
    axisPoints = np.round(axisPoints)
    axisPoints = axisPoints.astype(int)
    corners_int = np.round(corners_int)
    corners_int = corners_int.astype(int)
    frame = cv.line(frame, tuple(corners_int[0].ravel()), tuple(axisPoints[0].ravel()), (255, 4, 0), 3)
    frame = cv.line(frame, tuple(corners_int[0].ravel()), tuple(axisPoints[1].ravel()), (2, 255, 0), 3)
    frame = cv.line(frame, tuple(corners_int[0].ravel()), tuple(axisPoints[2].ravel()), (46, 2, 255), 3)

    cv.imshow('img', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save XML file
    filename = path + '/config.xml'
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write('rvecs', rvecs)
    fs.write('tvecs', tvecs)
    fs.write('mtx', mtx)
    fs.write('dist', dist)
    fs.release()

    return rvecs, tvecs, mtx, dist


if __name__ == '__main__':
    for c in range(1, 5):
        path = 'data/cam' + str(c)
        rvecs, tvecs, mtx, dist = Camera_Calibration(path)
