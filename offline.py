import numpy as np
import cv2 as cv

# program flow
calibrate_intrinsics = False  # extrinsics calibration included
calibrate_extrinsics_only = True  # intrinsics calibration not included
show_axes = True
show_chessboard = False

# read checkerboard config variables
checkerboard_handle = cv.FileStorage('./data/checkerboard.xml', cv.FILE_STORAGE_READ)
board_width = int(checkerboard_handle.getNode('CheckerBoardWidth').real())
board_height = int(checkerboard_handle.getNode('CheckerBoardHeight').real())
tile_size = checkerboard_handle.getNode('CheckerBoardSquareSize').real()
checkerboard_handle.release()


# manual annotation of chessboard corners
def manual_annotation(window_name, image, board_size):
    # initialize 2d corner structure
    corners = []
    
    # set mouse callback
    cv.setMouseCallback(window_name, click, (corners, window_name, image))
    
    # corners should start left top, and run clockwise
    while len(corners) < 4:
        cv.waitKey(50)

    # transform chessboard area to rectangle
    (width, height) = board_size
    transform_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    perspective_matrix = cv.getPerspectiveTransform(np.array(corners, dtype=np.float32), transform_points)
    inverse_perspective_matrix = np.linalg.pinv(perspective_matrix)

    # interpolate chessboard corners, 
    grid_coordinates = np.zeros((width * height, 2), np.float32)
    grid_coordinates[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    return cv.perspectiveTransform(np.array([[grid_point] for grid_point in grid_coordinates], np.float32), inverse_perspective_matrix)


# mouse click event handler
def click(event_name, x, y, flags, event_data):
    (corners, window_name, image) = event_data
    if event_name == cv.EVENT_LBUTTONDOWN:
        corners.append([x, y])

        # draw circle
        cv.circle(image, (x, y), 3, ( 255, 255, 0 ), -1)
        cv.imshow(window_name, image)


# intrinsic camera calibration from a video with chessboard
def calibrate_intrinsics_from_video(file_name):
    # open intrinsics.avi
    file_handle = cv.VideoCapture(file_name)
    num_frames = int(file_handle.get(cv.CAP_PROP_FRAME_COUNT))

    # initialize image list
    image_list = []

    # loop over all frames with step size of 50
    for i_image in range(0, num_frames, 50):
        ret, frame = file_handle.read()
        if ret:
            image_list.append(frame)
            
    # 3d world coordinates of chessboard cornders
    chessboard_3d_coordinates = np.zeros((board_width * board_height, 3), np.float32)
    chessboard_3d_coordinates[:, :2] = tile_size * np.mgrid[0:board_width, 0:board_height].T.reshape(-1,2)

    # initialize arrays for 2d and 3d chessboard corner coordinates
    corner_2d_points = []
    corner_3d_points = []

    # loop over chessboard images
    for image in image_list:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # attempt to find the chessboard corners
        ret, found_corners = cv.findChessboardCorners(gray_image, (board_width, board_height), None)
        if ret:
            # corners found, refine
            refined_corners = cv.cornerSubPix(gray_image, found_corners, (7, 7), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        else:
            continue
            # corners not found, proceed to manual annotation
            cv.imshow('manual_calibration', image)
            annotated_corners = manual_annotation('manual_calibration', image, (board_width, board_height))
            refined_corners = cv.cornerSubPix(gray_image, annotated_corners, (1, 1), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv.destroyAllWindows()

        # add 2d and 3d chessboard corner coordinates
        corner_2d_points.append(refined_corners)
        corner_3d_points.append(chessboard_3d_coordinates)

        # display image with chessboard corners
        if show_chessboard:
            cv.drawChessboardCorners(image, (board_width, board_height), refined_corners, ret)
            cv.imshow('Chessboard', image)
            cv.waitKey(500)

    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(corner_3d_points, corner_2d_points, gray_image.shape[::-1], None, None)

    # close calibration windows
    if show_chessboard:
        cv.destroyAllWindows()

    return mtx, dist


# extrinsic camera calibration from the first frame of a video with chessboard
def calibrate_extrinsics_from_frame(window_name, mtx, dist):
    camera_handle = cv.VideoCapture(window_name)
    
    # 3d world coordinates of chessboard corners
    chessboard_3d_coordinates = np.zeros((board_width * board_height, 3), np.float32)
    chessboard_3d_coordinates[:, :2] = tile_size * np.mgrid[0:board_width, 0:board_height].T.reshape(-1,2)

    # initialize rotation and translation matrices
    rvecs = []
    tvecs = []

    # read first frame
    ret, frame = camera_handle.read()
    if ret:
        cv.imshow(window_name, frame)
        
        # manually annotate corners
        annotated_corners = manual_annotation(window_name, frame, (board_width, board_height))

        # solve for extrinsic parameters
        ret, rvec, tvec = cv.solvePnP(chessboard_3d_coordinates, annotated_corners, mtx, dist)

    cv.destroyAllWindows()
    
    return rvec, tvec


# main loop of the program.
def main():
    # Configure all four cameras
    for i in range(4):
        camera_path = './data/cam' + str(i + 1)
        
        # ================================================================
        # calibrate intrinsics & extrinsics, or read calibration from file
        # ================================================================

        if calibrate_intrinsics:
            # calibrate intrinsics
            (mtx, dist) = calibrate_intrinsics_from_video(camera_path + '/intrinsics.avi')

            # calibrate extrinsics
            (rvec, tvec) = calibrate_extrinsics_from_frame(camera_path + '/checkerboard.avi', mtx, dist)

            # save calibration
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_WRITE)
            file_handle.write('CameraMatrix', mtx)
            file_handle.write('DistortionCoeffs', dist)
            file_handle.write('Rotation', rvec)
            file_handle.write('Translation', tvec)
            file_handle.release()
        elif calibrate_extrinsics_only:
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)

            # read intrinsics
            mtx = file_handle.getNode('CameraMatrix').mat()
            dist = file_handle.getNode('DistortionCoeffs').mat()

            file_handle.release()

            # calibrate extrinsics
            (rvec, tvec) = calibrate_extrinsics_from_frame(camera_path + '/checkerboard.avi', mtx, dist)

            # save calibration
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_WRITE)
            file_handle.write('CameraMatrix', mtx)
            file_handle.write('DistortionCoeffs', dist)
            file_handle.write('Rotation', rvec)
            file_handle.write('Translation', tvec)
            file_handle.release()
        else:
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)

            # read intrinsics
            mtx = file_handle.getNode('CameraMatrix').mat()
            dist = file_handle.getNode('DistortionCoeffs').mat()

            # read extrinsics
            rvec = file_handle.getNode('Rotation').mat()
            tvec = file_handle.getNode('Translation').mat()

            file_handle.release()

        # ================================================================
        # visualize calibration
        # ================================================================

        # show axes
        if show_axes:
            video_handle = cv.VideoCapture(camera_path + '/checkerboard.avi')
            
            ret, frame = video_handle.read()
            if ret:
                # Draw and display the axes
                axes_3d = (5 * tile_size) * np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,-1]], np.float32)
                axes_2d, jac = cv.projectPoints(axes_3d, rvec, tvec, mtx, dist)
                cv.line(frame, (int(axes_2d[0][0][0]), int(axes_2d[0][0][1])), (int(axes_2d[1][0][0]), int(axes_2d[1][0][1])), (0,0,255), 2)
                cv.line(frame, (int(axes_2d[0][0][0]), int(axes_2d[0][0][1])), (int(axes_2d[2][0][0]), int(axes_2d[2][0][1])), (0,255,0), 2)
                cv.line(frame, (int(axes_2d[0][0][0]), int(axes_2d[0][0][1])), (int(axes_2d[3][0][0]), int(axes_2d[3][0][1])), (255,0,0), 2)
                cv.imshow(camera_path + '/checkerboard.avi', frame)
                cv.waitKey()
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
