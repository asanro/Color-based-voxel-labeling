import glm
import random
import numpy as np
import cv2 as cv
import colormodel as cm

# global variables
block_size = 1.0
voxel_size = 30.0  # voxel every 3cm
lookup_table = []
camera_handles = []
background_models = []
global offline, color_model

# generate the floor grid locations
def generate_grid(width, depth):
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


# determines which voxels should be set
def set_voxel_positions(width, height, depth, frames, curr_time):
    global offline, color_model, img_floor
    global pers_centroids_total
    global color_centroids_total

    if len(lookup_table) == 0:
        create_lookup_table(width, height, depth)
        offline = True
        img_floor = np.ones((500, 500, 3), dtype=np.uint8) * 255
        pers_centroids_total=[]
        color_centroids_total=[]

    # initialize voxel list
    voxel_list = []

    # swap y and z
    voxel_grid = np.ones((width, depth, height), np.float32)

    # Loop over all the cameras
    for i_camera in range(4):
        path_name = './data/cam' + str(i_camera + 1)

        if curr_time == 0:
            # train MOG2 on background video, remove shadows, default learning rate
            background_models.append(cv.createBackgroundSubtractorMOG2())
            background_models[i_camera].setShadowValue(0)

            # open background.avi
            camera_handle = cv.VideoCapture(path_name + '/background.avi')
            num_frames = int(camera_handle.get(cv.CAP_PROP_FRAME_COUNT))

            # train background model on each frame
            for i_frame in range(num_frames):
                ret, image = camera_handle.read()
                if ret:
                    background_models[i_camera].apply(image)

            # close background.avi
            camera_handle.release()

            # open video.avi
            camera_handles.append(cv.VideoCapture(path_name + '/video.avi'))
            num_frames = int(camera_handles[i_camera].get(cv.CAP_PROP_FRAME_COUNT))


        # determine foreground
        foreground_image = background_subtraction(frames[i_camera][curr_time], background_models[i_camera])

        # set voxel to off if it is not visible in the camera, or is not in the foreground
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxel_grid[x, z, y]:
                        continue
                    voxel_index = z + y * depth + x * (depth * height)

                    projection_x = int(lookup_table[i_camera][voxel_index][0][0])
                    projection_y = int(lookup_table[i_camera][voxel_index][0][1])

                    if projection_x < 0 or projection_y < 0 or projection_x >= foreground_image.shape[
                        1] or projection_y >= foreground_image.shape[0] or not foreground_image[
                        projection_y, projection_x]:
                        voxel_grid[x, z, y] = 0.0

    colors = []
    lookup_table_cams = [[], [], [], []]
    lookup_table_on1 = {}
    lookup_table_on2 = {}
    lookup_table_on3 = {}
    lookup_table_on4 = {}
    count = 0
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_grid[x, z, y] > 0:
                    v = [x * block_size - width / 2, y * block_size, z * block_size - depth / 2]
                    voxel_list.append(v)
                    # look up table on for cam 1
                    count = count + 1
                    voxel_index = z + y * depth + x * (depth * height)
                    for i in range(4):
                        projection_x = int(lookup_table[i][voxel_index][0][0])
                        projection_y = int(lookup_table[i][voxel_index][0][1])
                        if i == 0:
                            lookup_table_on1[tuple(v)] = [projection_x, projection_y]
                            lookup_table_cams[i] = lookup_table_on1
                        if i == 1:
                            lookup_table_on2[tuple(v)] = [projection_x, projection_y]
                            lookup_table_cams[i] = lookup_table_on2
                        if i == 2:
                            lookup_table_on3[tuple(v)] = [projection_x, projection_y]
                            lookup_table_cams[i] = lookup_table_on3
                        if i == 3:
                            lookup_table_on4[tuple(v)] = [projection_x, projection_y]
                            lookup_table_cams[i] = lookup_table_on4

                    # color
                    colors.append([x / width, z / depth, y / height])
    """             
    # put voxels that are on in list
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_grid[x, z, y] > 0:
                    voxel_list.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
                    colors.append([x / width, z / depth, y / height])
   
    """

    if offline:
        color_model = cm.create_colormodel(voxel_list, lookup_table_cams)
        offline = False

    new_voxel_list, new_color = cm.online(color_model, voxel_list, lookup_table_cams, frames, curr_time, img_floor, pers_centroids_total, color_centroids_total)

    return new_voxel_list, new_color


# create lookup table
def create_lookup_table(width, height, depth):
    # create 3d voxel grid
    voxel_space_3d = []

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                voxel_space_3d.append(
                    [voxel_size * (x * block_size - width / 2), voxel_size * (z * block_size - depth / 2),
                     - voxel_size * (y * block_size)])

    # Loop over all the number of cameras
    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)

        # use config.xml to read camera calibration
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        mtx = file_handle.getNode('CameraMatrix').mat()
        dist = file_handle.getNode('DistortionCoeffs').mat()
        rvec = file_handle.getNode('Rotation').mat()
        tvec = file_handle.getNode('Translation').mat()
        file_handle.release()

        # project voxel 3d points to 2d in each camera
        voxel_space_2d, jac = cv.projectPoints(np.array(voxel_space_3d, np.float32), rvec, tvec, mtx, dist)
        lookup_table.append(voxel_space_2d)


# applies background subtraction to obtain foreground mask
def background_subtraction(image, background_model):
    foreground_image = background_model.apply(image, learningRate=0)

    # Find the biggest contour in the image
    contours, hierarchy = cv.findContours(foreground_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    biggest_contours = []
    for contour in contours:
        if cv.contourArea(contour) > 1000:
            biggest_contours.append(contour)

    # Draw filled version of biggest contour onto empty mask, resulting in final foreground
    mask = np.zeros(foreground_image.shape, np.uint8)

    cv.drawContours(mask, biggest_contours, -1, 255, -1)

    # Applying the morphologic closing operation (dilating, then eroding) gave results
    # that helped denoising the edges of the foreground without losing much detail, if any
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

    # remove noise through dilation and erosion
    erosion_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilation_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    foreground_image = cv.erode(mask, erosion_elt)
    foreground_image = cv.erode(foreground_image, erosion_elt)
    foreground_image = cv.erode(foreground_image, erosion_elt)
    foreground_image = cv.erode(foreground_image, erosion_elt)

    foreground_image = cv.dilate(foreground_image, dilation_elt)
    foreground_image = cv.dilate(foreground_image, dilation_elt)
    foreground_image = cv.dilate(foreground_image, dilation_elt)
    foreground_image = cv.dilate(foreground_image, dilation_elt)

    return foreground_image


# Gets stored camera positions
def get_cam_positions():
    cam_positions = []

    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        tvec = file_handle.getNode('Translation').mat()
        rvec = file_handle.getNode('Rotation').mat()
        file_handle.release()

        # obtain positions
        rotation_matrix = cv.Rodrigues(rvec)[0]
        positions = -np.matrix(rotation_matrix).T * np.matrix(tvec)
        cam_positions.append([positions[0][0], -positions[2][0], positions[1][0]])
    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


# Gets stored camera rotations
def get_cam_rotation_matrices():
    cam_rotations = []

    for i in range(4):
        camera_path = './data/cam' + str(i + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        rvec = file_handle.getNode('Rotation').mat()
        file_handle.release()

        # normalize rotations
        angle = np.linalg.norm(rvec)
        axis = rvec / angle

        # apply rotation to compensate for difference between OpenCV and OpenGL
        transform = glm.rotate(-0.5 * np.pi, [0, 0, 1]) * glm.rotate(-angle,
                                                                     glm.vec3(axis[0][0], axis[1][0], axis[2][0]))
        transform_to = glm.rotate(0.5 * np.pi, [1, 0, 0])
        transform_from = glm.rotate(-0.5 * np.pi, [1, 0, 0])
        cam_rotations.append(transform_to * transform * transform_from)
    return cam_rotations
