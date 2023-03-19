import cv2 as cv
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

clusterPersons = [[], [], [], []]


def match_center(centers, centers_filtered):
    """
    match_center is a function which mat h the centers obtained from the first clustering
    with the centers obtained with the second clustering after the K-means due to a
    disordering.
    """
    # Get the proper order of the new centers_filtered
    center_ordered = centers
    for i in range(len(centers)):
        sum_all = []
        for j in range(len(centers)):
            sum = np.sum(abs(centers[j] - centers_filtered[i]))
            sum_all.append(sum)
        min_value = min(sum_all)
        min_index = sum_all.index(min_value)
        center_ordered[i] = centers_filtered[min_index]

    # Order the centers_filtered
    # reordered_array = centers_filtered[order, :]

    return center_ordered


def remove_outliers(labels, centers, voxels):
    """
    remove_outliers removes the ghost voxels that are making the K-means clustering messy
    by taking the 0.05 percentile of the distances between the centroids of the clusters.
    Finally, it removes those outlier voxels and calculates the new cluster centroids and
    labels.
    """
    distances = []

    # Calculate the distances for all clusters.
    for i in range(len(voxels)):
        center = centers[labels[i]]
        distance = np.sqrt((voxels[i][0] - center[0][0]) ** 2 + (voxels[i][1] - center[0][1]) ** 2)
        distances.append(distance)

    # Calculate the 0.05 percentile of the distances
    threshold = np.percentile(distances, 95)

    mask = distances < threshold

    voxels_filtered = voxels[mask]
    labels_filtered = voxels[mask]

    # Cluster voxels in 3d space based on x/y information
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    _, labels_filtered, centers_filtered = cv.kmeans(voxels_filtered, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    # centers_filtered_matched = match_center(centers, centers_filtered)

    return labels_filtered, centers, voxels_filtered


def cluster(voxels):
    """
    Cluster is a function which calculates the K-means algorithm with OpenCV, returns the voxel labeled
    and the centroids of each cluster.
    """
    # Cluster voxels in 3d space based on x/y information
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    # Convert to numpy array and drop height information
    voxels = np.float32(voxels)[:, [0, 2]]

    _, labels_def, centers = cv.kmeans(voxels, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    labels_filtered, centers, voxels_filtered = remove_outliers(labels_def, centers, voxels)

    return labels_def, centers


def assign_color(p_voxels, label):
    """
    assign_color matches each cluster with its correspondent color
    """
    if label == 0:
        color = [[0, 0, 225] for _ in range(np.shape(p_voxels)[0])]
    if label == 1:
        color = [[0, 255, 0] for _ in range(np.shape(p_voxels)[0])]
    if label == 2:
        color = [[255, 0, 225] for _ in range(np.shape(p_voxels)[0])]
    if label == 3:
        color = [[255, 0, 0] for _ in range(np.shape(p_voxels)[0])]
    return color


def final_labels(cam_labels):
    """
    Final_labels makes a final decision of the labels obtained for each person in each camera.
    The function's criteria gives more weight to the first camera and less to the others. However,
    whenever there are two cameras with the same label combination, that is the final output considered.
    """
    seen = {}
    for lst in cam_labels:
        key = tuple(lst)
        if key in seen:
            seen[key] += 1
            if seen[key] == 2:
                del seen[key]
        else:
            seen[key] = 1
    for lst in cam_labels:
        if lst == cam_labels[1]:
            return lst
    for lst in cam_labels:
        if tuple(lst) in seen:
            return lst


def online(offline_models, voxels, lookuptable, all_frames, nframe, img_floor, pers_centroids_total,
           color_centroids_total):
    """
    online is a function dedicated to develop the online phase of assignment 3. It is composed of a
    K-means clustering, a comparison of the offline color models to the online GMMs probabilities, a
    label matching to obtain the final labelling of each person and a 2D path tracking on the floor.
    """
    # Get voxels + labels that should be used for creating a color model (those above mean height)
    voxelslabels, centroids = cluster(voxels)

    labels = np.ravel(voxelslabels)
    voxels = np.float32(voxels)

    cam_labels = [[], [], [], []]

    print('HAS ENTERED ONLINE')
    print('The current time is: ')
    print(nframe)

    # Number of cameras
    for i in range(4):
        colors = []
        VoxelClusters = []
        matching = [[], [], [], []]

        frame = cv.cvtColor(all_frames[i][nframe], cv.COLOR_BGR2HSV)

        for label in range(4):
            voxels_person = voxels[labels == label]  # save voxel if the label is same
            pixelCluster, colorCluster = [], []

            VoxelClusters.append(voxels_person)

            # Take only above the belt and cut the head
            tshirt = np.mean(voxels_person[:, 1], dtype=np.int_)
            voxel_roi = voxels_person[:, 1] > tshirt
            voxels_person_roi = voxels_person[voxel_roi]

            head = np.max(voxels_person_roi[:, 1])
            voxel_roi = voxels_person_roi[:, 1] < 3 / 4 * head
            voxels_person_roi = voxels_person_roi[voxel_roi]

            # For each cluster group of labelled voxels
            for v in voxels_person_roi:
                # Find coordinates belonging to voxel
                pixel = lookuptable[i][tuple(v)]
                pixelCluster.append(pixel)

            # Compare colors in cluster to all models of current camera
            pred_values = []
            for j in range(len(offline_models[i])):
                # Cluster ROI
                roi = np.array([frame[y, x] for [x, y] in pixelCluster], dtype=np.float32)

                overall_logprob = 0.0
                for sample in roi:
                    (logprob, _), _ = offline_models[i][j].predict2(sample)
                    overall_logprob += logprob

                pred_values.append(overall_logprob)

            matching[label] = pred_values

        # Makes the matching of the person
        new_labels = optimize.linear_sum_assignment(np.array(matching))
        cam_labels[i] = new_labels[1].tolist()

    color_centroids = []
    pers_centroids = []

    cam_labels_maj = final_labels(cam_labels)

    for j in range(len(cam_labels_maj)):
        color = assign_color(VoxelClusters[j], cam_labels_maj[j])
        colors.append(color)
        color_centroids.append(color[0])
        pers_centroids.append(centroids[j])

    VoxelClusters = np.concatenate(VoxelClusters).tolist()
    colors = np.concatenate(colors).tolist()

    draw_floor(pers_centroids, color_centroids, img_floor, pers_centroids_total, color_centroids_total)

    return VoxelClusters, colors


def draw_floor(positions, colors, img_floor, pers_centroids_total, color_centroids_total):
    """
    Plots the points at the given positions with the specified RGB colors.
    Args:
    positions (list of lists): A 4x2 list of x,y positions for the four points.
    colors (list of lists): A 4x3 list of RGB colors for the four points.
    """
    # Create a blank image with a white background

    # Loop through each point and plot it with its corresponding color
    # Scale the positions to the image size
    positions = [[(x + 100) / 200.0 * 500.0, (y + 100) / 200.0 * 500.0] for [x, y] in positions]

    for i in range(len(positions)):
        x, y = positions[i]
        r, g, b = colors[i]
        color = (int(b), int(g), int(r))
        cv.circle(img_floor, (int(x), int(y)), 5, color, -1)

    # Show the image
    cv.imshow('Draw', img_floor)
    cv.waitKey(1)

    pers_centroids_total.append([positions])
    color_centroids_total.append([colors])

    blue, red, green, pink = [], [], [], []
    # First, let's order the points
    # Loop through the frames
    for j in range(len(pers_centroids_total)):
        # Loop through all the people
        for i in range(4):
            print(pers_centroids_total[j][0][i])
            # Blue
            if color_centroids_total[j][0][i] == [0, 0, 225]:
                blue.append(pers_centroids_total[j][0][i])
            if color_centroids_total[j][0][i] == [0, 255, 0]:
                green.append(pers_centroids_total[j][0][i])
            if color_centroids_total[j][0][i] == [255, 0, 0]:
                red.append(pers_centroids_total[j][0][i])
            if color_centroids_total[j][0][i] == [255, 0, 225]:
                pink.append(pers_centroids_total[j][0][i])

    interpolated_paths(red, pink, blue, green)


def floor_image(path, color, img_floor):
    """
    floor_image makes the cubic interpolation for smoothing the tracking paths
    from each person
    """
    # Your list of 0 points with x and y coordinates
    x = [p[0] for p in path]
    y = [p[1] for p in path]

    # Interpolate the points using a cubic spline interpolation
    f = interp1d(x, y, kind='cubic')
    x_new = np.linspace(x[0], x[-1], num=1000, endpoint=True)
    y_new = f(x_new)
    # Plot the interpolated path on the image
    for i in range(len(x_new) - 1):
        pt1 = (int(x_new[i]), int(y_new[i]))
        pt2 = (int(x_new[i + 1]), int(y_new[i + 1]))
        cv.line(img_floor, pt1, pt2, color, 2)
    # Show the image with the plotted path

    return img_floor


def interpolated_paths(red, pink, blue, green):
    """
    interpolated_paths performs the 2D image which represents the paths that
    take each persons
    """
    # Choose the number of points that you want to plot.
    number = len(red)
    colors_all = [red[:number], pink[:number], blue[:number], green[:number]]
    colors_rgb = [(255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0)]

    if len(blue) > 4:
        # Create an empty image to plot the path
        img_floor = np.ones((500, 500, 3), dtype=np.uint8) * 255

        for i in range(len(colors_all)):
            img_floor = floor_image(colors_all[i], colors_rgb[i], img_floor)

    cv.imshow('Interpolated paths', img_floor)
    cv.waitKey(1)
    cv.waitKey(1)


def create_colormodel(voxels, lookuptable):
    """
    create_colormodel makes the color models in the offline phase
    """
    # Get voxels + labels that should be used for creating a color model (those above mean height)
    labels, centers = cluster(voxels)
    labels = np.ravel(labels)

    voxels = np.float32(voxels)

    cam_color_models = [[], [], [], []]

    # Loop over all cameras
    for i in range(4):
        cam = i + 1
        frame = cv.imread(f'./data/cam{cam}/video.png')

        # Convert to HSV
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        color_models = []
        for label in range(4):

            voxels_person = voxels[labels == label]  # save voxel if the label is same
            pixelCluster, colorCluster = [], []

            # Take only above the belt and cut the head
            tshirt = np.mean(voxels_person[:, 1], dtype=np.int_)
            voxel_roi = voxels_person[:, 1] > tshirt
            voxels_person_roi = voxels_person[voxel_roi]

            head = np.max(voxels_person_roi[:, 1])
            voxel_roi = voxels_person_roi[:, 1] < 3 / 4 * head
            voxels_person_roi = voxels_person_roi[voxel_roi]

            # For each cluster group of labelled voxels
            for v in voxels_person_roi:
                # Find coordinates belonging to voxel
                pixel = lookuptable[i][tuple(v)]
                pixelCluster.append(pixel)

            # Cluster ROI
            roi = np.array([frame[y, x] for [x, y] in pixelCluster])
            roi = np.float32(roi)

            # Create a GMM model
            model = cv.ml.EM_create()
            model.setClustersNumber(3)

            # Create model per person (cluster)
            model.trainEM(roi)

            color_models.append(model)

        cam_color_models[i] = color_models

    return cam_color_models
