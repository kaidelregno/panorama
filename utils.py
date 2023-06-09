import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import os
import random

def load_images(images_folder_path):
    images = []
    for filename in os.listdir(images_folder_path):
        img = cv2.imread(os.path.join(images_folder_path,filename))
        if img is not None:
            images.append(img)
    return images

def run_sift(image, num_features):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=num_features)
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des

def find_sift_correspondences(kp1, des1, kp2, des2, ratio):
    correspondences = []
    for i, kp in enumerate(kp1):
        distances = np.sqrt(np.sum((des2 - des1[i])**2, axis=1))
        sort_order = np.argsort(distances)
        if distances[sort_order[0]] < ratio * distances[sort_order[1]]:
            correspondences.append((kp.pt, kp2[sort_order[0]].pt))

    return correspondences

def pairwise_correspondences(images, ratio = 0.6):
    pw_correspondences = np.empty((len(images), len(images)), dtype=object)
    kps = []
    dess = []
    for image in images:
        kp, des = run_sift(image, 1000)
        kps.append(kp)
        dess.append(des)

    for i in range(len(images)):
        for j in range(i+1, len(images)):
            if i == j:
                pw_correspondences[i][j] = None
            else:
                pw_correspondences[i][j] = find_sift_correspondences(kps[i], des[i], kp[j], des[j], ratio)

    return kps, dess, pw_correspondences

def compute_homography(correspondences):
    A = []
    for correspondence in correspondences:
        point1, point2 = correspondence
        x1, y1 = point1
        x2, y2 = point2
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
    A = np.asarray(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H

def apply_homography(points, homography):
    new_points = []
    for point in points:
        new_point = np.append(point, 1)
        new_point = np.matmul(homography, new_point)
        new_point = new_point / new_point[-1]
        new_points.append(new_point[:-1])

    return new_points

def compute_inliers(homography, correspondences, threshold):
    inliers = []
    outliers = []
    for correspondence in correspondences:
        point1, point2 = correspondence
        point1 = np.append(point1, 1)
        point2 = np.append(point2, 1)
        point2_hat = np.dot(homography, point1)
        point2_hat = point2_hat / point2_hat[-1]
        if np.sqrt(np.sum((point2_hat[:-1]-point2[:-1])**2)) < threshold:
            inliers.append(correspondence)
        else:
            outliers.append(correspondence)

    return inliers, outliers

def ransac(correspondences, num_iterations, num_sampled_points, threshold):
    best_inliers = []
    best_homography = None
    best_outliers = []
    for i in range(num_iterations):
        sampled_correspondences = random.sample(correspondences, num_sampled_points)
        homography = compute_homography(sampled_correspondences)
        inliers, outliers = compute_inliers(homography, correspondences, threshold)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_outliers = outliers
            best_homography = homography

    return best_homography, best_inliers, best_outliers

def pairwise_ransac(pw_correspondences, num_iterations = 50, num_sampled_points = 6, threshold = 3, alpha = 8, beta = 0.3):
    pw_homographies = np.empty((len(pw_correspondences), len(pw_correspondences)), dtype=object)
    pw_inliers = np.empty((len(pw_correspondences), len(pw_correspondences)), dtype=object)
    pw_outliers = np.empty((len(pw_correspondences), len(pw_correspondences)), dtype=object)
    matches = []
    for i in range(pw_correspondences.shape[0]):
        for j in range(i+1, pw_correspondences.shape[0]):
            pw_homographies[i][j], pw_inliers[i][j], pw_outliers[i][j] = ransac(pw_correspondences[i][j], num_iterations, num_sampled_points, threshold)
            if (len(pw_inliers[i][j]) > alpha + beta * len(pw_correspondences[i][j])):
                matches.append((i, j))

    best_matches = np.zeros(pw_correspondences.shape[0])
    for match in matches:
        i, j = match
        if len(pw_inliers[i][j]) > len(pw_inliers[i][best_matches]):
            best_matches[i] = j



    return pw_homographies, pw_inliers, pw_outliers, matches, best_matches

def compute_reprojection_residual(params1, params2, )

def bundle_adjust(pw_inliers, matches, best_matches, max_iter = 100):
    #we assume that the camera rotates about its optical center.
    #we parameterize the homography by focal length and a rotation matrix
    #K = [[f, 0, 0], [0, f, 0], [0, 0, 1]]
    #R = [[0, -r3, r2], [r3, 0, -r1], [-r2, r1, 0]]

    params = np.zeros(4*pw_inliers.shape[0])
    for step in range(max_iter):
        JTJ = np.zeros((4*pw_inliers.shape[0], 4*pw_inliers.shape[0]))
        JTr = np.zeros(4*pw_inliers.shape[0])
        for match in matches:
            i,j = match
            for correspondence in pw_inliers[i][j]:
                point1, point2 = correspondence
                
            
