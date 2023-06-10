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
                pw_correspondences[i][j] = find_sift_correspondences(kps[i], dess[i], kps[j], dess[j], ratio)

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
    if len(correspondences) < num_sampled_points:
        return None, [], []
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


    return pw_homographies, pw_inliers, pw_outliers, matches


def bundle_adjust(pw_inliers, matches, max_iter = 100):
    #we assume that the camera rotates about its optical center.
    #we parameterize the homography by focal length and a rotation matrix
    #K = [[f, 0, 0], [0, f, 0], [0, 0, 1]]
    #R = [[0, -r3, r2], [r3, 0, -r1], [-r2, r1, 0]]

    params = np.ones(4*pw_inliers.shape[0])
    trust = 0.1
    Cinv = np.zeros((4*pw_inliers.shape[0], 4*pw_inliers.shape[0]))
    error = np.inf
    #initialize parameters
    for i in range(pw_inliers.shape[0]):
        Cinv[4*i:4*i+4, 4*i:4*i+4] = np.diag([256/(np.pi**2), 256/(np.pi**2), 256/(np.pi**2), 100])


    #levenberg-marquardt loop
    for step in range(max_iter):
        JTJ = np.zeros((4*pw_inliers.shape[0], 4*pw_inliers.shape[0]))
        JTr = np.zeros(4*pw_inliers.shape[0])

        error_prev = error
        error = 0
        for match in matches:
            i,j = match

            fi = params[4*i]
            fj = params[4*j]
            ri1 = params[4*i+1]
            ri2 = params[4*i+2]
            ri3 = params[4*i+3]
            rj1 = params[4*j+1]
            rj2 = params[4*j+2]
            rj3 = params[4*j+3]

            Ki = np.array([[fi, 0, 0], [0, fi, 0], [0, 0, 1]])
            Kj = np.array([[fj, 0, 0], [0, fj, 0], [0, 0, 1]])
            Ri = np.array([[0, -ri3, ri2], [ri3, 0, -ri1], [-ri2, ri1, 0]])
            Rj = np.array([[0, -rj3, rj2], [rj3, 0, -rj1], [-rj2, rj1, 0]])

            for correspondence in pw_inliers[i][j]:
                point1, point2 = correspondence
                xk, yk = point1
                xl, yl = point2

                dfik = np.array([-(ri3*rj1 - (xl*(ri2*rj2 + ri3*rj3))/fj + (ri2*rj1*yl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj), \
                                -(ri3*rj2 - (yl*(ri1*rj1 + ri3*rj3))/fj + (ri1*rj2*xl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)])

                dfjk = np.array([(((ri1*rj3*xl)/fj^2 + (ri2*rj3*yl)/fj^2)*(fi*ri3*rj1 - (xl*(fi*ri2*rj2 + fi*ri3*rj3))/fj + (fi*ri2*rj1*yl)/fj))/(ri1*rj1 + ri2*rj2 \
                                 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2 - ((xl*(fi*ri2*rj2 + fi*ri3*rj3))/fj^2 - (fi*ri2*rj1*yl)/fj^2)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj), \
                                (((ri1*rj3*xl)/fj^2 + (ri2*rj3*yl)/fj^2)*(fi*ri3*rj2 - (yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj + (fi*ri1*rj2*xl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - \
                                 (ri2*rj3*yl)/fj)^2 - ((yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj^2 - (fi*ri1*rj2*xl)/fj^2)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)])\
                
                dri1k = np.array([((rj1 - (rj3*xl)/fj)*(fi*ri3*rj1 - (xl*(fi*ri2*rj2 + fi*ri3*rj3))/fj + (fi*ri2*rj1*yl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2, \
                                ((rj1 - (rj3*xl)/fj)*(fi*ri3*rj2 - (yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj + (fi*ri1*rj2*xl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2 - \
                                     ((fi*rj2*xl)/fj - (fi*rj1*yl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)])

                dri2k = np.array([((fi*rj2*xl)/fj - (fi*rj1*yl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj) + ((rj2 - (rj3*yl)/fj)*(fi*ri3*rj1 - (xl*(fi*ri2*rj2 + \
                                 fi*ri3*rj3))/fj + (fi*ri2*rj1*yl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2, \
                                ((rj2 - (rj3*yl)/fj)*(fi*ri3*rj2 - (yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj + (fi*ri1*rj2*xl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2])

                dri3k = np.array([-(fi*rj1 - (fi*rj3*xl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj), \
                                -(fi*rj2 - (fi*rj3*yl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)])

                drj1k = np.array([(ri1*(fi*ri3*rj1 - (xl*(fi*ri2*rj2 + fi*ri3*rj3))/fj + (fi*ri2*rj1*yl)/fj))/(ri1*rj1 + \
                                 ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2 - (fi*ri3 + (fi*ri2*yl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj), \
                                (ri1*(fi*ri3*rj2 - (yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj + (fi*ri1*rj2*xl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2 + \
                                     (fi*ri1*yl)/(fj*(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj))])

                drj2k = np.array([(ri2*(fi*ri3*rj1 - (xl*(fi*ri2*rj2 + fi*ri3*rj3))/fj + (fi*ri2*rj1*yl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2 + \
                                (fi*ri2*xl)/(fj*(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)), \
                                (ri2*(fi*ri3*rj2 - (yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj + (fi*ri1*rj2*xl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2 - \
                                    (fi*ri3 + (fi*ri1*xl)/fj)/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)])

                drj3k = np.array([(fi*ri3*xl)/(fj*(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)) - (((ri1*xl)/fj + (ri2*yl)/fj)*(fi*ri3*rj1 - \
                                 (xl*(fi*ri2*rj2 + fi*ri3*rj3))/fj + (fi*ri2*rj1*yl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2, \
                                (fi*ri3*yl)/(fj*(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)) - (((ri1*xl)/fj + (ri2*yl)/fj)*(fi*ri3*rj2 - \
                                 (yl*(fi*ri1*rj1 + fi*ri3*rj3))/fj + (fi*ri1*rj2*xl)/fj))/(ri1*rj1 + ri2*rj2 - (ri1*rj3*xl)/fj - (ri2*rj3*yl)/fj)^2])

                dfil = np.array([(((ri3*rj1*xk)/fi^2 + (ri3*rj2*yk)/fi^2)*(fj*ri1*rj3 - (xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi + (fj*ri1*rj2*yk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi \
                                - (ri3*rj2*yk)/fi)^2 - ((xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi^2 - (fj*ri1*rj2*yk)/fi^2)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi), \
                                (((ri3*rj1*xk)/fi^2 + (ri3*rj2*yk)/fi^2)*(fj*ri2*rj3 - (yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi + (fj*ri2*rj1*xk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - \
                                 (ri3*rj2*yk)/fi)^2 - ((yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi^2 - (fj*ri2*rj1*xk)/fi^2)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)])

                dfjl = np.array([-(ri1*rj3 - (xk*(ri2*rj2 + ri3*rj3))/fi + (ri1*rj2*yk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi), \
                                -(ri2*rj3 - (yk*(ri1*rj1 + ri3*rj3))/fi + (ri2*rj1*xk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)])

                dri1l = np.array([(rj1*(fj*ri1*rj3 - (xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi + (fj*ri1*rj2*yk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - \
                                (ri3*rj2*yk)/fi)^2 - (fj*rj3 + (fj*rj2*yk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi), \
                                (rj1*(fj*ri2*rj3 - (yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi + (fj*ri2*rj1*xk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - \
                                    (ri3*rj2*yk)/fi)^2 +  (fj*rj1*yk)/(fi*(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi))])
                                
                dri2l = np.array([(rj2*(fj*ri1*rj3 - (xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi + (fj*ri1*rj2*yk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - \
                                 (ri3*rj2*yk)/fi)^2 + (fj*rj2*xk)/(fi*(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)), \
                                (rj2*(fj*ri2*rj3 - (yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi + (fj*ri2*rj1*xk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - \
                                    (ri3*rj2*yk)/fi)^2 - (fj*rj3 + (fj*rj1*xk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)])

                dri3l = np.array([(fj*rj3*xk)/(fi*(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)) - (((rj1*xk)/fi + (rj2*yk)/fi)*(fj*ri1*rj3 - \
                                (xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi + (fj*ri1*rj2*yk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)^2, \
                                (fj*rj3*yk)/(fi*(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)) - (((rj1*xk)/fi + (rj2*yk)/fi)*(fj*ri2*rj3 - \
                                    (yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi + (fj*ri2*rj1*xk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)^2])

                drj1l = np.array([((ri1 - (ri3*xk)/fi)*(fj*ri1*rj3 - (xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi + (fj*ri1*rj2*yk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)^2, \
                                ((ri1 - (ri3*xk)/fi)*(fj*ri2*rj3 - (yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi + (fj*ri2*rj1*xk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)^2 - \
                                     ((fj*ri2*xk)/fi - (fj*ri1*yk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)])

                drj2l = np.array([((fj*ri2*xk)/fi - (fj*ri1*yk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi) + ((ri2 - (ri3*yk)/fi)*(fj*ri1*rj3 - \
                                (xk*(fj*ri2*rj2 + fj*ri3*rj3))/fi + (fj*ri1*rj2*yk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)^2, \
                                ((ri2 - (ri3*yk)/fi)*(fj*ri2*rj3 - (yk*(fj*ri1*rj1 + fj*ri3*rj3))/fi + (fj*ri2*rj1*xk)/fi))/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)^2]) 
                
                drj3l = np.array([-(fj*ri1 - (fj*ri3*xk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi), \
                                -(fj*ri2 - (fj*ri3*yk)/fi)/(ri1*rj1 + ri2*rj2 - (ri3*rj1*xk)/fi - (ri3*rj2*yk)/fi)])

                pijk = np.matmul(Ki, np.matmul(Ri, np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1)))))
                pijl = np.matmul(Kj, np.matmul(Rj, np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1)))))

                residual1 = point1 - pijk[:-1] / pijk[-1]
                residual2 = point2 - pijl[:-1] / pijl[-1]



                dik = np.hstack((dfik.reshape([2,1]), dri1k.reshape([2,1]), dri2k.reshape([2,1]), dri3k.reshape([2,1])))
                djk = np.hstack((dfjk.reshape([2,1]), drj1k.reshape([2,1]), drj2k.reshape([2,1]), drj3k.reshape([2,1])))
                dil = np.hstack((dfil.reshape([2,1]), dri1l.reshape([2,1]), dri2l.reshape([2,1]), dri3l.reshape([2,1])))
                djl = np.hstack((dfjl.reshape([2,1]), drj1l.reshape([2,1]), drj2l.reshape([2,1]), drj3l.reshape([2,1])))

                JTJ[4*i:4*i+4, 4*j:4*j+4] += np.matmul(dik.T, djk)
                JTJ[4*j:4*j+4, 4*i:4*i+4] += np.matmul(djl.T, dil)

                JTr[4*i:4*i+4] += np.matmul(dik.T, residual1)
                JTr[4*j:4*j+4] += np.matmul(djl.T, residual2)

                error += np.sum(residual1**2) + np.sum(residual2**2)
    
        params_prev = params
        params = np.linalg.solve(JTJ + trust * Cinv, JTr)
        for i in range(pw_inliers.shape[0]):
            fmean += params[4*i]
        fmean /= pw_inliers.shape[0]

        for i in range(pw_inliers.shape[0]):
            Cinv[4*i:4*i+4, 4*i:4*i+4] = np.diag([256/(np.pi**2), 256/(np.pi**2), 256/(np.pi**2), 100/(fmean **2)])

        if error < error_prev:
            trust = 0.8 * trust
        else:
            trust = 2 * trust
            params = params_prev

        if error - error_prev < 0.01:
            break

    return params

        
        


                                
                

                
                
            
