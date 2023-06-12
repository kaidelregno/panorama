import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import os
import random
import torch.nn.functional as F
import torch
import scipy.optimize as opt

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
            correspondences.append(np.array([kp.pt, kp2[sort_order[0]].pt]))

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

def bundle_adjust_torch(pw_inliers, matches, max_iter = 100):

    params = torch.ones(4*pw_inliers.shape[0], requires_grad=True)
    trust = 0.1
    Cinv = torch.zeros((4*pw_inliers.shape[0], 4*pw_inliers.shape[0]))
    error = np.inf
    for i in range(pw_inliers.shape[0]):
        Cinv[4*i:4*i+4, 4*i:4*i+4] = torch.diag(torch.tensor([100, 256/(np.pi**2), 256/(np.pi**2), 256/(np.pi**2)]))
    
    for step in range(max_iter):
        JTJ = torch.zeros((4*pw_inliers.shape[0], 4*pw_inliers.shape[0]))
        JTr = torch.zeros(4*pw_inliers.shape[0])

        error_prev = error
        error = 0
        fmean = 0
        num_correspondences = 0
        params_prev = params.clone()

        for match in matches:
            i,j = match

            Ki = torch.tensor([[params[4*i], 0, 0], [0, params[4*i], 0], [0, 0, 1]], requires_grad=True)
            Kj = torch.tensor([[params[4*j], 0, 0], [0, params[4*j], 0], [0, 0, 1]], requires_grad=True)
            Ri = torch.tensor([[0, -params[4*i+3], params[4*i+2]], [params[4*i+3], 0, -params[4*i+1]], [-params[4*i+2], params[4*i+1], 0]], requires_grad=True)
            Rj = torch.tensor([[0, -params[4*j+3], params[4*j+2]], [params[4*j+3], 0, -params[4*j+1]], [-params[4*j+2], params[4*j+1], 0]], requires_grad=True)
            for correspondence in pw_inliers[i][j]:
                point1, point2 = correspondence
                point1 = torch.tensor(point1, requires_grad=True)
                point2 = torch.tensor(point2, requires_grad=True)

                pijk = torch.matmul(Ki, torch.matmul(Ri, torch.matmul(Rj.T, torch.matmul(torch.inverse(Kj), torch.cat((point2, torch.tensor([1])))))))
                pijl = torch.matmul(Kj, torch.matmul(Rj, torch.matmul(Ri.T, torch.matmul(torch.inverse(Ki), torch.cat((point1, torch.tensor([1])))))))

                pijk = (pijk[:-1] / pijk[-1]).reshape([2,1])
                pijl = (pijl[:-1] / pijl[-1]).reshape([2,1])

                pijk.backward()
                dik = torch.cat((params[4*i].grad, params[4*i+1].grad, params[4*i+2].grad, params[4*i+3].grad), dim = 1)
                djk = torch.cat((params[4*j].grad, params[4*j+1].grad, params[4*j+2].grad, params[4*j+3].grad), dim = 1)

                params.grad.zero_()
                pijk.grad.zero_()
                pijl.grad.zero_()
                Ki.grad.zero_()
                Kj.grad.zero_()
                Ri.grad.zero_()
                Rj.grad.zero_()

                pijl.backward()
                dil = torch.cat((params[4*i].grad, params[4*i+1].grad, params[4*i+2].grad, params[4*i+3].grad), dim = 1)
                djl = torch.cat((params[4*j].grad, params[4*j+1].grad, params[4*j+2].grad, params[4*j+3].grad), dim = 1)

                params.grad.zero_()
                pijk.grad.zero_()
                pijl.grad.zero_()
                Ki.grad.zero_()
                Kj.grad.zero_()
                Ri.grad.zero_()
                Rj.grad.zero_()

                JTJ[4*i:4*i+4, 4*i:4*i+4] += torch.matmul(dik.T, djk)
                JTJ[4*i:4*i+4, 4*j:4*j+4] += torch.matmul(dil.T, djl)

                JTr[4*i:4*i+4] += torch.matmul(-dik.T, pijk)
                JTr[4*j:4*j+4] += torch.matmul(-dil.T, pijl)

                error += torch.sum((pijk - point1)**2) + torch.sum((pijl - point2)**2)
                num_correspondences += 2

        error = error / num_correspondences
        if error < error_prev:
            trust = 0.8 * trust
            params = torch.linalg.solve(JTJ + trust * Cinv, JTr)
        else:
            trust = 2 * trust
            params = params_prev

        if abs(error - error_prev) < 1e-5:
            print("Converged at step ", step)
            break

        
        for i in range(pw_inliers.shape[0]):
            fmean += params[4*i]
        fmean /= pw_inliers.shape[0]

        for i in range(pw_inliers.shape[0]):
            Cinv[4*i:4*i+4, 4*i:4*i+4] = np.diag([100/(fmean **2), 256/(np.pi**2), 256/(np.pi**2), 256/(np.pi**2)])

        print("Step %d: Average reprojection error = %f" % (step, error / num_correspondences))

    return params





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
        Cinv[4*i:4*i+4, 4*i:4*i+4] = np.diag([100, 256/(np.pi**2), 256/(np.pi**2), 256/(np.pi**2)])
    



    #levenberg-marquardt loop
    for step in range(max_iter):
        JTJ = np.zeros((4*pw_inliers.shape[0], 4*pw_inliers.shape[0]))
        JTr = np.zeros(4*pw_inliers.shape[0])

        error_prev = error
        error = 0
        fmean = 0
        num_correspondences = 0
        dfik_err = 0
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

            thetai = np.sqrt(ri1**2 + ri2**2 + ri3**2)
            thetaj = np.sqrt(rj1**2 + rj2**2 + rj3**2)

            Ri = np.eye(3) + np.sin(thetai)/thetai * Ri + (1-np.cos(thetai))/(thetai**2) * np.matmul(Ri, Ri)
            Rj = np.eye(3) + np.sin(thetaj)/thetaj * Rj + (1-np.cos(thetaj))/(thetaj**2) * np.matmul(Rj, Rj)

            for correspondence in pw_inliers[i][j]:
                point1, point2 = correspondence
                xk, yk = point1
                xl, yl = point2

                pijk = np.matmul(Ki, np.matmul(Ri, np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1)))))
                pijl = np.matmul(Kj, np.matmul(Rj, np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1)))))

                dhomok = np.array([[1/pijk[-1], 0, -pijk[0]/(pijk[-1]**2)], [0, 1/pijk[-1], -pijk[1]/(pijk[-1]**2)]])
                dhomol = np.array([[1/pijl[-1], 0, -pijl[0]/(pijl[-1]**2)], [0, 1/pijl[-1], -pijl[1]/(pijl[-1]**2)]])

                dfik = np.matmul(dhomok, np.matmul(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]), np.matmul(Ri, np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))
                dfjk = np.matmul(dhomok, np.matmul(Ki, np.matmul(Ri, np.matmul(Rj.T, np.matmul(np.array([[-1/(fj**2), 0, 0], [0, -1/fj**2, 0], [0, 0, 0]]), np.append(point2, 1))))))
                dfjl = np.matmul(dhomol, np.matmul(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]), np.matmul(Rj, np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))
                dfil = np.matmul(dhomol, np.matmul(Kj, np.matmul(Rj, np.matmul(Ri.T, np.matmul(np.array([[-1/(fi**2), 0, 0], [0, -1/fi**2, 0], [0, 0, 0]]), np.append(point1, 1))))))

                dri1k = np.matmul(dhomok, np.matmul(Ki, np.matmul(np.matmul(Ri, np.array([[0,0,0],[0,0,-1],[0,1,0]])), np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))
                dri2k = np.matmul(dhomok, np.matmul(Ki, np.matmul(np.matmul(Ri, np.array([[0,0,1],[0,0,0],[-1,0,0]])), np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))
                dri3k = np.matmul(dhomok, np.matmul(Ki, np.matmul(np.matmul(Ri, np.array([[0,-1,0],[1,0,0],[0,0,0]])), np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))
                drj1k = np.matmul(dhomok, np.matmul(Ki, np.matmul(Ri, np.matmul(np.matmul(Rj.T, np.array([[0,0,0],[0,0,1],[0,-1,0]])), np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))
                drj2k = np.matmul(dhomok, np.matmul(Ki, np.matmul(Ri, np.matmul(np.matmul(Rj.T, np.array([[0,0,-1],[0,0,0],[1,0,0]])), np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))
                drj3k = np.matmul(dhomok, np.matmul(Ki, np.matmul(Ri, np.matmul(np.matmul(Rj.T, np.array([[0,1,0],[-1,0,0],[0,0,0]])), np.matmul(np.linalg.inv(Kj), np.append(point2, 1))))))

                dri1l = np.matmul(dhomol, np.matmul(Kj, np.matmul(np.matmul(Rj, np.array([[0,0,0],[0,0,-1],[0,1,0]])), np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))
                dri2l = np.matmul(dhomol, np.matmul(Kj, np.matmul(np.matmul(Rj, np.array([[0,0,1],[0,0,0],[-1,0,0]])), np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))
                dri3l = np.matmul(dhomol, np.matmul(Kj, np.matmul(np.matmul(Rj, np.array([[0,-1,0],[1,0,0],[0,0,0]])), np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))
                drj1l = np.matmul(dhomol, np.matmul(Kj, np.matmul(Rj, np.matmul(np.matmul(Ri.T, np.array([[0,0,0],[0,0,1],[0,-1,0]])), np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))
                drj2l = np.matmul(dhomol, np.matmul(Kj, np.matmul(Rj, np.matmul(np.matmul(Ri.T, np.array([[0,0,-1],[0,0,0],[1,0,0]])), np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))
                drj3l = np.matmul(dhomol, np.matmul(Kj, np.matmul(Rj, np.matmul(np.matmul(Ri.T, np.array([[0,1,0],[-1,0,0],[0,0,0]])), np.matmul(np.linalg.inv(Ki), np.append(point1, 1))))))

                pijk_num = np.matmul(Ki, np.matmul(np.array([[0,0,0], [0, 0, 1.01], [0, 1.01, 0]])*Ri, np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1)))))
                pijk_num = pijk_num[:-1] / pijk_num[-1]
                dri1k_num = (pijk_num - pijk[:-1] / pijk[-1]) / 0.01

                dfik_err += np.sum(dri1k-dri1k_num[:-1])
                # print("dfik: ", dfik)
                # print("dfik_num: ", dfik_num)

                residual1 = point1 - pijk[:-1] / pijk[-1]
                residual2 = point2 - pijl[:-1] / pijl[-1]



                dik = np.hstack((dfik.reshape([2,1]), dri1k.reshape([2,1]), dri2k.reshape([2,1]), dri3k.reshape([2,1])))
                djk = np.hstack((dfjk.reshape([2,1]), drj1k.reshape([2,1]), drj2k.reshape([2,1]), drj3k.reshape([2,1])))
                dil = np.hstack((dfil.reshape([2,1]), dri1l.reshape([2,1]), dri2l.reshape([2,1]), dri3l.reshape([2,1])))
                djl = np.hstack((dfjl.reshape([2,1]), drj1l.reshape([2,1]), drj2l.reshape([2,1]), drj3l.reshape([2,1])))

                JTJ[4*i:4*i+4, 4*j:4*j+4] += np.matmul(dik.T, djk)
                JTJ[4*j:4*j+4, 4*i:4*i+4] += np.matmul(djl.T, dil)

                JTr[4*i:4*i+4] += np.matmul(-dik.T, residual1)
                JTr[4*j:4*j+4] += np.matmul(-djl.T, residual2)

                error += np.sum(residual1**2) + np.sum(residual2**2)
                num_correspondences += 2
    
        params_prev = params
        params = np.linalg.solve(JTJ + trust * Cinv, JTr)
        for i in range(pw_inliers.shape[0]):
            fmean += params[4*i]
        fmean /= pw_inliers.shape[0]

        for i in range(pw_inliers.shape[0]):
            Cinv[4*i:4*i+4, 4*i:4*i+4] = np.diag([100/(fmean **2), 256/(np.pi**2), 256/(np.pi**2), 256/(np.pi**2)])

        if error < error_prev:
            trust = 0.8 * trust
        else:
            trust = 2 * trust
            params = params_prev

        if abs(error - error_prev) < 0.01:
            print("Converged at step: ", step)
            break

        print("Step %d: Average reprojection error = %f" % (step, error / num_correspondences))
        print("step %d: dfik_err = %f" % (step, dfik_err))

    return params

import jaxopt

def jax_residual(params, pw_inliers, matches):
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

        thetai = np.sqrt(ri1**2 + ri2**2 + ri3**2)
        thetaj = np.sqrt(rj1**2 + rj2**2 + rj3**2)

        Ri = np.eye(3) + np.sin(thetai)/thetai * Ri + (1-np.cos(thetai))/(thetai**2) * np.matmul(Ri, Ri)
        Rj = np.eye(3) + np.sin(thetaj)/thetaj * Rj + (1-np.cos(thetaj))/(thetaj**2) * np.matmul(Rj, Rj)

        for correspondence in pw_inliers[i][j]:
            point1, point2 = correspondence

            pijk = np.matmul(Ki, np.matmul(Ri, np.matmul(Rj.T, np.matmul(np.linalg.inv(Kj), np.append(point2, 1)))))
            pijl = np.matmul(Kj, np.matmul(Rj, np.matmul(Ri.T, np.matmul(np.linalg.inv(Ki), np.append(point1, 1)))))

            residual1 = point1 - pijk[:-1] / pijk[-1]
            residual2 = point2 - pijl[:-1] / pijl[-1]

            error += np.sum(residual1**2) + np.sum(residual2**2)

    return error

def bundle_adjust_jax(pw_inliers, matches, max_iter = 100):
    
    params = jaxopt.LevenbergMarquardt(jax_residual(params, pw_inliers, matches), maxiter=max_iter, verbose=True)


        




        
        


                                
                

                
                
            
