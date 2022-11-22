import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt')
        self.gt_poses = self._load_poses(data_dir + '/poses.txt')
        self.images_l = self._load_images(data_dir + '/image_l')
        self.images_r = self._load_images(data_dir + '/image_r')

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=60, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.f_l, self.b = self.get_focal_and_baseline(self.P_l, self.P_r)

    def get_focal_and_baseline(self, Pl, PR):
        k, _, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(Pl)
        tl = (tl / tl[3])[:3]
        _, _, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(PR)
        tr = (tr / tr[3])[:3]
        return k[0][0], tr[0]-tl[0]
    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    def get_matches(self, i, images_l_or_r):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(images_l_or_r[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(images_l_or_r[i], None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        # plt.hist(q1[:,0], bins = q1.shape[0])
        # plt.show()
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        # img3 = cv2.drawMatches(images_l_or_r[i], kp1, images_l_or_r[i-1],kp2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # cv2.waitKey(200)
        return q1, q2
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def calc_3d(self, q1_l, depth1, q2_l, depth2):
        cx = self.K_l[0, 2]
        cy = self.K_l[1, 2]
        fx = self.K_l[0, 0]
        fy = self.K_l[1, 1]
        def get_3d(q, depth):
            Q = np.zeros((0, 3))
            for i in range(q.shape[0]):
                # num of interest points
                u = q[i, 0]
                v = q[i, 1]
                z = depth[int(v), int(u)]
                x = z*(u-cx)/fx
                y = z*(v-cy)/fy
                # print(x.shape, y.shape, z.shape)
                Q = np.vstack((Q, np.array([x, y, z])))
            return Q
        Q1 = get_3d(q1_l, depth1)
        Q2 = get_3d(q2_l, depth2)
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 100
        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose N random feature points
            N = 6
            sample_idx = np.random.choice(range(q1.shape[0]), N)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))
            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
                print(error)
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix
    def get_depth_map(self, disparity_map):
        disparity_map[disparity_map == 0.0] = 0.1
        disparity_map[disparity_map == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth_map = np.ones(disparity_map.shape)
        depth_map = self.f_l * self.b / disparity_map
        depth_map[depth_map>3000] = 50
        depth_map[depth_map == 0] = 50
        # plt.imshow(depth_map)
        # plt.show()
        # plt.hist(depth_map.flatten())
        # plt.show()
        return depth_map
    def estimate_pose_PnP(self, q2_l, Q1):
        # The SolvePnPRansac() function computes a pose that relates points in the global
        # coordinate frame to the camera's pose. See the jupyter notebook for details. Just remember to inv here.
        _, r, t, _ = cv2.solvePnPRansac(Q1, q2_l, self.K_l, None, iterationsCount = 100)
        R, _ = cv2.Rodrigues(r)
        t = t[:, 0]
        transformation_matrix = self._form_transf(R, t)
        transformation_matrix = np.linalg.inv(transformation_matrix)
        return transformation_matrix
    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        q1_l, q2_l = self.get_matches(i, self.images_l)
        # q1_r, q2_r = self.get_matches(i, self.images_r)

        disp_1 = self.disparities[-1]
        disp_2 = np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16)
        self.disparities.append(disp_2)

        depth_map1 = self.get_depth_map(disp_1)
        depth_map2 = self.get_depth_map(disp_2)
        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(q1_l, depth_map1, q2_l, depth_map2)

        # Calculate 3D points From disparity map
        # Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3), use q1_l
        # Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3), use q2_l



        # Estimate the transformation matrix
        # transformation_matrix = self.estimate_pose(q1_l, q2_l, Q1, Q2)
        transformation_matrix = self.estimate_pose_PnP(q2_l, Q1)
        return transformation_matrix

    # def local_bundle_adjustment(self, num_prev_frames, )

def main():
    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2
    vo = VisualOdometry(data_dir)

    # play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i < 1:
            cur_pose = gt_pose
        else:
            transf = vo.get_pose(i)
            cur_pose = np.matmul(cur_pose, transf)
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
                             file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
