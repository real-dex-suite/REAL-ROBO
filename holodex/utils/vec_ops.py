import numpy as np
import cv2

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def moving_average(vector, moving_average_queue, limit):
    moving_average_queue.append(vector)

    if len(moving_average_queue) > limit:
        moving_average_queue.pop(0)

    mean_vector = np.mean(moving_average_queue, axis = 0)
    return mean_vector

def get_distance(start_vector, end_vector):
    return np.linalg.norm(end_vector - start_vector)

def linear_transform(curr_val, source_bound, target_bound):
    multiplier = (target_bound[-1] - target_bound[0]) / (source_bound[-1] - source_bound[0])
    target_val = ((curr_val - source_bound[0]) * multiplier) + target_bound[0]
    return target_val

def persperctive_transform(input_coordinates, given_bound, target_bound):
    transformation_matrix = cv2.getPerspectiveTransform(np.float32(given_bound), np.float32(target_bound))
    transformed_coordinate = np.matmul(np.array(transformation_matrix), np.array([input_coordinates[0], input_coordinates[1], 1]))
    transformed_coordinate = transformed_coordinate / transformed_coordinate[-1]

    return transformed_coordinate[0], transformed_coordinate[1]

def calculate_angle(coord_1, coord_2, coord_3):
    vector_1 = coord_2 - coord_1
    vector_2 = coord_3 - coord_2

    inner_product = np.inner(vector_1, vector_2)
    norm = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    angle = np.arccos(inner_product / norm)
    return angle

def coord_in_bound(bound, coord):
    return cv2.pointPolygonTest(np.float32(bound), np.float32(coord), False)

def best_fit_transform(A, B):
        """
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        """

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t


def check_best_fit_transform():
    from scipy.spatial.transform import Rotation as R
    random_rotation = R.random().as_matrix()
    random_translation = np.random.rand(3)
    random_transform = np.eye(4)
    random_transform[:3, :3] = random_rotation
    random_transform[:3, 3] = random_translation
    test_data = np.random.rand(100, 3)
    transformed_data = (random_transform @ np.hstack([test_data, np.ones((test_data.shape[0], 1))]).T).T[:, :3]
    transform, _, _ = best_fit_transform(test_data, transformed_data)
    return transform


if __name__ == "__main__":
    check_best_fit_transform()