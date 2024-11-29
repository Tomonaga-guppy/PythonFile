import numpy as np

class Camera():
    def __init__(self, id=None):
        """
        :param id: camera identification number
        :type id: unknown or int
        """
        self.K = np.eye(3)  # camera intrinsic parameters
        self.Kundistortion = np.array([])  # could be altered based on K using set_undistorted_view(alpha)
        #  to get undistorted image with all / corner pixels visible
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.kappa = np.zeros((2,))
        self.id = id
        self.size_px = np.zeros((2,))
        self.update_P()

    def update_P(self):
        """
        Update camera P matrix from K, R and t.
        """
        self.P = self.K.dot(np.hstack((self.R, self.t)))

    def set_K(self, K):
        """
        Set K and update P.
        :param K: intrinsic camera parameters
        :type K: numpy.ndarray, shape=(3, 3)
        """
        self.K = K
        self.update_P()

    def set_R(self, R):
        """
        Set camera extrinsic parameters and updates P.
        :param R: camera extrinsic parameters matrix
        :type R: numpy.ndarray, shape=(3, 3)
        """
        self.R = R
        self.update_P()

    def set_t(self, t):
        """
        Set camera translation and updates P.
        :param t: camera translation vector
        :type t: numpy.ndarray, shape=(3, 1)
        """
        self.t = t
        self.update_P()
