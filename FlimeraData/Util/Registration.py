"""
Module that handles the registration of FlimeraData data.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import match_descriptors, ORB, plot_matched_features
from skimage.measure import ransac
import skimage.transform
from loguru import logger

class Scaling:
    """Represents a scaling to be applied to a 2D image."""
    def __init__(self, scale_factor_x: float, scale_factor_y: float):
        """
        :param scale_factor_x: the scale factor along the x-axis.
        :param scale_factor_y: the scale factor along the y-axis.
        """
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y

class Translation:
    """Represents a translation to be applied to a 2D image."""
    def __init__(self, translation_x: float | int, translation_y: float | int):
        """
        :param translation_x: the translation along the x-axis, units pixels.
        :param translation_y: the translation along the y-axis, units pixels.
        """
        self.translation_x = translation_x
        self.translation_y = translation_y

class Rotation:
    """Represents a rotation to be applied to a 2D image."""
    def __init__(self, rotation: float):
        """
        :param rotation: the clockwise rotation to be applied. Units radians.
        """
        self.rotate_radians = rotation

class Shear:
    """Represents a shear to be applied to a 2D image."""
    def __init__(self, shear_x: float, shear_y: float):
        """
        :param shear_x: the clockwise shear angle for x.
        :param shear_y: the clockwise shear angle for y.
        """
        self.shear_x = shear_x
        self.shear_y = shear_y

def _calculate_rotation(matrix: np.ndarray) -> Rotation:
    """Calculate the rotation from a homogenous affine transform matrix."""
    return Rotation(np.atan2(matrix[1, 0], matrix[0, 0]))

def _calculate_scaling(matrix: np.ndarray) -> Scaling:
    """Calculate the scaling from a homogenous affine transform matrix."""
    rotation = _calculate_rotation(matrix)
    msy = matrix[0, 1] * np.cos(rotation.rotate_radians) + matrix[1, 1] * np.sin(rotation.rotate_radians)
    scale_factor_y = (msy * np.cos(rotation.rotate_radians) - matrix[0, 1]) / np.sin(rotation.rotate_radians)
    scale_factor_x = np.sqrt(np.sum(matrix ** 2, 0))[0]
    return Scaling(scale_factor_x=scale_factor_x, scale_factor_y=scale_factor_y)

def _calculate_translation(matrix: np.ndarray) -> Translation:
    """Calculate the translation from a homogenous affine transform matrix."""
    translation_x = matrix[0, 2]
    translation_y = matrix[1, 2]
    return Translation(translation_x=float(translation_x), translation_y=float(translation_y))

def _calculate_shear(matrix: np.ndarray) -> Shear:
    """Calculate the shearing from a homogenous affine transform matrix"""
    rotation = _calculate_rotation(matrix)
    shear = (np.atan2(-matrix[0, 1], matrix[1, 1]) - rotation.rotate_radians)
    return Shear(shear_x=shear, shear_y=0)

class AffineTransform:
    """
    Describes a 2D Affine Transform.

    We work in the homogenous coordinate system, allowing the transformation to be applied using a single matrix
    operation.

    This is designed to be immutable, so once the transform is created, its attributes cannot be changed.
    """
    def __init__(self, scaling: Scaling, translation: Translation, rotation: Rotation, shear: Shear):
        # First we populate the matrix
        self._matrix = self._calculate_homogenous_matrix(scaling=scaling, translation=translation, rotation=rotation, shear=shear)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """
        Construct an AffineTransform by supplying the matrix.

        :param matrix: the 3x3 Affine transform matrix, in homogenous coordinates.
        :return: the corresponding transform.
        """
        rotation = _calculate_rotation(matrix)
        scaling = _calculate_scaling(matrix)
        translation = _calculate_translation(matrix)
        shear = _calculate_shear(matrix)

        return cls(scaling=scaling,
                   translation=translation,
                   rotation=rotation,
                   shear=shear)

    #------------------------------------------------ Private properties -----------------------------------------------
    @property
    def matrix(self) -> np.ndarray:
        """Query the current homogenous transform matrix for the affine transform."""
        return self._matrix

    @property
    def rotation(self) -> Rotation:
        """Query the rotation."""
        return _calculate_rotation(matrix=self.matrix)

    @property
    def scaling(self) -> Scaling:
        """Query the scaling"""
        return _calculate_scaling(matrix=self.matrix)

    @property
    def translation(self) -> Translation:
        """Query the translation"""
        return _calculate_translation(matrix=self.matrix)

    @property
    def shear(self) -> Shear:
        """Query the shear."""
        return _calculate_shear(matrix=self.matrix)

    @property
    def inverse_transform(self):
        """Calculate the inverse transform."""
        return self.from_matrix(matrix=np.linalg.inv(self.matrix))

    #------------------------------------------------ Private methods --------------------------------------------------
    @staticmethod
    def _calculate_homogenous_matrix(scaling: Scaling,
                                     rotation: Rotation,
                                     translation: Translation,
                                     shear: Shear) -> np.ndarray:
        """
        Calculate the transform matrix for a homogenous coordinate system.

        :return: the transform matrix [[a00, a01, b0],[a10, a11, b1],[0, 0, 1]]
        """
        a00 = scaling.scale_factor_x * (np.cos(rotation.rotate_radians) + np.tan(shear.shear_y) * np.sin(rotation.rotate_radians))
        a01 = -scaling.scale_factor_y * (np.tan(shear.shear_x) * np.cos(rotation.rotate_radians) + np.sin(rotation.rotate_radians))
        b0 = translation.translation_x

        a10 = scaling.scale_factor_x * (np.sin(rotation.rotate_radians) - np.tan(shear.shear_y) * np.cos(rotation.rotate_radians))
        a11 = -scaling.scale_factor_y * (np.tan(shear.shear_x) * np.sin(rotation.rotate_radians) - np.cos(rotation.rotate_radians))
        b1 = translation.translation_y

        return np.array([[a00, a01, b0],[a10, a11, b1],[0, 0, 1]])

def transform_x_coordinates(x, y, transform: AffineTransform):
    """
    Calculate affine-transformed x coordinates using a given transform and arrays of point coordinates.

    :param x: ordered x coordinates (i.e. x[0] is the x coordinate for a point whose y coordinate is y[0]).
    :param y: ordered y coordinates (i.e. y[0] is the y coordinate for a point whose x coordinate is x[0]).
    :param transform: the affine transform to apply.
    :return: the transformed x coordinates.
    """
    a00 = transform.matrix[0, 0]
    a01 = transform.matrix[0, 1]
    b0 = transform.matrix[0, 2]
    return a00 * x + a01 * y + b0

def transform_y_coordinates(x, y, transform: AffineTransform):
    """
    Calculate affine-transformed y coordinates using a given transform and arrays of point coordinates.

    :param x: ordered x coordinates (i.e. x[0] is the x coordinate for a point whose y coordinate is y[0]).
    :param y: ordered y coordinates (i.e. y[0] is the y coordinate for a point whose x coordinate is x[0]).
    :param transform: the affine transform to apply.
    :return: the transformed y coordinates.
    """
    a10 = transform.matrix[1, 0]
    a11 = transform.matrix[1, 1]
    b1 = transform.matrix[1, 2]
    return a10 * x + a11 * y + b1

def _check_sensible_affine_parameters(image_dimensions : tuple,
                                      model: skimage.transform.AffineTransform,
                                      max_translation_fraction=0.5,
                                      max_rotation=1,
                                      max_scaling=0.5) -> bool:
    """
    Check whether an affine transform is sensible or not based on some threshold parameters.
    :param image_dimensions: the dimensions of the images being registered.
    :param model: the model to be testing
    :param max_translation_fraction: the maximum fractional translation that is deemed sensible
    :param max_rotation: the maximum rotation deemed sensible. Units radians.
    :param max_scaling: the maximum scale factor deemed sensible (in both the positive and negative directions).
    :return: true if the model is deemed sensible, otherwise false.
    """
    sensible_parameters = True
    for dim in range(len(image_dimensions)):
        if np.abs(model.translation[dim]) > image_dimensions[dim] * max_translation_fraction: sensible_parameters = False
        if model.scale[dim] < max_scaling or model.scale[dim] > 1 + max_scaling: sensible_parameters = False

    # Check the rotation value is sensible
    if model.rotation < -max_rotation or model.rotation > max_rotation: sensible_parameters = False
    return sensible_parameters

def _compare_images(image_1: np.ndarray, image_2: np.ndarray, fraction: float=0.5) -> float:
    """
    Calculate the MSE between the central region of two images.

    :param image_1: the first image to compare
    :param image_2: the second image to compare
    :param fraction: the fraction of the image dimensions to consider.
    :return: the mean-square error between the two (cropped) images.
    """
    crop_y = int(np.round(image_1.shape[0] * fraction))
    crop_x = int(np.round(image_1.shape[1] * fraction))
    start_y = image_1.shape[0] // 2 - crop_y // 2
    start_x = image_1.shape[1] // 2 - crop_x // 2

    image_1_cropped = image_1[start_y: start_y + crop_y, start_x: start_x + crop_x].copy()
    image_2_cropped = image_2[start_y: start_y + crop_y, start_x: start_x + crop_x].copy()

    return np.square(image_1_cropped - image_2_cropped).mean()

def _find_key_points_and_descriptors(image: np.ndarray, orb: ORB) -> tuple:
    """
    Find and detect key-points and corresponding descriptors using Skimage ORB methods.

    Simply wraps the skimage features, returning the key-points and descriptors as a tuple.
    :param image: the image that we are detecting features in.
    :param orb: the pre-initialised ORB object.
    :return: tuple of key-points and corresponding descriptors.
    """
    orb.detect_and_extract(image)
    key_points = orb.keypoints
    descriptors = orb.descriptors
    return key_points, descriptors

class RegistrationParameters:
    """Simple object to store parameters used in registration"""
    def __init__(self,
                 num_key_points: int=250,
                 num_trials: int=50,
                 interpolation_order: int=3,
                 plot_matches: bool=True,
                 compare_fraction: float=0.5,
                 max_translation_fraction: float=0.5,
                 max_rotation: float|int=1.0,
                 max_scaling:float=0.5,
                 normalise_images:bool=True):
        """
        Constructor for Registration Parameters.

        :param num_key_points: the number of key points to detect in each image.
        :param num_trials: the number of cycles of RANSAC fitting to perform before selecting the best model.
        :param interpolation_order: the order of interpolation to apply when warping images during registration.
        :param plot_matches: whether to plot the matched features of the best affine transform.
        :param compare_fraction: the fraction of the images to consider when comparing images when selecting the best
                                 registration. Must be < 1.
        :param max_translation_fraction: the maximum translation factor to allow in the model. Fractional units.
        :param max_rotation: the maximum rotation to allow in the model. Units radians.
        :param max_scaling: the maximum scaling to allow in the model (in either the positive or negative direction).
        :param normalise_images: whether to normalise the images in the preprocessing step.
        """
        self.num_key_points = num_key_points
        self.num_trials = num_trials
        self.interpolation_order = interpolation_order
        self.plot_matches = plot_matches
        self.compare_fraction = compare_fraction
        self.max_translation_fraction = max_translation_fraction
        self.max_rotation = max_rotation
        self.max_scaling = max_scaling
        self.normalise_images = normalise_images

def calculate_registration(image_1: np.ndarray,
                           image_2: np.ndarray,
                           registration_parameters: RegistrationParameters=None,
                           log: bool=True
                           ) -> skimage.transform.AffineTransform:
    """
    Calculate an affine transform that registers image_1 to image_2.

    This function is designed to calculate the affine registration that describes the best registration between 2 images.
    It wraps Skimage methods that find and match features in each image, before refining the matches and fitting a
    model transform using RANSAC fitting.

    This process has some inherent randomness due to the nature of RANSAC fitting (which works on randomly-selected sets
    of key-points in each image). As such, the process is repeated several times, and the model transform that provides
    the closest match between images is returned as the solution.

    :param image_1: the first image to register
    :param image_2: the second image to register
    :param registration_parameters: object describing registration parameters.
    :param log: whether to output logging information
    :return: the affine transform that best describes the mapping of image_1 to image_2.
    """
    # Initialise registration parameters if none provided.
    if registration_parameters is None: registration_parameters = RegistrationParameters()

    # First, we need to find key-points in both registration images.
    orb = ORB(n_keypoints=registration_parameters.num_key_points)

    # Ensure images are floating point, and normalise if required
    image_1 += 0.
    image_2 += 0.
    if registration_parameters.normalise_images:
        image_1 /= np.max(image_1)
        image_2 /= np.max(image_2)

    # Find key-points in the first image
    if log: logger.info("Detecting key-points in image 1")
    key_points_image_1, descriptors_image_1 = _find_key_points_and_descriptors(image=image_1, orb=orb)

    # Find key_points in image 2
    if log: logger.info("Detecting key-points in image 2")
    key_points_image_2, descriptors_image_2 = _find_key_points_and_descriptors(image=image_2, orb=orb)

    # Perform initial keypoint matching. The Skimage documentation describes this as "brute-force" matching.
    # Every descriptor/key-point in the first-image is matched to its closest match in the second image, and vice versa.
    initial_matches = match_descriptors(descriptors_image_1, descriptors_image_2, cross_check=True)

    # The previous step inevitably contains some mis-matches. Estimating the affine transform from all points will,
    # therefore, introduce errors. We use the RANSAC algorithm to detect mis-matches (model outliers).
    # First, we need to order the key points according to their matches.
    src_key_points = key_points_image_2[initial_matches[:, 1]][:, ::-1]
    dst_key_points = key_points_image_1[initial_matches[:, 0]][:, ::-1]

    possible_transforms, possible_inliers, mses = [], [], []

    if log: logger.info("Finding possible transforms")
    while len(possible_transforms) < registration_parameters.num_trials:
        # Use the RANSAC algorithm from skimage to estimate the outliers and fit a model to the data
        model_robust, inliers = ransac(data=(src_key_points, dst_key_points), model_class=skimage.transform.AffineTransform,
                                 min_samples=3, residual_threshold=10, max_trials=500)

        # We do a quick sanity check on the result, to make sure we have some sensible parameters
        sensible_parameters = _check_sensible_affine_parameters(image_dimensions=image_1.shape,
                                                                model=model_robust,
                                                                max_translation_fraction=registration_parameters.max_translation_fraction,
                                                                max_rotation=registration_parameters.max_rotation,
                                                                max_scaling=registration_parameters.max_scaling)

        # If the returned model contains sensible parameters, we store it.
        if sensible_parameters:
            possible_transforms.append(model_robust)
            possible_inliers.append(inliers)

    # Now we have a set of possible models. We need to select the best one. We calculate this by warping the images
    # according to each transform, and calculate the MSE between the registered images. The best model is the one that
    # provides the lowest MSE. Note: we only compare the central 50% of the images, since discontinuities at the edges
    # will distort the results.
    for model in possible_transforms:
        image_1_warped = skimage.transform.warp(image_1, model, order=registration_parameters.interpolation_order,
                                                clip=False, preserve_range=True)
        mses.append(_compare_images(image_1_warped, image_2, fraction=registration_parameters.compare_fraction))

    # The best model is the one that provides the lowest MSE between registered images
    best_model = possible_transforms[np.argmin(mses)]

    if registration_parameters.plot_matches:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        plt.title("Matches after RANSAC fitting")

        plot_matched_features(image0=image_1, image1=image_2,
                              keypoints0=key_points_image_1, keypoints1=key_points_image_2,
                              matches=initial_matches[possible_inliers[np.argmin(mses)]], ax=ax)
        ax.axis('off')
        plt.show()

        plt.figure(figsize=(16, 8))
        plt.plot(mses, "."), plt.title("Similarity after Transforms"), plt.xlabel("Trial"), plt.ylabel("MSE")

    return best_model


