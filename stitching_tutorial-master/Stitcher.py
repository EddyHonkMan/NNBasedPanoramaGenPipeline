from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pathlib import Path
import stitching

class Stitcher:
    def __init__(self, background_frames):

        self.frames = background_frames
        self.img_handler = stitching.ImageHandler(True)
        self.img_handler.set_img_names(self.frames)
        self.medium_imgs = []
        self.low_imgs = []
        self.final_imgs = []
        self.features = None
        self.matches = None
        self.camera = None
        self.low_corners = []
        self.low_sizes = []
        self.final_corners = []
        self.final_sizes = []
        self.warped_low_imgs = []
        self.warped_low_masks = []
        self.warped_final_imgs = []
        self.warped_final_masks = []
        self.seam_masks = []
        self.compensated_imgs = []

    # def get_image_paths(img_set):
    #     return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]
    # SAL_imgs = get_image_paths('SAL')
    def resize(self):
        """
        The first step is to resize the images to medium (and later to low) resolution.
        The class which can be used is the ImageHandler class.
        If the images should not be stitched on full resolution,
        this can be achieved by setting the final_megapix parameter to a number above 0.
        """

        self.medium_imgs = list(self.img_handler.resize_to_medium_resolution())
        self.low_imgs = list(self.img_handler.resize_to_low_resolution(self.medium_imgs))
        self.final_imgs = list(self.img_handler.resize_to_final_resolution())


    def find_features(self):
        """
        On the medium images, we now want to find features that can describe conspicuous
        elements within the images which might be found in other images as well.
        """
        finder = stitching.FeatureDetector()
        self.features = [finder.detect_features(img) for img in self.medium_imgs]

    def match(self):
        matcher = stitching.FeatureMatcher()
        self.matches = matcher.match_features(self.features)

    def camera_correction(self):
        """
        With the features and matches we now want to calibrate cameras which can be used to warp the images,
        so they can be composed correctly.
        """
        camera_estimator = stitching.CameraEstimator()
        camera_adjuster = stitching.CameraAdjuster()
        wave_corrector = stitching.WaveCorrector()

        self.cameras = camera_estimator.estimate(self.features, self.matches)
        self.cameras = camera_adjuster.adjust(self.features, self.matches,  self.cameras)
        self.cameras = wave_corrector.correct(self.cameras)

    def subset(self):
        subsetter = stitching.Subsetter()
        indices = subsetter.get_indices_to_keep(self.features, self.matches)
        self.medium_imgs = subsetter.subset_list(self.medium_imgs, indices)
        self.low_imgs = subsetter.subset_list(self.low_imgs, indices)
        self.final_imgs = subsetter.subset_list(self.final_imgs, indices)
        self.features = subsetter.subset_list(self.features, indices)
        self.matches = subsetter.subset_matches(self.matches, indices)

        img_names = subsetter.subset_list(self.img_handler.img_names, indices)
        img_sizes = subsetter.subset_list(self.img_handler.img_sizes, indices)

        self.img_handler.img_names, self.img_handler.img_sizes = img_names, img_sizes

    def wrap(self):
        """
        With the obtained cameras we now want to warp the images itself into the final plane
        """
        warper = stitching.Warper()

        # set the the medium focal length of the cameras as scale:
        warper.set_scale(self.cameras)

        # Warp low resolution images
        low_sizes = self.img_handler.get_low_img_sizes()
        camera_aspect = self.img_handler.get_medium_to_low_ratio()  # since cameras were obtained on medium imgs
        self.warped_low_imgs = list(warper.warp_images(self.low_imgs, self.cameras, camera_aspect))
        self.warped_low_masks = list(warper.create_and_warp_masks(low_sizes, self.cameras, camera_aspect))
        self.low_corners, self.low_sizes = warper.warp_rois(low_sizes, self.cameras, camera_aspect)

        # Warp final resolution images
        final_sizes = self.img_handler.get_final_img_sizes()
        camera_aspect = self.img_handler.get_medium_to_final_ratio()  # since cameras were obtained on medium imgs

        self.warped_final_imgs = list(warper.warp_images(self.final_imgs, self.cameras, camera_aspect))
        self.warped_final_masks = list(warper.create_and_warp_masks(final_sizes, self.cameras, camera_aspect))
        self.final_corners, self.final_sizes = warper.warp_rois(final_sizes, self.cameras, camera_aspect)

    def seam(self):
        seam_finder = stitching.SeamFinder()

        self.seam_masks = seam_finder.find(self.warped_low_imgs, self.low_corners, self.warped_low_masks)
        self.seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(self.seam_masks, self.warped_final_masks)]

        # seam_masks_plots = [seam_finder.draw_seam_mask(img, seam_mask) for img, seam_mask in
        #                     zip(self.warped_final_imgs, self.seam_masks)]
        # plot_images(seam_masks_plots, (15, 10))

    def exposure_error_compensation(self):
        compensator = stitching.ExposureErrorCompensator()

        compensator.feed(self.low_corners, self.warped_low_imgs, self.warped_low_masks)

        self.compensated_imgs = [compensator.apply(idx, corner, img, mask)
                            for idx, (img, mask, corner)
                            in enumerate(zip(self.warped_final_imgs, self.warped_final_masks, self.final_corners))]

    def blending(self):
        blender = stitching.Blender()
        blender.prepare(self.final_corners, self.final_sizes)
        for img, mask, corner in zip(self.compensated_imgs, self.seam_masks, self.final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()
        return panorama



def read_key_frames(video_path):
    """
    return video frames
    """
    cap = cv.VideoCapture(str(video_path))
    frames = []
    count = -1
    while True:
        count += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if count % 60 != 0:
            continue
        frames.append(frame)
        # cv.imwrite('imgs_{}.jpg'.format(count), frame)
    return frames




def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':

    bg_frames = read_key_frames('test1.mp4')
    # read_key_frames('test1.mp4')
    print(len(bg_frames))

    stitcher = Stitcher(bg_frames)

    # resize images
    stitcher.resize()

    # find features
    stitcher.find_features()

    # Match Features
    stitcher.match()

    # subset
    stitcher.subset()

    # camera Estimation
    stitcher.camera_correction()

    # warp image
    stitcher.wrap()

    # Seam Masks
    stitcher.seam()

    # Exposure Error Compensation
    stitcher.exposure_error_compensation()

    # Blending
    panorama = stitcher.blending()
    # plot_image(panorama, (10, 10))







