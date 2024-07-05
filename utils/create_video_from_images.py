import cv2
import os
from natsort import natsorted

def create_video_from_images(
        input_folder: str,
        output_video_name: str = 'output_video.mp4',
        codec: str = 'mp4v',
        fps: int = 30,
        fmt='.png'):
    """
    Creates a video from a folder of images.

    Args:
    - input_folder (``str``): The path to the folder containing the images.
    - output_video_name (``str``): The name of the output video.
    - codec (``str``): The codec to use for the video. Default: mp4v.
    - fps (``int``): The frames per second of the video. Default: 30.
    - fmt (``str``): The image format of the images in the folder.
        Default: .png.
    """
    # List all the image file names in the directory
    images = [img for img in os.listdir(input_folder) if img.endswith(fmt)]

    # Sort the image file names to ensure proper order
    images = natsorted(images)
    # breakpoint()
    # Get the first image to retrieve dimensions
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # breakpoint()
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # Loop through each image and write it to the video
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer and close the video file
    video.release()

    print(f"Video '{output_video_name}' creation complete!")


if __name__ == "__main__":
    create_video_from_images(
        input_folder='/media/gnort/HDD/Work_New/PheNet-Fisheye/Fisheye_Segment/save/test_img/pspnet_fisheye_5',
        output_video_name='/media/gnort/HDD/Work_New/PheNet-Fisheye/Fisheye_Segment/save/demo_videos/demo_4_darea_pspnet.mp4',
        codec='mp4v',
        fps=30)
