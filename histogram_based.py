import cv2
import numpy as np
from ipcv.motion_tracking.grid_window import Window

## Helper functions
def is_num(s):

    # Attempts to cast the input to a float. If an exception occurs, it can be
    # safely assumed that the input is not a datatype that is conventionally
    # used to represent a number.
    try:
        float(s)
        return True
    except ValueError:
        return False


def str_parse(input):

    # Used to parse the given filename for the numerical identifier so that 
    # the video frames can be properly ordered

    # Create empty string container
    str_build = ""

    # Iterate through characters that make up the input string
    for char in input:

        # If the character is a number, append it to the string container.
        if is_num(char):
            str_build += char
        # If the character is a period, return the string built up so far
        elif char == ".":
            return str_build


## Main functions
def calc_gradients(img):

    # Cast image to float32 datatype to balance precision and amount of data
    img = img.astype(np.float32)

    # Define Sobel operators used to compute partial derivatives of the image
    dx = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])


    dy = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    # Create empty arrays to hold Sobel-computed imagess
    dix = np.zeros(img.shape)
    diy = np.zeros(img.shape)

    for dim in range(0, img.shape[2]):
        dix[:,:,dim] = cv2.filter2D(img[:,:,dim], -1, dx)
        diy[:,:,dim] = cv2.filter2D(img[:,:,dim], -1, dy)

    # Create empty array to hold output edge image
    edge_image = np.zeros((img.shape[0], img.shape[1]))

    # Iterate through all pixels in the image to compute edges
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):

            # Define S matrix
            elem2 = (dix[r,c,2] * diy[r,c,2]) + (dix[r,c,1] * diy[r,c,1]) + (dix[r,c,0] * diy[r,c,0])
            S = np.matrix([[(dix[r,c,2])**2 + (dix[r,c,1])**2 + (dix[r,c,0])**2, elem2], [elem2, (diy[r,c,2])**2 + (diy[r,c,1])**2 + (diy[r,c,0])**2]])

            # Use the trace of S to determine the strength of the edge at (r,c)
            edge_image[r, c] = np.trace(S)

    # Normalize the new computed edge image
    edge_image /= np.max(edge_image)

    return edge_image


def calc_background(frames):

    # Store datatype of the input imagery and create empty array to hold
    # background
    img_dtype = frames[0].dtype
    img_build = np.zeros(frames[0].shape)

    # if len(frames) > 10:
    #     frames = frames[-10::]

    # if len(frames) > 100:
    #     frames = frames[-100::]

    # For each image passed in, add it to the empty array to build the
    # background image
    for img in frames:
        for dim in range(0, img.shape[2]):
            img_build[:,:,dim] +=img[:,:,dim]

    # Divide the summed frames by the number of frames added
    img_build /= len(frames)

    # Cast image to input datatype and return
    return img_build.astype(img_dtype)


def gen_histograms(frame, mode="edge"):

    img = frame.copy()

    # Compute edges of color imagery if using edge histograms method
    if mode == "edge":
        img = calc_gradients(img)

    # Divide image into 40 x 40 grid
    # Should this be a grid of 40 x 40 pixel sections?

    height = img.shape[0]
    width = img.shape[1]

    height_mod = height % 40
    width_mod = width % 40

    # Should pad duplicate END columns and rows of image to make sure it easily divisable into 40 x 40 Windows
    if height_mod != 0:
        img = np.pad(img, (0, 40-height_mod), mode="edge")

    if width_mod != 0:
        img = np.pad(img, (40-width_mod, 0), mode="edge")

    # Define 40 pixel by 40 pixel window
    # Let's define a class to hold the windows, generate histograms in __init__

    grid = []

    # Iterate through frame and assemble 40 pixel by 40 pixel windows across entire frame
    for r in range(0, height, 40):
        for c in range(0, width, 40):

            obj = Window(img[r:r+40, c:c+40], (r,c))
            grid.append(obj)


    # Assemble sub-grid
    for r in range(20, height-20, 40):
        for c in range(20, width-20, 40):

            obj = Window(img[r:r+40, c:c+40], (r,c))
            grid.append(obj)

    return grid


def compare_histograms(image_grid, background_grid):

    coords_list = []

    # Iterate through all images in the assembled iamge grid
    for x in range(0, len(image_grid)):

        # Choose desired histograms
        img_hist = image_grid[x].histogram
        back_hist = background_grid[x].histogram

        # Intersection of the histograms
        inter = cv2.compareHist(back_hist[0].astype(np.float32), img_hist[0].astype(np.float32), method=cv2.HISTCMP_INTERSECT)
        inter /= len(img_hist[0])

        # Chi-Squared measure of similarity
        chi2 = cv2.compareHist(back_hist[0].astype(np.float32), img_hist[0].astype(np.float32), method=cv2.HISTCMP_CHISQR)

        # Check thresholds and store coordinates if the pixel passes
        if inter < 40:
            if chi2 > 40:
                coords_list.append(image_grid[x].coords)

    return coords_list


## Test harness
if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    import os
    import os.path
    import ipcv

    # home = os.path.expanduser('~')
    # file_path = home + os.path.sep + 'src/python/examples/data'
    # file_path += os.path.sep + 'motion_tracking/hand/'
    # file_path += os.path.sep + 'motion_tracking/woman/'
    # file_path += os.path.sep + 'motion_tracking/car/'
    # file_path += os.path.sep + 'motion_tracking/torus/'
    file_path = 'video/hand/'

    # Treat folder as series of frames
    file_list = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    # Create "video" list with only .jpg files to avoid .txt or anything
    # entering. Should expand for other image types.
    video = []
    for filename in file_list:
        if filename.endswith(".jpg"):
            video.append(filename)

    frame = cv2.imread(file_path+video[0], cv2.IMREAD_UNCHANGED)

    img_container = []
    # Iterate through video files in order to generate histograms and background
    for infile in sorted(video, key=str_parse):
        print("Current Frame: " + infile)

        # If there are any previously stored frames, calculate a background image
        if len(img_container) != 0:
            background_img = calc_background(img_container)
            background_grid = gen_histograms(background_img)

            cv2.namedWindow("Background", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Background", background_img)
        else:
            background_img = np.zeros(frame.shape)

        # Read in current frame and store in img_container to use for later
        # background images
        frame = cv2.imread(file_path + infile, cv2.IMREAD_UNCHANGED)
        img_container.append(frame)

        # Call gen_histograms with the current frame
        frame_grid = ipcv.motion_tracking.gen_histograms(frame)

        # If a background image has been calculated, compare the current frame
        if len(img_container)-1 != 0:
            coords_list = compare_histograms(frame_grid, background_grid)

            subject_track = np.zeros(frame.shape)

            for coords in coords_list:
                subject_track[coords[0]:coords[0]+40, coords[1]:coords[1]+40, :] = frame[coords[0]:coords[0]+40, coords[1]:coords[1]+40, :]
        else:
            subject_track = np.zeros(frame.shape)

            cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Tracking", subject_track.astype(np.uint8))

        cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Frame", frame)

        cv2.namedWindow("Edge Frame", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Edge Frame", calc_gradients(frame))

        # out_gradients = np.zeros(frame.shape)
        #
        # for dim in range(0, 3):
        #     out_gradients[:,:,dim] = img_gradients

        # Build up video frame and cast to uint8 as per the constraints of the
        # video format

        # firstRow = np.concatenate((frame, out_gradients), axis=1)
        # secondRow = np.concatenate((background_img, subject_track), axis=1)
        # vid_frame = np.concatenate((firstRow, secondRow), axis=0)
        # vid_frame = vid_frame.astype(np.uint8)

        # Display the current frame that is being written for troubleshooting
        # cv2.imshow('Frame', vid_frame)

        cv2.waitKey(100)

    # Perform clean-up
    ipcv.flush()
