import numpy as np

class Window(object):

    def __init__(self, frame, coords):

        self.frame = frame
        self.coords = coords

        # Defaults to 10 bins for the histogram, may need to increase that number
        self.histogram = np.histogram(self.frame, bins=36)



    def __repr__(self):

        # return "A 40 pixel by 40 pixel grid ranging from ({0}, {1}) to ({2}, {3})".format(coords[0], coords[1], coords[0]+39, coords[1]+39)
        return "A 40 pixel by 40 pixel grid ranging from (" + str(self.coords[0]) + "," + str(self.coords[1]) + ") to (" + str(self.coords[0]+39) + "," + str(self.coords[1]+39) + ")"

    # def gen_histogram(self):
        # self.histogram = np.histogram(self.frame)
