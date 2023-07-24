import numpy as np
import cv2
import edgeDetection as ed
from dataclasses import dataclass

# Read and convert the input image to grayscale

input_image = cv2.imread('img01.jpeg')
input_image = cv2.cvtColor(src=input_image, code=cv2.COLOR_BGR2GRAY)

def houghTransform(image, rhoRes, thetaRes):

    # Generate theta values from 0 to 2*pi with a specified resolution

    thetaScale = np.linspace(0, 2 * np.pi, thetaRes)
    
    # Initialize the Hough accumulator matrix with zeros

    H = np.zeros((rhoRes, thetaRes))
    
    # Loop over the edge-detected image to accumulate Hough transform

    for i in range(len(image)):
        for j in range(len(image[0])):

            # Check if there's an edge pixel at (i, j)
            if (image[i][j] != 0):

                # Calculate the rho values for each theta value and update H accordingly

                rhoScale = [int(i * np.cos(theta) + j * np.sin(theta)) for theta in thetaScale]
                for k in range(thetaRes):
                    H[rhoScale[k]][k] += 1
                    
    return H

# Apply edge detection using Sobel operator and non-maximum suppression

gradients, gradient_direction = ed.sobelOperator(ed.gaussianBlur(input_image))
suppressed_image = ed.nonMaxSuppression(gradients, gradient_direction)
edge_detected_image = ed.pThresholding(suppressed_image, 0.90)

# Display the edge-detected image

ed.displayImage(edge_detected_image)

# Calculate the resolution for rho and theta in the Hough transform

im_X, im_Y = edge_detected_image.shape
rhoRes = int(np.sqrt(im_X**2 + im_Y**2))
thetaRes = 360

# Perform Hough Transform on the edge-detected image

hough_image = houghTransform(edge_detected_image, rhoRes, thetaRes)

# Display the Hough transform result

ed.displayImage(hough_image)

def nonMaximalHough(H, patch_dimension):

    # Iterate through H with a sliding window of size patch_dimension x patch_dimension
    x, y = H.shape

    for i in range(0, x - patch_dimension):
        for j in range(0, y - patch_dimension):

            # Find the maximal value in the window and suppress other values

            maximalValue = np.max(H[i:i+patch_dimension, j:j+patch_dimension])
            H[i:i+patch_dimension, j:j+patch_dimension] = np.where(H[i:i+patch_dimension, j:j+patch_dimension] < maximalValue, 0, maximalValue)
            
    return H

def houghLines(H, nLines):

    # Perform non-maximal suppression on H

    H = nonMaximalHough(H, patch_dimension = 3)

    # Find the nLines local maxima (peaks) in H

    localMaxima = []
    flattened_H = H.flatten()
    sorted_by_index = np.argsort(flattened_H)[::-1]

    for i in range(nLines):

        localMaxima.append(np.unravel_index(sorted_by_index[i], H.shape))
        
    return localMaxima

# Find the local maxima in the Hough transform to obtain the lines

localMaxima = houghLines(hough_image, nLines=20)

# Data class to represent a line segment with start and end points

@dataclass
class Line:
    start: tuple
    end: tuple

# Function to convert Hough line parameters into line segments

def HoughLineSegments(allLineParameters):
    lineArray = []
    for lineParameters in allLineParameters:
        lineRho = lineParameters[0]
        lineTheta = lineParameters[1] * np.pi/180

        # Calculate the line endpoints based on the Hough line parameters

        c = np.cos(lineTheta + 0.00001)
        s = np.sin(lineTheta + 0.00001)
        x1 = -10000
        y1 = int((-c/(s)) * x1 + lineRho/(s))
        x2 = 10000
        y2 = int((-c/(s)) * x2 + lineRho/(s))

        # Create a Line object with the start and end points and add it to lineArray

        lineArray.append(Line((y1, x1), (y2, x2)))

    return lineArray

def drawLines():

    lines = HoughLineSegments(localMaxima)
    inputImage = input_image.copy()
    inputImageLines = cv2.cvtColor(inputImage, cv2.COLOR_GRAY2BGR)
    
    for line in lines:

        cv2.line(inputImageLines, line.start, line.end, (0, 255, 0), 2)
    
    ed.displayImage(inputImageLines)

drawLines()