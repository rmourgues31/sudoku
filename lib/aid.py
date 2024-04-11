import cv2
import numpy as np
import glob
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle
import math


# The next two functions were taken from
# https://github.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/blob/master/four_point_object_extractor.py


def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def order_points_of_quadrilateral(pts):
    """
    Given an array of four points describing a quadrilateral,
    sorts them in the following order:
    (top-left, top-right, bottom-right, bottom-left)
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Summing the x and y coordinates of each point to one value,
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Computing the difference between the x and y value of each point, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(img, pts):
    """
    Given an array of four points describing a quadrilateral in an
    image, returns a homography matrix that warps this quadrilateral
    into a top-down view
    """
    # Obtain a consistent order of the points and unpack them
    # individually
    rect = order_points_of_quadrilateral(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(
        ((bottom_right[0] - bottom_left[0]) ** 2)
        + ((bottom_right[1] - bottom_left[1]) ** 2)
    )
    widthB = np.sqrt(
        ((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2)
    )
    max_width = min(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(
        ((top_right[0] - bottom_right[0]) ** 2)
        + ((top_right[1] - bottom_right[1]) ** 2)
    )
    height_b = np.sqrt(
        ((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2)
    )
    max_height = min(int(height_a), int(height_b))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [
            [10, 10],
            [max_width - 10, 10],
            [max_width - 10, max_height - 10],
            [10, max_height - 10],
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width + 10, max_height + 10))

    return warped, M


def find_cells(img):
    """
    Find the cells of a sudoku grid
    """
    img_area = img.shape[0] * img.shape[1]

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Array containing the cropped cell image and its position in the grid
    cells = []
    for c in contours:
        area = cv2.contourArea(c)

        # Approximate the contour in order to determine whether the contour is a quadrilateral
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.017 * peri, True)

        # We are looking for a contour of a specific area in relation to the grid size
        # and that is roughly quadrilateral
        # We filter for areas that are too small or too large in relation to the whole image
        if area / img_area > 0.0001 and area / img_area < 0.02 and len(approx) == 4:
            # Using masking, we crop the cell into its own 28 by 28 pixel image
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [c], -1, 255, -1)

            (y, x) = np.where(mask == 255)

            (top_y, top_x) = (np.min(y), np.min(x))
            (bottom_y, bottom_x) = (np.max(y), np.max(x))
            cell = img[top_y : bottom_y + 1, top_x : bottom_x + 1]

            cell = cell.copy()
            cell = cv2.resize(cell, (28, 28))

            # We also find the centroid of the cell in relation
            # to the grid
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cells.append(({"img": cell, "pos": (cX, cY)}))

    return cells


def find_grid(img):
    """
    Find a sudoku grid in an image. Returns a perspective adjusted image of the grid,
    cell information, and the homography matrix used for the perspective warp.
    """

    # Preprocess the image
    img_blur = cv2.blur(img, (3, 3))
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 3
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Approximate the contour in order to determine whether the contour is a quadrilateral
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # We are looking for a contour that is roughly a quadrilateral
        if len(approx) == 4:
            warped, M = four_point_transform(
                thresh,
                np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]]),
            )

            cells = find_cells(warped)

            # We can be fairly certain we found a sudoku grid if the grid contains 81 cells
            if len(cells) == 81:
                return True, warped, M

    return False, None, None


def get_sudoku_grid(cells):
    """
    Given a list of cells and they position, return a 2D array representing
    a sudoku grid, where each element of this 2D array contains of the value of the grid
    at that position.
    """
    cells.sort(key=lambda cell: cell["pos"][0])

    cells_X = []
    i = 0
    for cell in cells:
        cells_X.append(
            {"pos": cell["pos"], "value": cell["value"], "columnIndex": int(i / 9)}
        )
        i += 1

    cells_X.sort(key=lambda cell: cell["pos"][1])

    cells_in_grid = []
    i = 0
    for cell in cells_X:
        cells_in_grid.append(
            {
                "pos_in_grid": (cell["columnIndex"], int(i / 9)),
                "value": cell["value"],
                "pos": cell["pos"],
            }
        )
        i += 1

    grid = np.zeros((9, 9))
    grid_meta = np.zeros((9, 9, 3))
    for cell in cells_in_grid:
        grid[cell["pos_in_grid"][1], cell["pos_in_grid"][0]] = cell["value"]
        # We keep some info about the grid: centroid of each cell, and whether it is blank
        grid_meta[cell["pos_in_grid"][1], cell["pos_in_grid"][0]] = [
            cell["pos"][0],
            cell["pos"][1],
            cell["value"] == 0,
        ]

    return grid, grid_meta


def correct(num, x, y, grid):
    """
    Determine whether a particular number at a particular location
    is a valid sudoku move.
    """
    # check the row
    for i in range(9):
        # No need to check our current position
        if i == x:
            continue

        # if the digit is present in the row, it is not a valid move
        if grid[y, i] == num:
            return False

    # check the columnm
    for j in range(9):
        # No need to check our current position
        if j == y:
            continue

        # if the digit is present in the column, it is not a valid move
        if grid[j, x] == num:
            return False

    # check the square
    (square_left, square_top) = (math.floor(x / 3) * 3, math.floor(y / 3) * 3)
    for i in range(square_left, square_left + 3):
        for j in range(square_top, square_top + 3):
            if j == y and i == x:
                continue

            if grid[j, i] == num:
                return False

    return True


def solve(sudoku, empty_cells):
    """
    Create a solved sudoku grid using backtracking and recursion. Inspired from https://stackoverflow.com/a/58139570/2034508
    """
    if len(empty_cells) > 0:
        # Work on the first empty cell
        empty = empty_cells[0]
        (i, j) = empty

        # Test for a solution with numbers from 1 to 9
        for num in range(1, 10):
            # Skip invalid moves
            if not correct(num, j, i, sudoku):
                continue

            # If move is valid, set it, and solve the sudoku from there
            sudoku[i, j] = num
            if solve(sudoku, empty_cells[1:]):
                return True
            # If we reach this point, we could not solve the sudoku with that move
            # Reset, and try a different number
            sudoku[i, j] = 0

        # If we reach this point, we could not solve the sudoku with any number for that cell
        # Backtrack
        return False
    else:
        # If empty cell array is empty, we are done!
        return True


def print_solution(img, warped, M, solved_grid, grid_meta):
    """
    Create a solution image, where the solution is added to the original
    image, accounting for perspective
    """
    # Start with a blank image the dimensions of the warped image
    solution_img = np.ones(warped.shape) * 255

    for i in range(9):
        for j in range(9):
            value = solved_grid[j, i]
            pos = grid_meta[j, i]
            was_empty = grid_meta[j, i, 2]

            # We do not print sudoku hints, as they will already be in the image
            if not was_empty:
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(int(value))

            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            # Position the text in the center of the cell
            textX = int((pos[0] - textsize[0] / 2))
            textY = int((pos[1] + textsize[1] / 2))

            cv2.putText(solution_img, text, (textX, textY), font, 1, (0, 0, 0), 2)

    # We apply the inverse of the homography matrix calculated earlier. This will allow us
    # to place the sudoku solution in the original image
    unwarped_img = cv2.warpPerspective(
        solution_img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    # Add only the text portion of the solution image on the original image (the text is black, with value 0)
    # For added styling, we make the solution blue in the original image (written in BRG notation)
    img[np.where(unwarped_img == 0)] = (85, 15, 0)

    return img


if __name__ == "__main__":
    for file_path in glob.glob("in/*.jpg"):
        img = cv2.imread(file_path)

        # Ensure we keep the aspect ratio of the image
        ratio = img.shape[0] / img.shape[1]
        img = cv2.resize(img, (1100, int(1100 * ratio)))

        valid, img_grid, M = find_grid(img)

        # If no grid were found, give up
        if valid == False:
            continue

        # Generate a 2D array representation of the grid present in the image
        cells = find_cells(img_grid)
        
