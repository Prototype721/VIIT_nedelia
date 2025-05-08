import cv2
import numpy as np

def preprocess_image(image):

    def create_white_patch_image(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = image_gray.astype(np.float32)
        max_vals = img.max(axis=0)
        scale = 255.0 / max_vals
        balanced = img * scale
        image_gray_white = np.clip(balanced, 0, 255).astype(np.uint8)
        return image_gray_white

    def find_edges_document(pts):
        pts = np.array(pts, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]       # top-left
        rect[2] = pts[np.argmax(s)]       # bottom-right
        rect[1] = pts[np.argmin(diff)]    # top-right
        rect[3] = pts[np.argmax(diff)]    # bottom-left
        return rect

    def change_perspective(rect, image):
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Destination points for the transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped


    image_gray_white = create_white_patch_image(image)
    

    blurred = cv2.GaussianBlur(image_gray_white, (15, 15), 1)
    edges = cv2.Canny(blurred, 100, 200)

    
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2: contours = contours[0]; 
    else: contours = contours[1];
    
    largest_contour = max(contours, key=cv2.contourArea)

    #cv2.drawContours(image, largest_contour, -1, (0, 255, 0), 3)
    #cv2.imshow("edges", image)

    points = largest_contour.reshape(-1, 2)
    rect = find_edges_document(points)

    warped_image = change_perspective(rect, image)
    
    image_result = create_white_patch_image(warped_image)
    # cv2.imshow("White_patch", image_result)

    #image_result = cv2.resize(image_result, (1000, 1000))

    return image_result



