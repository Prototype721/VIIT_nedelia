from preprocess import preprocess_image
import cv2

def main():
    image = cv2.imread("test_image.jpg")

    image_result = preprocess_image(image)

    cv2.imshow("Image", image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









if __name__ == "__main__":
    main()