import cv2
import numpy as np

# image_path
img_path = r"C:\Users\Tran Cu Phu\Documents\Intern\API Test\processed\processed_image.png"

# read image
img_raw = cv2.imread(img_path)
# show cropped image
fixed_size = (500, 500)
resized_image = cv2.resize(img_raw, fixed_size, interpolation=cv2.INTER_AREA)

try:
    while True:
        # select ROI function
        roi = cv2.selectROI(resized_image)

        # print rectangle points of selected roi
        print(roi)

        # Crop selected roi from raw image
        roi_cropped = resized_image[
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
        ]

        # show cropped image
        cv2.imshow("ROI", roi_cropped)

        # cv2.imwrite("crop.jpeg", roi_cropped)

        # hold window
        cv2.waitKey(0)
except KeyboardInterrupt:
    pass
