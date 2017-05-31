import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

image = cv2.imread('./images/bbox-example-image.jpg', 0)
copy = np.copy(image)

def draw_boxes(img, bboxes, color=(0,0,0), thick=6):
    for b in bboxes:
        (pt1), (pt2) = b
        img = cv2.rectangle(img, pt1, pt2, color, thick)
    return img

def hand_drawn():
    bboxes = []
    bboxes.append(((853, 670),(1150, 514)))
    bboxes.append(((275, 572),(378, 502)))
    bboxes.append(((484, 562),(548, 515)))
    bboxes.append(((599, 553),(640, 519)))

    color = (0,0,255)
    boxed = draw_boxes(copy, bboxes, color)
    plt.imshow(boxed)
    plt.show()


def find_matches(img, template_list, method):
    for t in template_list:
        w, h = t.shape[::-1]
        img2 = img.copy()
        # apply template matching
        res = cv2.matchTemplate(img2, t, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)

    return img



# Load templates
template_images = glob.glob('./images/cutout*.jpg')
templates = []
for t in template_images:
    image = cv2.imread(t, 0)
    templates.append(image)

compMethods = [
        'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
        'cv2.TM_SQDIFF_NORMED'
    ]


for m in compMethods:
    method = eval(m)
    result = find_matches(copy, templates, method)
    plt.imshow(result, cmap='gray')
    plt.suptitle(m)
    plt.show()