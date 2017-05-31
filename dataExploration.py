import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Prepare data
images = glob.glob('./subset/*.jpeg')
cars = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

def data_look(car_list, notcar_list):
    data_dict = {}
    # key for cars
    data_dict["n_cars"] = len(car_list)
    # key for non-cars
    data_dict["n_notcars"] = len(notcar_list)

    example_img = mpimg.imread(car_list[0])

    data_dict["image_shape"] = example_img.shape
    data_dict["data_type"] = example_img.dtype
    return data_dict

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Get both features and visualizations
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features



def get_image_info():
    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and' , data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:', data_info["data_type"])

    car_ind = np.random.randint(0, len(cars))
    notcar_ind  = np.random.randint(0, len(notcars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # plot the images
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title("Random car")
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title("Non car image")
    plt.show()


# Generate random image to look at
ind = np.random.randint(0, len(cars))
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Hog Params
orient = 9
pix_per_cell = 8
cell_per_block = 2

features, hog_image = get_hog_features(gray,orient,pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.show()




