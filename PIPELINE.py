import cv2
import numpy as np
import glob
import pandas as pd
import geopy.distance
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from scipy import spatial
import time

start_time = time.time()
mean = 30.0   # makes it brighter
std = 50.0    # standard deviation - fuzzy

df = pd.read_excel('D900.xlsx', sheet_name='D10')
data = df.values
df = pd.read_excel('DO.xlsx', sheet_name='D11')  # D2
live = df.values

match_index = []
for i in range(len(live)):
    distance, index = spatial.KDTree(data).query(live[i])
    match_index.append(index)
print(match_index)

sift = cv2.xfeatures2d.SIFT_create()

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

filenames1 = [img for img in glob.glob("over/*.png")]  # D2 - Live
filenames1.sort()

images = []
titles = []
quality = 0

# Live Images
for img in filenames1:
    n = cv2.imread(img)

    # Noise
    noisy_img = n + np.random.normal(mean, std, n.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    cv2.imwrite("noisy.jpg", noisy_img_clipped)
    noisy = cv2.imread("noisy.jpg")
    # images.append(noisy)


    # Compression
    cv2.imwrite('com.jpg', noisy, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    original = cv2.imread("com.jpg")
    images.append(original)

    # Original
    # images.append(n)

    titles.append(img)

filenames2 = [img for img in glob.glob("D900/*.png")]  # D1 - Data
filenames2.sort()

im = []
titl = []
gps = []
count1 = 0
count2 = 0
count3 = 0

# Database Images
for m_array in match_index:
    for l in range(len(filenames2)):
        if l == m_array:
            if l < 39:
                for k in range(75):
                    n = cv2.imread(filenames2[k])

                    # Compression
                    cv2.imwrite('com.jpg', n, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    L = cv2.imread("com.jpg")
                    im.append(L)

                    # Original
                    # im.append(n)

                    titl.append(filenames2[k])
                    gps.append(data[k])
                count1 = count1 + 1

            if l > (len(filenames2) - 39):
                for k in range(len(filenames2) - 75, len(filenames2)):
                    n = cv2.imread(filenames2[k])

                    # Compression
                    cv2.imwrite('com.jpg', n, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    L = cv2.imread("com.jpg")
                    im.append(L)

                    # Original
                    # im.append(n)

                    titl.append(filenames2[k])
                    gps.append(data[k])
                count2 = count2 + 1

            if 39 < l < (len(filenames2) - 39):
                for k in range(l - 38, l + 37):
                    n = cv2.imread(filenames2[k])

                    # Compression
                    cv2.imwrite('com.jpg', n, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    L = cv2.imread("com.jpg")
                    im.append(L)

                    # Original
                    # im.append(n)

                    titl.append(filenames2[k])
                    gps.append(data[k])
                count3 = count3 + 1


print(len(im))
print(len(images))
x = 1
j = 0
match_gps = []
test_arr = []
for live_image, title in zip(images, titles):
    best_sim = 0
    test = 0
    for i in range(75 * j, 75 * (j + 1)):
        image_to_compare = im[i]
        # 1) Check if 2 images are equals
        if live_image.shape == image_to_compare.shape:
            # print("The images have same size and channels")
            difference = cv2.subtract(live_image, image_to_compare)
            b, g, r = cv2.split(difference)

            # if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            #     print("The images are completely Equal")
            # else:
            #     print("The images are NOT equal")

        # 2) Check for similarities between the 2 images

        kp_1, desc_1 = sift.detectAndCompute(live_image, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)

        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        sim = len(good_points) / number_keypoints * 100

        if sim > best_sim:
            best_sim = sim
            # if best_sim > 1.1:
            best_gps = gps[i]
            test = i
            # else:
            #     best_gps = 0

        print(x)
        x = x + 1

    test_arr.append(test)
    match_gps.append(best_gps)
    j = j + 1


diff = []
for i in range(0, len(live)):
    diff.append(geodesic(live[i], match_gps[i]).meters)

data_route = []
data_gps = []
data_route.append(0)
data_gps.append(0)
for i in range(1, len(match_gps)):
    data_route.append(geodesic(match_gps[i - 1], match_gps[i]).meters)
    data_gps.append(data_route[i] + data_gps[i - 1])

live_route = []
live_gps = []
live_route.append(0)
live_gps.append(0)
for i in range(1, len(live)):
    live_route.append(geodesic(live[i - 1], live[i]).meters)
    live_gps.append(live_route[i] + live_gps[i - 1])

df = pd.DataFrame(diff)
df.to_csv('file.csv',index=False)
df1 = pd.DataFrame(match_gps)
df1.to_csv('file1.csv',index=False)
df2 = pd.DataFrame(data_gps)
df2.to_csv('data_route.csv',index=False)
df3 = pd.DataFrame(live_gps)
df3.to_csv('live_route.csv',index=False)


print("--- %s seconds ---" % (time.time() - start_time))
print(test_arr)
plt.plot(diff[:], 'o-', label='Difference in route')
plt.legend()
plt.show()
