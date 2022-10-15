import cv2
import numpy as np
import os

img_dir = './chessboard/'
img_list = os.listdir(img_dir)

img_path_list = [os.path.join(img_dir, i) for i in img_list]
print(img_path_list)
images=[]
for img_path in img_path_list:
    img = cv2.imread(img_path)
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append((img_gray,img))

print('\nWas able to find %s images' % len(images))
#input()

pattern_size = (9, 6)
obj_points = []
img_points = []
# Assumed object points relation
a_object_point = np.zeros((pattern_size[1] * pattern_size[0], 3),
                          np.float32)
a_object_point[:, :2] = np.mgrid[0:pattern_size[0],
                                 0:pattern_size[1]].T.reshape(-1, 2)

# Termination criteria for sub pixel corners refinement
stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 30, 0.001)

print('Finding points ', end='')
debug_images = []
DEBUG = True
for image, color_image in images:
    found, corners = cv2.findChessboardCorners(image, pattern_size, None)
    if found:
        cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), stop_criteria)
        obj_points.append(a_object_point)
        img_points.append(corners)
        print('.', end='')
    else:
        print('-', end='')

    if DEBUG:
        cv2.drawChessboardCorners(color_image, pattern_size, corners, found)
        debug_images.append(color_image)
        cv2.imshow('corners'+str(found),  cv2.resize(color_image, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #sys.stdout.flush()
print('\nWas able to find points in %s images' % len(img_points))
input()
img_size = (images[0][0].shape[1], images[0][0].shape[0])
print('img sz', img_size)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                       img_points,
                                                       img_size,
                                                       None,
                                                       None)
print('***************************calib mtx', mtx)
print('***************************calib dist', dist)


test_img_list = os.listdir('test_img')
test_img_list = ['test_img/'+ i for i in test_img_list]
for color_path in test_img_list:
    color_image = cv2.imread(color_path)

    color_image = cv2.resize(color_image, (800, 600))
    w_s = color_image.shape[1] / img_size[0]
    h_s = color_image.shape[0] / img_size[1]
    print(w_s, h_s)
    mtx_s_w = mtx[0][:]*w_s
    mtx_s_h = mtx[1][:]*h_s
    mtx_s_s = np.array([0.0,0.0,1.0])
    mtx_s = np.stack([mtx_s_w, mtx_s_h, mtx_s_s], 0)

    print(mtx_s)
    undist = cv2.undistort(color_image, mtx_s, dist, None, mtx_s)

    pad = np.zeros([undist.shape[0], undist.shape[1]+color_image.shape[1], color_image.shape[2]], undist.dtype)
    pad[:, :color_image.shape[1]] = color_image
    pad[:,color_image.shape[1]:] = undist

    #cv2.imshow('undist', undist)
    cv2.imshow('undist', pad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

f=open('calibration','w')
f.write('ImageSize:\n')
f.write(str(img_size)+'\n')
f.write('CameraMatrix:\n')
f.write(str(mtx)+'\n')
f.write('DistCoeffs:\n')
f.write(str(dist)+'\n')
f.close()

