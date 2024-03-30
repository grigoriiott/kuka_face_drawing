import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from get_image import *
from hair_points import *

def prepare_layer(idxs, landmarks):
    layer = []
    for i in range(len(landmarks)):
        if landmarks[i][0] in idxs:
            layer.append(landmarks[i])
    return np.asarray(layer)

def sort_lists(layer, idxs):
    tmp = []
    for i in range(len(layer)):
        idx = idxs[i]
        tmp.append(layer[np.argmax(layer[:, 0] == idx)])
    return np.asarray(tmp)

def sobel_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    total_grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (total_grad * 255 / total_grad.max()).astype(np.uint8)
    return grad_norm

def check_0(mask, tmp, i, j, kernel_size):
    top_flag = False
    for i_m in range(mask.shape[0]):
        for j_m in range(mask.shape[1]):
            if mask[i_m][j_m] == 1:
                tmp[0] = [i*kernel_size+i_m, j*kernel_size+j_m]
                top_flag = True
                break
        if top_flag == True:
            break
    return tmp

def check_2(mask, tmp, i, j, kernel_size):
    bottom_flag = False
    for i_m in range(mask.shape[0]-1, -1, -1):
        for j_m in range(mask.shape[1]-1, -1, -1):
            if mask[i_m][j_m] == 1:
                tmp[2] = [i*kernel_size+i_m, j*kernel_size+j_m]
                bottom_flag = True
                break
        if bottom_flag == True:
            break
    return tmp

def check_1(mask, tmp, i, j, kernel_size):
    for j_m in range(mask.shape[1]):
        if mask[int(kernel_size/2)][j_m] == True:
            tmp[1] = [i*kernel_size+int(kernel_size/2), j*kernel_size+j_m]
    return tmp

def go_by_filter(image):
    landmarks_list = []
    tmp = [[0, 0], [0, 0], [0, 0]] # top, middle, down
    kernel_size = 10
    for i in range(int(image.shape[0]/kernel_size)):
        for j in range(int(image.shape[1]/kernel_size)):
            tmp = [0, 0, 0] # top, middle, down
            mask = image[i*kernel_size:i*kernel_size+kernel_size, j*kernel_size:j*kernel_size+kernel_size]
            tmp = check_0(mask, tmp, i, j, kernel_size)
            tmp = check_2(mask, tmp, i, j, kernel_size)
            tmp = check_1(mask, tmp, i, j, kernel_size)
            if tmp[0] != 0:
                if tmp[1] == 0:
                    tmp[1] = tmp[0]
                landmarks_list.append(tmp)
    return landmarks_list

def prepare_eye(idxs, landmarks, left = True):
    layer = []
    for i in range(len(landmarks)):
        if landmarks[i][0] in idxs:
            layer.append(landmarks[i])
    layer = np.asarray(layer)
    layer = sort_lists(layer, idxs)
    layer_sorted = []
    layer_sorted.append(np.asarray([layer[0][2], layer[0][1]]))
    x_point = int((layer[1][2]-layer[0][2])/2)+layer[0][2]
    y_point = int((layer[1][1]-layer[0][1])/2)+layer[0][1]
    if left:
        y_point -= 2
    else:
        y_point += 2
    layer_sorted.append(np.asarray([x_point, y_point]))
    layer_sorted.append(np.asarray([layer[1][2], layer[1][1]]))
    return np.asarray(layer_sorted)



def get_all_points():
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, static_image_mode=True)

    #change to use image from the folder
    #img = image_getter()
    img = cv2.imread("7.jpg")

    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
    
    sobel_three = sobel_filter(cv2.cvtColor(cv2.medianBlur(cv2.medianBlur(img, 5), 5), cv2.COLOR_BGR2GRAY))
    sobel_three = (sobel_three/255)>0.3

    res = go_by_filter(sobel_three)

    res_hair = get_hair_points(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_lms = faceMesh.process(img)

    ih, iw, ic = img.shape
    lms_list = []
    for idx, lm in enumerate(res_lms.multi_face_landmarks[0].landmark):
        x, y = int(lm.x*iw), int(lm.y*ih)
        lms_list.append([idx, x, y])
    lms_list = np.asarray(lms_list)

    first_eye_layer_idx = [33, 246, 161, 160, 159, 157, 
                        173, 133, 155, 153, 145, 7]
    first_eye_layer = prepare_layer(first_eye_layer_idx, lms_list)
    first_eye_layer = sort_lists(first_eye_layer, first_eye_layer_idx)

    second_eye_layer_idx = [263, 390, 373, 374, 380, 382, 
                        362, 398, 384, 386, 387, 466]
    second_eye_layer = prepare_layer(second_eye_layer_idx, lms_list)
    second_eye_layer = sort_lists(second_eye_layer, second_eye_layer_idx)

    lips_first_layer_idx = [0, 267, 270, 409, 375, 405, 
                        17, 181, 91, 61, 185, 37]
    lips_first_layer = prepare_layer(lips_first_layer_idx, lms_list)
    lips_first_layer = sort_lists(lips_first_layer, lips_first_layer_idx)

    lips_second_layer_idx = [13, 312, 310, 415, 318, 
                         402, 14, 87, 88, 191, 
                         80, 82]
    lips_second_layer = prepare_layer(lips_second_layer_idx, lms_list)
    lips_second_layer = sort_lists(lips_second_layer, lips_second_layer_idx)

    #nose_layer_idx = [114, 198, 131, 115, 
    #                  20, 94, 250, 309,
    #                  344, 360, 420, 343]
    #nose_layer_one = prepare_layer(nose_layer_idx, lms_list)
    #nose_layer_one = sort_lists(nose_layer_one, nose_layer_idx)

    #[[0, 0], [0, 0], [0, 0]]
    eyebrow_layer_first_idx = [276, 282, 285]
    eyebrow_layer_first = prepare_layer(eyebrow_layer_first_idx, lms_list)
    eyebrow_layer_first = sort_lists(eyebrow_layer_first, eyebrow_layer_first_idx)
    eyebrow_layer_first = eyebrow_layer_first[:, 1:]
    tmp_1 = eyebrow_layer_first[:, 0].copy()
    eyebrow_layer_first[:, 0] = eyebrow_layer_first[:, 1]
    eyebrow_layer_first[:, 1] = tmp_1

    eyebrow_layer_second_idx = [46, 52, 55]
    eyebrow_layer_second = prepare_layer(eyebrow_layer_second_idx, lms_list)
    eyebrow_layer_second = sort_lists(eyebrow_layer_second, eyebrow_layer_second_idx)
    eyebrow_layer_second = eyebrow_layer_second[:, 1:]
    tmp_1 = eyebrow_layer_second[:, 0].copy()
    eyebrow_layer_second[:, 0] = eyebrow_layer_second[:, 1]
    eyebrow_layer_second[:, 1] = tmp_1

    eyebrow_layer_up1_idx = [276, 334, 336]
    eyebrow_layer_up1 = prepare_layer(eyebrow_layer_up1_idx, lms_list)
    eyebrow_layer_up1 = sort_lists(eyebrow_layer_up1, eyebrow_layer_up1_idx)
    eyebrow_layer_up1 = eyebrow_layer_up1[:, 1:]
    tmp_1 = eyebrow_layer_up1[:, 0].copy()
    eyebrow_layer_up1[:, 0] = eyebrow_layer_up1[:, 1]
    eyebrow_layer_up1[:, 1] = tmp_1

    eyebrow_layer_up2_idx = [46, 66, 107]
    eyebrow_layer_up2 = prepare_layer(eyebrow_layer_up2_idx, lms_list)
    eyebrow_layer_up2 = sort_lists(eyebrow_layer_up2, eyebrow_layer_up2_idx)
    eyebrow_layer_up2 = eyebrow_layer_up2[:, 1:]
    tmp_1 = eyebrow_layer_up2[:, 0].copy()
    eyebrow_layer_up2[:, 0] = eyebrow_layer_up2[:, 1]
    eyebrow_layer_up2[:, 1] = tmp_1

    nose_in1_idx = [241, 238, 239]
    nose_in1_layer = prepare_layer(nose_in1_idx, lms_list)
    nose_in1_layer = sort_lists(nose_in1_layer, nose_in1_idx)
    nose_in1_layer = nose_in1_layer[:, 1:]
    tmp_1 = nose_in1_layer[:, 0].copy()
    nose_in1_layer[:, 0] = nose_in1_layer[:, 1]
    nose_in1_layer[:, 1] = tmp_1

    nose_in2_idx = [461, 458, 459]
    nose_in2_layer = prepare_layer(nose_in2_idx, lms_list)
    nose_in2_layer = sort_lists(nose_in2_layer, nose_in2_idx)
    nose_in2_layer = nose_in2_layer[:, 1:]
    tmp_1 = nose_in2_layer[:, 0].copy()
    nose_in2_layer[:, 0] = nose_in2_layer[:, 1]
    nose_in2_layer[:, 1] = tmp_1

    nose_layer3_idx = [363, 344, 462]
    nose_layer_three = prepare_layer(nose_layer3_idx, lms_list)
    nose_layer_three = sort_lists(nose_layer_three, nose_layer3_idx)
    nose_layer_three = nose_layer_three[:, 1:]
    tmp_1 = nose_layer_three[:, 0].copy()
    nose_layer_three[:, 0] = nose_layer_three[:, 1]
    nose_layer_three[:, 1] = tmp_1

    nose_layer2_idx = [134, 115, 242]
    nose_layer_two = prepare_layer(nose_layer2_idx, lms_list)
    nose_layer_two = sort_lists(nose_layer_two, nose_layer2_idx)
    nose_layer_two = nose_layer_two[:, 1:]
    tmp_1 = nose_layer_two[:, 0].copy()
    nose_layer_two[:, 0] = nose_layer_two[:, 1]
    nose_layer_two[:, 1] = tmp_1

    nose_layer1_idx = [6, 195, 4]
    nose_layer_one = prepare_layer(nose_layer1_idx, lms_list)
    nose_layer_one = sort_lists(nose_layer_one, nose_layer1_idx)
    nose_layer_one = nose_layer_one[:, 1:]
    tmp_1 = nose_layer_one[:, 0].copy()
    nose_layer_one[:, 0] = nose_layer_one[:, 1]
    nose_layer_one[:, 1] = tmp_1

    oval_layer_idx = [127, 93, 58, 136, 
                  149, 148, 377, 378, 
                  365, 288, 323, 356]
    oval_layer = prepare_layer(oval_layer_idx, lms_list)
    oval_layer = sort_lists(oval_layer, oval_layer_idx)

    eye_left_1_idx = [159, 145] #center, left, max, min
    eye_left_1_layer = prepare_eye(eye_left_1_idx, lms_list)

    eye_left_2_idx = [158, 153] #center, left, max, min
    eye_left_2_layer = prepare_eye(eye_left_2_idx, lms_list, False)

    eye_right_1_idx = [385, 380] #center, left, max, min
    eye_right_1_layer = prepare_eye(eye_right_1_idx, lms_list)

    eye_right_2_idx = [386, 374] #center, left, max, min
    eye_right_2_layer = prepare_eye(eye_right_2_idx, lms_list, False)

    res.append(eyebrow_layer_first)
    res.append(eyebrow_layer_second)
    res.append(eyebrow_layer_up1)
    res.append(eyebrow_layer_up2)
    res.append(nose_in1_layer)
    res.append(nose_in2_layer)
    res.append(nose_layer_three)
    res.append(nose_layer_two)
    res.append(nose_layer_one)
    res.append(eye_left_1_layer)
    res.append(eye_left_2_layer)
    res.append(eye_right_1_layer)
    res.append(eye_right_2_layer)
    res = np.asarray(res)

    res_sum = np.append(res, res_hair, axis=0)

    return [res_sum, first_eye_layer, second_eye_layer, 
            lips_first_layer, lips_second_layer, oval_layer]
    
if __name__=="__main__":
    res = get_all_points()
    print(res)

