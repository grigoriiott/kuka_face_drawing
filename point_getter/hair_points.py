import torch
from hair_model.nets.MobileNetV2_unet import MobileNetV2_unet
import numpy as np
import cv2

def load_model():
    model = MobileNetV2_unet(None)
    state_dict = torch.load("hair_model/checkpoints/model.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def img_prep(image):
    image = image/255
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LINEAR)
    image = np.expand_dims(np.moveaxis(image.astype(np.float32), -1, 0), 0)
    return image

def check_0_hair(msk_filter, tmp, j, kernel_size):
    top_flag = False
    for i_m in range(msk_filter.shape[0]):
        for j_m in range(msk_filter.shape[1]):
            if msk_filter[i_m][j_m] == 1:
                tmp[0] = [i_m, j*kernel_size+j_m]
                top_flag = True
                break
        if top_flag == True:
            break
    return tmp

def check_2_hair(msk_filter, tmp, j, kernel_size):
    bottom_flag = False
    for i_m in range(msk_filter.shape[0]-1, -1, -1):
        for j_m in range(msk_filter.shape[1]-1, -1, -1):
            if msk_filter[i_m][j_m] == 1:
                tmp[2] = [i_m, j*kernel_size+j_m]
                bottom_flag = True
                break
        if bottom_flag == True:
            break
    return tmp

def check_1_hair(msk_filter, tmp, j, kernel_size):
    if tmp[0] != 0:
        point_i = tmp[0][0]+int((tmp[2][0] - tmp[0][0])/2)
        left = 0
        right = 0
        for j_m in range(msk_filter.shape[1]):
            if msk_filter[point_i, j_m] == 1:
                left = j_m
                break
        for j_m in range(msk_filter.shape[1]-1, -1, -1):
            if msk_filter[point_i, j_m] == 1:
                right = j_m
                break
        point_j = int((right-left)/2)+left
        if (left+right) == 0:
            tmp[1] = [0, 0]
        else:
            tmp[1] = [point_i, point_j+j*kernel_size]
    return tmp

def hair_filter_2(mask):
    landmarks_list = []
    tmp = [[0, 0], [0, 0], [0, 0]] # top, middle, down
    kernel_size = 50
    prescaler = 2
    for j in range(int(mask.shape[1]/(kernel_size/prescaler))):
        tmp = [0, 0, 0]
        msk_filter = mask[:, j*int(kernel_size/prescaler):j*int(kernel_size/prescaler)+kernel_size]
        tmp = check_0_hair(msk_filter, tmp, j, int(kernel_size/prescaler))
        tmp = check_2_hair(msk_filter, tmp, j, int(kernel_size/prescaler))
        tmp = check_1_hair(msk_filter, tmp, j, int(kernel_size/prescaler))
        if tmp[0] != 0 and (tmp[1][0]+tmp[1][1]) != 0:
            if tmp[1] == 0:
                tmp[1] = tmp[0]
            landmarks_list.append(tmp)
    return np.asarray(landmarks_list)

def get_hair_points(img):
    model = load_model()
    image_prep = img_prep(img)
    seg_res = model.forward(torch.from_numpy(image_prep)).detach().numpy()[0]

    seg_res[0] = seg_res[0] - seg_res[0].min()
    seg_res[0] = seg_res[0]/seg_res[0].max()
    seg_res[1] = seg_res[1] - seg_res[1].min()
    seg_res[1] = seg_res[1]/seg_res[1].max()
    seg_res[2] = seg_res[2] - seg_res[2].min()
    seg_res[2] = seg_res[2]/seg_res[2].max()

    seg_res[1] = seg_res[1]>0.4
    seg_res[0] = seg_res[0]>0.7
    seg_res[2] = seg_res[2]>0.5
    tmp1 = np.logical_xor(seg_res[2], seg_res[0])
    tmp2 = np.logical_xor(seg_res[2], seg_res[1])
    hair_mask = np.logical_and(seg_res[2], tmp1)
    hair_mask = np.logical_and(hair_mask, tmp2)
    hair_mask = hair_mask.astype(np.float32)

    filtered = cv2.medianBlur(cv2.medianBlur(hair_mask, 5), 5)
    filtered = cv2.medianBlur(cv2.medianBlur(filtered, 5), 5)
    filtered = cv2.resize(filtered, (200, 200), interpolation=cv2.INTER_LINEAR)

    res_hair = hair_filter_2(filtered)

    return res_hair

if __name__ == '__main__':
    img = cv2.imread("7.jpg")
    res_hair = get_hair_points(img)
    print(res_hair)