import cv2
import numpy as np
import time

def image_getter():
    #cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    start_time = 0
    cur_time = 0
    start_flag = False
    got_flag = False

    coords = (100, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 0, 255)
    thicknes = 2

    ret_img = None

    while (cur_time-start_time)<7:
        if start_flag == False:
            start_time = time.time()
            cur_time = time.time()
            start_flag = True
        success, img = cap.read()
        #print(start_time, cur_time, cur_time-start_time)
        h, w, c = img.shape
        #print(img.shape)
        img = img[:, 80:640-80, :]
        if (cur_time-start_time)<6:
            img = cv2.putText(img, str(round(cur_time-start_time)), coords, 
                            font, font_scale, color, thicknes, cv2.LINE_AA)
        if (cur_time-start_time)>6.5 and got_flag == False:
            ret_img = img.copy()
            got_flag == True

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        cur_time = time.time()

    cv2.imwrite('got_img/last_img.jpg', ret_img)
    return ret_img

if __name__=="__main__":
    res = image_getter()
    cv2.imshow("ret img", res)
    cv2.waitKey(0)