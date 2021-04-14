import cv2
import math
import numpy as np

def get_mask_img(background_file, mask_file):
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye_tree_eyeglasses.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade/Nariz.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade/Mouth.xml')

    img = cv2.imread(background_file)
    mask_img = cv2.imread(mask_file)
    (h, w) = img.shape[:2]

    if w > h:
        x = 720
        y = int((x*h)/w)
        dim = (x, y)
    else:
        y = 720
        x = int((y * w)/h)
        dim = (x, y)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray,1.09,2)

    for (fx, fy, fw, fh) in faces:
        fx = fx - int(0.05*fw)
        fy = fy - int(0.1*fh)
        fw = fw + int(0.1*fw)
        fh = fh + int(0.2*fh)
        face = resized_img[fy: fy + fh, fx: fx + fw]
        face_gray = img_gray[fy: fy + fh, fx: fx + fw]

        eyes = eye_cascade.detectMultiScale(face_gray,1.1,2)
        nose = nose_cascade.detectMultiScale(face_gray,1.2,2)
        mouth = mouth_cascade.detectMultiScale(face_gray,1.3,2)

        if not len(nose) or not len(mouth):
            continue

        # cv2.rectangle(resized_img, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        print('eyes : '+str(len(eyes))+' results found')
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        print('nose : '+str(len(nose))+' results found')
        min_distance = 10000
        nx, ny, nw, nh = (0, 0, 0, 0)
        for (_nx, _ny, _nw, _nh) in nose:
            center_nx = int(_nx+_nw/2)
            center_ny = int(_ny+_nh/2)
            center_fx = int(fw/2)
            center_fy = int(fh/2)
            distance = math.sqrt(math.pow(center_nx-center_fx,2)+math.pow(center_ny-center_fy,2))
            if distance<min_distance:
                nx, ny, nw, nh = (_nx, _ny, _nw, _nh)
                min_distance=distance
        center_nose = (int(nx+nw/2), int(ny+nh/2))
        # cv2.rectangle(face, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
        # cv2.circle(face,(center_nose[0], center_nose[1]), 2, (0, 0, 255), 2)

        print('mouth : '+str(len(mouth))+' results found')
        max_my=0
        mx, my, mw, mh = (0, 0, 0, 0)
        for (_mx, _my, _mw, _mh) in mouth:
            if _my>max_my:
                mx, my, mw, mh = (_mx, _my, _mw, _mh)
                max_my=_my
        center_mouth = (int(mx+mw/2), int(my+mh/2))
        # cv2.rectangle(face, (mx, my), (mx + mw, my + mh), (0, 0, 0), 2)
        # cv2.circle(face,(center_mouth[0], center_mouth[1]), 2, (0, 0, 0), 2)

        (m_h, m_w, m_c) = mask_img.shape
        center_mask = (int((center_nose[0]+center_mouth[0])/2), int((center_nose[1]+center_mouth[1])/2))
        center_distance = int(math.sqrt(math.pow(center_nose[0]-center_mouth[0],2)+math.pow(center_nose[1]-center_mouth[1],2)))
        center_angle = np.rad2deg(math.asin(abs(center_mouth[0]-center_nose[0])/center_distance))
        print("distance : "+str(center_distance))
        print("angle : "+str(center_angle))

        resized_mask_img = cv2.resize(mask_img,(int(3.5*center_distance),int(2.5*center_distance)))
        height, width = resized_mask_img.shape[:2]
        print((center_nose[1], center_mouth[1]))
        if center_nose[0]<=center_mouth[0]:
            M = cv2.getRotationMatrix2D((width/2, height/2),center_angle,1)
        elif center_nose[0]>center_mouth[0]:
            M = cv2.getRotationMatrix2D((width/2, height/2),360-center_angle,1)
        rotated_mask_img = cv2.warpAffine(resized_mask_img,M,(width,height))
        gray_mask = cv2.cvtColor(rotated_mask_img,cv2.COLOR_BGR2GRAY)
        _, MASK_inv = cv2.threshold(gray_mask,10,255,cv2.THRESH_BINARY_INV)

        background_height, background_width, _ = face.shape
        mask_height, mask_width, _ = rotated_mask_img.shape

        y = center_mask[0]-int(mask_width/2) 
        x = center_mask[1]-int(mask_height/2)

        roi = face[x: x+mask_height, y: y+mask_width]
        try :
            roi_mask = cv2.add(rotated_mask_img, roi, mask=MASK_inv)
            result = cv2.add(roi_mask, rotated_mask_img)
            np.copyto(roi, result)
        except Exception as e:
            print(e)

    cv2.imshow('img', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()