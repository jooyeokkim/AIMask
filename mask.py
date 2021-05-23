import cv2
import math
import numpy as np

def get_mask_img(background_file, mask_file):
    # 학습된 xml file
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade/Nariz.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade/Mouth.xml')

    img = cv2.imread(background_file) # background img
    mask_img = cv2.imread(mask_file) # mask img
    (h, w) = img.shape[:2]

    # img 크기 조절
    if w > h:
        x = 720
        y = int((x*h)/w)
        dim = (x, y)
    else:
        y = 720
        x = int((y * w)/h)
        dim = (x, y)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) # 이미지 인식이 더 잘 되게 회색으로 변경

    faces = face_cascade.detectMultiScale(img_gray,1.05,2) # 얼굴 인식

    for (fx, fy, fw, fh) in faces:
        # 추출 영역 약간 확장 - 마스크 이미지 이탈 방지
        fx = fx - int(0.1*fw)
        fy = fy - int(0.1*fh)
        fw = fw + int(0.2*fw)
        fh = fh + int(0.26*fh)

        face = resized_img[fy: fy + fh, fx: fx + fw] # 얼굴 영역 추출
        face_gray = img_gray[fy: fy + fh, fx: fx + fw] # 회색 이미지의 얼굴 영역 추출 

        #코, 입 인식
        nose = nose_cascade.detectMultiScale(face_gray,1.18,2)
        mouth = mouth_cascade.detectMultiScale(face_gray,1.2,2)

        # 얼굴 영역에 코와 입 모두 있어야 통과
        if not len(nose) or not len(mouth):
            continue

        # AI가 인식한 코 후보들 중 얼굴 한가운데에 있는 후보를 코로 선정
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

        # AI가 인식한 입 후보들 중 얼굴 제일 아래에 있는 후보를 입으로 선정
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

        # 마스크 위치 선정
        (m_h, m_w, m_c) = mask_img.shape
        center_mask = (int((center_nose[0]+center_mouth[0])/2), int((center_nose[1]+center_mouth[1])/2*1.1)) # 코와 입 중간의 약간 아래쪽
        center_distance = int(math.sqrt(math.pow(center_nose[0]-center_mouth[0],2)+math.pow(center_nose[1]-center_mouth[1],2))) # 코와 입 사이의 길이
        center_angle = np.rad2deg(math.asin(abs(center_mouth[0]-center_nose[0])/center_distance)) # 코와 입이 이루는 각
        print("distance : "+str(center_distance))
        print("angle : "+str(center_angle))

        # 마스크 크기 설정
        resized_mask_img = cv2.resize(mask_img,(int(3.5*center_distance),int(2.3*center_distance)))
        height, width = resized_mask_img.shape[:2]
        print((center_nose[1], center_mouth[1]))

        # 마스크 회전
        if center_nose[0]<=center_mouth[0]:
            M = cv2.getRotationMatrix2D((width/2, height/2),center_angle,1)
        elif center_nose[0]>center_mouth[0]:
            M = cv2.getRotationMatrix2D((width/2, height/2),360-center_angle,1)
        rotated_mask_img = cv2.warpAffine(resized_mask_img,M,(width,height))

        # 회전된 마스크 이미지에서 마스크 부분만 추출 - 비트맵 효과
        # mask : 배경 이미지 위에 씌우는 이미지
        gray_mask = cv2.cvtColor(rotated_mask_img,cv2.COLOR_BGR2GRAY)
        _, MASK_inv = cv2.threshold(gray_mask,10,255,cv2.THRESH_BINARY_INV) # 색상 값이 10 이상인 부분들은 255로 변경

        background_height, background_width, _ = face.shape
        mask_height, mask_width, _ = rotated_mask_img.shape

        # 마스크 중간 좌표 저장
        mask_y = center_mask[0]-int(mask_width/2) 
        mask_x = center_mask[1]-int(mask_height/2)

        # 마스크 이미지를 덧붙일 부분 추출
        roi = face[mask_x: mask_x+mask_height, mask_y: mask_y+mask_width]
        
        # 마스크 합성(add)
        try :
            roi_mask = cv2.add(rotated_mask_img, roi, mask=MASK_inv)
            result = cv2.add(roi_mask, rotated_mask_img)
            np.copyto(roi, result)
        except Exception as e:
            print(e)

    # 이미지 최종 저장
    result_url="result_image/"+background_file.split('/')[-1]
    cv2.imwrite(result_url, resized_img)
    cv2.destroyAllWindows()