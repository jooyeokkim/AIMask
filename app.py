from flask import Flask, render_template, request, send_file
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.image import imread
from mask import get_mask_img
import cv2

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/test', methods=['POST','GET'])
def test():
    print("여기로 전송")
    myfile = request.form['myfile']
    url = 'http://121.153.146.164:8081/demo/resources/upload/'+myfile
    save_url = 'spring_image/'+myfile
    try:
        img = urllib.request.urlopen(url).read()
        with open(save_url,"wb") as f:
            f.write(img)
            print("저장 완료")
    except urllib.error.HTTPError:
        print("접근할 수 없는 url입니다.")

    image = cv2.imread(save_url,cv2.IMREAD_UNCHANGED)
    resized_image = cv2.resize(image,(300,300))
    cv2.imshow(myfile,resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return '전송 최종 완료'

@app.route('/aimask', methods=['POST','GET'])
def mask():
    print("여기로 전송")
    myfile = request.form['myfile']
    url = 'http://121.153.146.164:8081/demo/resources/upload/'+myfile
    save_url = 'image/'+myfile
    try:
        img = urllib.request.urlopen(url).read()
        with open(save_url,"wb") as f:
            f.write(img)
            print("저장 완료")
    except urllib.error.HTTPError:
        print("접근할 수 없는 url입니다.")

    get_mask_img(save_url,'image/mask_image.png')
    return send_file(save_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0')