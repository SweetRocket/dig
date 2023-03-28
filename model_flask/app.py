# Flask import
from flask import Flask
from flask import g

# Flask socks
from flask_sock import Sock
from simple_websocket import Server as WSServer
from simple_websocket.ws import ConnectionClosed

# other import
import cv2
import numpy as np
import torch
import cmapy
import json
import platform
import base64
import time
from threading import Thread


class Singleton(type):
    """ 
    Simple Singleton that keep only one value for all instances
    """

    def __init__(cls, name, bases, dic):
        super(Singleton, cls).__init__(name, bases, dic)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance


class Models(metaclass=Singleton):
    """
    Models

    모델을 로딩하며, 미리 불러와 저장하는 클래스
    """

    # 클래스 초기화
    def __init__(self, equip_model_path='models/equip/best.pt', handmotion_model_path='models/handmotion/best.pt', confidence=0.5):
        # 연산 device 설정
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # torch hub를 사용한 model 로드
        self.equip_model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=equip_model_path)
        self.handmotion_model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=handmotion_model_path)

        # 해당 모델을 사용할 device로 옮김
        self.equip_model.to(self.device)
        self.handmotion_model.to(self.device)

        # 모델을 eval 모드로 설정
        self.equip_model.eval()
        self.handmotion_model.eval()

        # confidence 설정
        self.equip_model.conf = confidence
        self.handmotion_model.conf = confidence

    # 이미지에서 Object Detection을 수행
    def detect(self, img, mhi_img, size=640):
        ret = []

        # 이미지 predict
        results = {
            'handmotion': self.handmotion_model(mhi_img, size),
            'equip': self.equip_model(img, size)
        }

        # 한 return 값으로 합침
        for model_name, result in results.items():
            for i in range(len(result.xyxy[0])):
                r = result.pandas().xyxy[0].iloc[i, :].values.tolist()
                x1, y1, x2, y2, confidence, cls_id, cls_name = r
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence, cls_id = float(confidence), int(cls_id)

                ret.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': confidence,
                    'cls_id': cls_id,
                    'cls_name': cls_name,
                    'model': model_name,
                })

        return ret


class Streamer():
    """
    모델을 감지한 값을 실시간으로 스트리밍하기 위한 클래스
    """

    # 클래스 초기화
    def __init__(self, device_id, pos=15, cmap='nipy_spectral', confidence=0.5):
        # 기본 설정
        self.device = None
        self.device_id = device_id
        self.started = False

        # thread
        self.detect_thread = None
        self.ws_thread = None

        # MHI용 설정
        self.pos = pos
        self.cmap = cmapy.cmap(cmap)

        # 배경 제거 모델
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # 모델
        self.models = Models(confidence=confidence)

        # ws
        self.ws_client_list = []

        # status
        self.started = False
        self.worked = False

    # 실행
    # Thread를 사용하여, 실시간으로 frame을 받아와 저장
    def run(self):
        self.stop()

        if self.device is None:
            if platform.system() == 'Windows':
                self.device = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            else:
                self.device = cv2.VideoCapture(self.device_id)

            self.width = int(self.device.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.device.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.mhi_history = np.zeros(
                (self.height, self.width), dtype=np.int16)

            # fps
            try:
                self.fps_delay = 1/self.device.get(cv2.CAP_PROP_FPS)
            except:
                self.fps_delay = 1/30

        # Thread 시작
        self.thread_start()

        self.started = True

    def thread_start(self):
        if self.detect_thread is None:
            self.detect_thread = Thread(target=self.update, args=())
            self.detect_thread.daemon = False
            self.detect_thread.start()

        if self.ws_thread is None:
            self.ws_thread = Thread(target=self.ws_update, args=())
            self.ws_thread.daemon = True
            self.ws_thread.start()

    # Thread 및 Device 정지
    def stop(self):
        self.started = False

        if self.device is not None:
            self.device.release()

    # Thread 함수
    def update(self):
        while True:
            # 시작할때까지 대기
            if not self.started:
                continue
            try:
                self.frame = self.get_frame()
                self.mhi_frame = self.get_mhi()
                self.detected = self.detect()
                self.boxed_frame = self.draw_box()
                self.worked = True
            except Exception as e:
                print(e)
                continue

    def ws_update(self):
        while True:
            if not self.started or not self.worked:
                continue
            time.sleep(self.fps_delay)
            clients = self.ws_client_list.copy()
            for client in clients:
                try:
                    j = self.prepare_ws_data()
                    client.send(j)
                except ConnectionClosed:
                    self.remove_ws_client(client)
                except Exception as e:
                    client.send(json.dumps(
                        {'status': 'error', 'error': str(e)}))
                    client.close()
                    self.remove_ws_client(client)

    # convert numpy cv2 array to jpg base64 string
    def img_array_to_jpg_base64(self, img):
        jpg = cv2.imencode('.jpg', img)[1]
        return base64.b64encode(jpg).decode('utf-8')

    def prepare_ws_data(self):
        return json.dumps({
            'status': 'ok',
            'detected': self.detected,
            'boxed': self.img_array_to_jpg_base64(self.boxed_frame),
        })

    def add_ws_client(self, client):
        self.ws_client_list.append(client)

    def remove_ws_client(self, client):
        self.ws_client_list.remove(client)

    # 빈 frame
    def blank(self):
        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    # 빈 jpg
    def blank_jpg(self):
        return cv2.imencode('.jpg', self.blank())[1]

    # frame을 받아와서 queue에 저장
    def get_frame(self):
        # 진행전 전처리
        if self.device is None or self.device.isOpened() is False:
            frame = self.blank()
        else:
            ret, frame = self.device.read()
            if ret != True:
                frame = self.blank()
        return frame

    # mhi stands for motion history image
    def get_mhi(self):
        # 배경 제거 마스크 계산
        fgmask = self.fgbg.apply(self.frame)

        # make 24 frame of history based on fgmask
        self.mhi_history = np.where(
            fgmask == 255, 255 + self.pos, self.mhi_history)
        self.mhi_history = np.where(
            self.mhi_history > -100, self.mhi_history - self.pos, self.mhi_history)

        # make clip for safety
        self.mhi_history = np.clip(self.mhi_history, 0, 255)
        history_frame = self.mhi_history.astype(np.uint8)

        # color map
        color_map_history = cv2.applyColorMap(history_frame, self.cmap)

        return color_map_history

    # 감지
    def detect(self):
        return self.models.detect(self.frame, self.mhi_frame)

    # bbox 그리기
    def draw_box(self):
        f = self.frame.copy()

        for d in self.detect():
            name = f'{d["model"]}_{d["cls_name"]}, {d["confidence"]:.2f}'

            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(f, name, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return f


# 중복 방지 및 context 유지를 위하여 flask에서 사용하는 global 변수를 사용
def get_streamer(device_id, pos=15, cmap='nipy_spectral', confidence=0.5) -> Streamer:
    n = f'streamer_{device_id}'
    s = getattr(g, n, None)
    if s is None:
        s = Streamer(device_id, pos, cmap, confidence)
        setattr(g, n, s)
    return s


# flask app
app = Flask(__name__)
sock = Sock(app)

# json을 stream (via WebSocket)


@sock.route('/feed/<int:device_id>')
def json_feed(ws: WSServer, device_id):
    # streamer 불러옴
    streamer = get_streamer(device_id)

    # streamer가 시작되지 않았다면 시작
    if streamer.started is False:
        streamer.run()

    try:
        streamer.add_ws_client(ws)
        while True:
            data = ws.receive()
            if data == 'stop':
                ws.send(json.dumps({'status': 'stopped'}))
                break
        streamer.remove_ws_client(ws)
    except Exception as e:
        print(e)
        ws.send(json.dumps({'status': 'error', 'error': str(e)}))
    finally:
        ws.close()

    # 모든 client가 종료되면 streamer 종료
    if len(streamer.ws_client_list) == 0:
        streamer.stop()


# Flask 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)  # debug=True
