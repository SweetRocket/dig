# Flask import
from queue import LifoQueue
from flask import Flask, render_template
from flask import g

# Flask socks
from flask_sock import Sock
from simple_websocket import Server as WSServer

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

client_list = {}
output_queue = LifoQueue(maxsize=128)


# 오류 발생시 출력만 하고 넘어가기 위한 decorator
def tryExceptDecorator(*deco_args, **deco_kwargs):
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                if deco_kwargs.get('print_exception', False):
                    print(e)
        return wrapper
    return decorator


# 오류 발생시 출력만 하고 넘어가기 위한 function
def tryExceptFunction(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if kwargs.get('print_exception', False):
            print(e)


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


def fire_and_forget(func):
    """
    fire_and_forget decorator
    """
    def wrapper(*args, **kwargs):
        Thread(target=func, args=args, kwargs=kwargs).start()
    return wrapper


class Models(metaclass=Singleton):
    """
    Models

    모델을 로딩하며, 미리 불러와 저장하는 클래스
    """

    # 클래스 초기화
    def __init__(self, equip_model_path='models/equip/best.pt', handmotion_model_path='models/handmotion/best.pt', confidence=0.5):
        # 연산 device 설정
        gpu = 0
        torch_device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)

        self.loaded = False
        self.device = torch.device(torch_device)

        self.equip_model_path = equip_model_path
        self.handmotion_model_path = handmotion_model_path
        self.confidence = confidence

        self.is_loading = False

    # 모델 lazy load
    def load(self):
        # 로딩중이거나 이미 로딩 된 경우 그냥 return
        if self.is_loading or self.loaded:
            return
        self.is_loading = True

        # torch hub를 사용한 model 로드
        self.equip_model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=self.equip_model_path)
        self.handmotion_model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=self.handmotion_model_path)

        # 해당 모델을 사용할 device로 옮김
        self.equip_model.to(self.device)
        self.handmotion_model.to(self.device)

        # 모델을 eval 모드로 설정
        self.equip_model.eval()
        self.handmotion_model.eval()

        # confidence 설정
        self.equip_model.conf = self.confidence
        # self.handmotion_model.conf = self.confidence

        # 로딩 완료 flag 설정
        self.loaded = True
        self.is_loading = False

    # 이미지에서 Object Detection을 수행

    def detect(self, img, mhi_img, size=640):
        # 모델이 로딩되지 않았다면 로딩
        if not self.loaded:
            self.load()

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

        # thread
        self.detect_thread = None

        # MHI용 설정
        self.pos = pos
        self.cmap = cmapy.cmap(cmap)

        # 배경 제거 모델
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # 모델
        self.models = Models()

        # ws
        self.ws_client_list = []

        # status
        self.run_trigger = False
        self.started = False
        self.worked = False

        self.box_color = {
            'OK': (0, 255, 0),
            'NO': (0, 0, 255),
            'HAND': (255, 0, 0)
        }

    def status(self):
        return self.started and self.worked

    # 실행
    # Thread를 사용하여, 실시간으로 frame을 받아와 저장
    def run(self):
        if self.run_trigger:
            return
        self.run_trigger = True

        # 모델이 로딩되지 않았다면 로딩
        if not self.models.loaded:
            self.models.load()

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
                self.fps_delay = 1/25

        # Thread 시작
        if self.detect_thread is None:
            self.detect_thread = Thread(target=self.update, args=())
            self.detect_thread.daemon = True
            self.detect_thread.start()

        self.started = True
        self.run_trigger = False

    # Thread 및 Device 정지
    def stop(self):
        self.started = False

        if self.device is not None:
            self.device.release()

        self.worked = False

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

                @fire_and_forget
                def queue_put():
                    output_queue.put_nowait(
                        (self.device_id, self.prepare_data()))

                queue_put()
            except Exception as e:
                continue

    def prepare_data(self):
        try:
            boxed = self.boxed_frame.copy()
            det = self.detected.copy()

            j = {
                'status': 'ok',
                'detected': det,
                'boxed_frame': self.img_array_to_jpg_base64(boxed),
            }
        except Exception as e:
            j = {
                'status': 'error',
                'error': str(e),
            }
        return json.dumps(j)

    # convert numpy cv2 array to jpg base64 string
    def img_array_to_jpg_base64(self, img):
        jpg = cv2.imencode('.jpg', img)[1]
        return base64.b64encode(jpg).decode('utf-8')

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
            color = self.get_box_color(d['model'], d['cls_name'])

            name = f'{d["model"]}_{d["cls_name"]}, {d["confidence"]:.2f}'

            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)
            cv2.putText(f, name, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return f

    # box color 설정
    def get_box_color(self, model, name):
        if model in ['equip']:
            if 'OK' in name:
                return self.box_color['OK']
            return self.box_color['NO']
        if model == 'hand':
            return self.box_color['HAND']
        return (255, 255, 255)


def clear_output_queue():
    with output_queue.mutex:
        output_queue.queue.clear()


# # 중복 방지 및 context 유지를 위하여 flask에서 사용하는 global 변수를 사용
def get_streamer(device_id, pos=15, cmap='nipy_spectral', confidence=0.5) -> Streamer:
    n = f'streamer_{device_id}'
    s = getattr(g, n, None)
    if s is None:
        s = Streamer(device_id, pos, cmap, confidence)
        setattr(g, n, s)
    return s
# def get_streamer(device_id, pos=15, cmap='nipy_spectral', confidence=0.5) -> Streamer:
#     return Streamer(device_id, pos, cmap, confidence)


# flask app
app = Flask(__name__, template_folder='')
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@sock.route('/feed/<int:device_id>')
def json_feed(ws: WSServer, device_id):
    """
        WebSocket을 사용하여 실시간으로 좌표 로그 및 영상 전송
    """
    global client_list

    # streamer 불러옴
    streamer = get_streamer(device_id)

    if streamer.started is False and not streamer.run_trigger and len(client_list.get(device_id, [])) == 0:
        streamer.run()

    if client_list.get(device_id, None) is None:
        client_list[device_id] = []

    client_list[device_id].append(ws)
    try:
        while True:
            msg = ws.receive()
            # stop 메세지를 받으면 종료
            if msg == 'stop':
                client_list[device_id].remove(ws)
                break
    except Exception as e:
        # 오류 데이터 전송
        ws.send(json.dumps({'status': 'error', 'error': str(e)}))

    # # 모든 client가 종료되면 streamer 종료
    # if len(client_list[device_id]) == 0:
    #     streamer.stop()


# 디버깅용 index template
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# socket server thread
def send_socket_message():
    # 만약을 위해 global 변수 사용
    global client_list, output_queue

    while True:
        # GIL hang 방지
        time.sleep(0.01)

        try:
            # ouput_queue에서 데이터를 가져옴
            queue = output_queue.get_nowait()

            # output_queue 비우기
            clear_output_queue()

            # 데이터 분리
            key, item = queue

            # 클라이언트 리스트 로드
            clients = client_list.get(key, []).copy()
        except Exception as e:
            # 오류 발생시 재시도
            continue

        @fire_and_forget
        def func(client):
            try:
                client.send(item)
            except:
                # 오류 발생시 클라이언트 close 시도
                tryExceptFunction(client.close)

                # 오류 발생시 클라이언트 리스트에서 제거 시도
                tryExceptFunction(client_list[key].remove, client)

        for client in clients:
            try:
                func(client)
            except Exception as e:
                # 오류 발생시 클라이언트 리스트에서 제거 시도
                tryExceptFunction(client_list[key].remove, client)


# Flask 실행
if __name__ == '__main__':
    ws = Thread(target=send_socket_message)
    ws.daemon = True
    ws.start()
    app.run(host='0.0.0.0', threaded=True)  # debug=True
