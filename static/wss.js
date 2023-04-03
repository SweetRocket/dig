// 임시로 사용할 빈 함수
function _emptyFunc() {
  return false;
}

// 웹 소켓 스트리밍 함수
class WebSocketStreamer {

  // 생성자
  constructor(url, img, log) {
    this.url = url;

    this.img_write = img;
    this.log_write = log;

    this.websocket = null;
  }

  // 연결 끊기
  disconnect() {
    if (this.websocket != null) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  // 서버 연결
  connect() {
    if (this.websocket == null) {
      // 서버 연결
      this.websocket = new WebSocket(this.url);
      let self = this;

      // 서버 연결 상태
      this.websocket.onopen = function (event) {
        this.onOpen(this, event);
      };

      // 서버 연결 종료
      this.websocket.onclose = function (event) {
        this.onClose(this, event);
      };

      // 서버 연결 에러
      this.websocket.onerror = function (event) {
        this.onError(this, event);
      };

      // 서버로부터 메시지 수신
      this.websocket.onmessage = function (event) {
        // 서버로부터 받은 메시지를 JSON으로 파싱
        const msg = JSON.parse(event.data);

        // detected가 없을 경우 빈 배열로 설정
        const detected = msg?.detected ?? [];

        // 서버 상태가 error일 경우 에러 메시지 출력
        if (msg?.status == "error") {
          self.errorFunc(msg);
          console.error(msg);
          return;
        }

        // 함수 실행
        self.img_write("data:image/png;base64," + msg?.boxed_frame ?? "");
        self.log_write(detected, msg);
      };
    }
  }
}
