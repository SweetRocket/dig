// Input 필드 중 file 에서 파일을 읽는 함수
function readFile(file) {
  return new Promise((resolve, reject) => {
    try {
      // FileReader 객체 생성
      const reader = new FileReader();

      // 파일 읽기가 완료되면 resolve
      reader.onload = (e) => {
        resolve(e.target.result);
      };

      // 파일을 data url 형식으로 읽기
      reader.readAsDataURL(file);
    } catch (err) {
      // 에러 발생시 reject
      reject(err);
    }
  });
}

// Input 필드에서 파일을 읽는 함수 (단일 파일)
function readFileInput(fileInput) {
    return readFile(fileInput.files[0]);
}

// Input 필드에서 파일을 읽는 함수 (다중 파일)
function readMultiFileInput(fileInput) {
    return Promise.all(Array.from(fileInput.files).map(readFile));
}

// 객체를 url 쿼리 파라미터로 변환
function buildQueryParams(url, params) {
    // undefined, null 값 제거
    params = _.pickBy(params, (value) => {
        return value !== undefined && value !== null;
    });
    
    // 쿼리 파라미터로 변환
    const query = Object.keys(params)
        .map((key) => `${key}=${params[key]}`)
        .join('&');
    
    // url에 쿼리 파라미터 추가
    return `${url}?${query}`;
}