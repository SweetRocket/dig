function readFile(file) {
  return new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      reject(err);
    }
  });
}

function readFileInput(fileInput) {
    return readFile(fileInput.files[0]);
}

function readMultiFileInput(fileInput) {
    return Promise.all(Array.from(fileInput.files).map(readFile));
}

function buildQueryParams(url, params) {
    params = _.pickBy(params, (value) => {
        return value !== undefined && value !== null;
    });
    const query = Object.keys(params)
        .map((key) => `${key}=${params[key]}`)
        .join('&');
    return `${url}?${query}`;
}