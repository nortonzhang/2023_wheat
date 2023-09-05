// const form = document.querySelector('form');
// form.addEventListener('submit', event => {
//     event.preventDefault();
//     const input = document.querySelector('input[type="file"]');
//     const formData = new FormData();
//     formData.append('image', input.files[0]);
//
//     fetch('/predict', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         const result = document.getElementById('result');
//         result.innerHTML = '';
//         const h2 = document.createElement('h2');
//         h2.textContent = '诊断结果：';
//         result.appendChild(h2);
//         const img = document.createElement('img');
//         img.src = URL.createObjectURL(input.files[0]);
//         img.style.width = '300px';
//         result.appendChild(img);
//         const p = document.createElement('p');
//         p.textContent = `病害类型：${data.result}`;
//         result.appendChild(p);
//     })
//     .catch(error => console.error(error));
// });


const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const message = document.getElementById('message');

form.addEventListener('submit', (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
        message.innerHTML = '请选择文件';
        return;
    }
    if (!file.type.startsWith('image/')) {
        message.innerHTML = '请上传图片文件';
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload');
    xhr.send(formData);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                message.innerHTML = '上传成功';
            } else {
                message.innerHTML = '上传失败';
            }
        }
    };
});
