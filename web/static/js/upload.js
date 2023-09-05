const uploadForm = document.querySelector('#uploadForm');

uploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const image = document.querySelector('#image').files[0];
    const formData = new FormData();
    formData.append('image', image);
    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).then(data => {
        console.log(data);
        addImageToList(data.filename);
    }).catch(error => {
        console.error('There was an error:', error);
    });
});

function addImageToList(filename) {
    const imageList = document.querySelector('#imageList');
    const img = document.createElement('img');
    img.src = '/static/images/' + filename;
    img.alt = filename;
    img.style.width = '200px';
    img.style.height = '200px';
    imageList.appendChild(img);
}
