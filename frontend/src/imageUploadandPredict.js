import React, { useState } from 'react';
import './App.css'; // Importing the external CSS file
const ImageUploadandPredict = () => {
  const [imagePreview, setImagePreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState('');

  const readURL = (input) => {
    if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          setImagePreview(reader.result);
        }
      };
      reader.readAsDataURL(input.files[0]);
    }
  };

  const handleImageChange = (e) => {
    e.preventDefault();
    const input = e.target;
    console.log('Input files:', input.files);
    readURL(input);
  };

  const handlePredict = async () => {
    setLoading(true);
    setResult('');

    const formData = new FormData();
    const fileInput = document.getElementById('imageUpload');
    if (fileInput && fileInput.files) {
      formData.append('image', fileInput.files[0]);
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.text();
        setResult(data);
        console.log('Success!');
      } else {
        console.error('Failed to make prediction.');
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <><div>

      <h1>Image Recgnition using Transfer Learnig</h1>

    </div><div>
        <input id="imageUpload" type="file" accept=".jpg,.jpeg,.png" onChange={handleImageChange} />
        <button id="btn-predict" onClick={handlePredict}>
          Predict
        </button>
        <div className="image-section" style={{ display: imagePreview ? 'block' : 'none' }}>
          {imagePreview && (
            <div id="imagePreview" style={{ backgroundImage: `url(${imagePreview})`, display: 'block', width: '200px', height: '200px', marginBottom: '10px',alignSelf:"auto" }} />
          )}
        </div>
        <div className="loader" style={{ display: loading ? 'block' : 'none' }}>
          Loading...
        </div>
        <div id="result" style={{ display: result ? 'block' : 'none' }}>
          Result: {result}
        </div>
      </div></>
  );
};

export default ImageUploadandPredict;
