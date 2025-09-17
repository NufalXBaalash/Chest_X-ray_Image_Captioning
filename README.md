# Image Caption Generator

A Streamlit web application that generates natural language captions for images using a deep learning model.

## Features

- 🖼️ **Image Upload**: Upload images in various formats (PNG, JPG, JPEG, GIF, BMP)
- 🎯 **Caption Generation**: Generate descriptive captions using a pre-trained model
- ⚙️ **Configurable Parameters**: Adjust generation parameters for different results
- 🎨 **Modern UI**: Clean and intuitive user interface
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Image_caption
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the following files in your project directory:
   - `model.keras` - The trained captioning model
   - `wordtoix.pkl` - Word to index mapping
   - `ixtoword.pkl` - Index to word mapping
   - `encodings.pkl` - Pre-computed image encodings (optional but recommended)

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

4. **Download encodings.pkl (Recommended):**
   ```bash
   pip install gdown
   gdown 1aPRfA7008147pp0Ni7SUKO-0JZHjRcCe
   ```
   Or download manually from: [Google Drive Link](https://drive.google.com/file/d/1aPRfA7008147pp0Ni7SUKO-0JZHjRcCe/view?usp=drive_link)

5. Upload an image and adjust the generation parameters as needed

6. Click "Generate Caption" to create a description for your image

## Parameters

- **Maximum Words**: Controls the maximum length of generated captions (5-50 words)
- **Temperature**: Controls randomness in generation (0.1-2.0, lower = more focused, higher = more creative)
- **Top-K Sampling**: Limits word selection to top K most probable words (1-20)

## File Structure

```
Image_caption/
├── app.py              # Main Streamlit application
├── model.keras         # Trained captioning model
├── wordtoix.pkl        # Word to index mapping
├── ixtoword.pkl        # Index to word mapping
├── encodings.pkl       # Pre-computed image encodings (download from Google Drive)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- Streamlit 1.28+
- Other dependencies listed in `requirements.txt`

## Features

### With encodings.pkl:
- **Pre-encoded Images**: Browse and select from pre-computed image encodings
- **Fast Generation**: Skip image preprocessing for faster caption generation
- **Dataset Images**: Access to the full dataset of pre-encoded images

### Without encodings.pkl:
- **Image Upload**: Upload and process new images
- **Real-time Processing**: Images are processed on-the-fly
- **Flexible Input**: Support for various image formats

## Notes

- The application expects images to be preprocessed to 224x224 pixels
- Model loading is cached for better performance
- Error handling is included for robust operation
- The interface is optimized for both desktop and mobile use
- **encodings.pkl** provides access to pre-computed image features for faster processing