import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from PIL import Image
import io
import os
import cv2

# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .parameter-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load the trained model and tokenizer data with caching"""
    try:
        # Load trained model
        model = load_model('model.keras', compile=False)
        
        # Load tokenizer mappings
        with open("wordtoix.pkl", "rb") as f:
            words_to_index = pickle.load(f)
        
        with open("ixtoword.pkl", "rb") as f:
            index_to_words = pickle.load(f)
        
        # Load image encodings if available
        encodings = None
        if os.path.exists("encodings.pkl"):
            with open("encodings.pkl", "rb") as f:
                encodings = pickle.load(f)
        
        return model, words_to_index, index_to_words, encodings
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None, None, None

def preprocess_image(uploaded_file):
    """Preprocess uploaded image to match model input requirements"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image (adjust size as needed for your model)
        image = image.resize((224, 224))  # Common size for image captioning models
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Reshape for model input (1, height, width, channels)
        image_array = image_array.reshape(1, 224, 224, 3)
        
        return image_array, image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def generate_caption(model, image_features, words_to_index, index_to_words, 
                    max_steps=25, temperature=0.7, top_k=5, max_length=124):
    """
    Generate a caption for a given image feature vector.
    
    Args:
        model: Trained captioning model
        image_features: Preprocessed image features
        words_to_index: Word to index mapping
        index_to_words: Index to word mapping
        max_steps: Maximum number of words to generate
        temperature: Temperature for sampling (controls randomness)
        top_k: Number of top probable words to sample from
        max_length: Maximum sequence length
    
    Returns:
        str: Generated caption
    """
    try:
        in_text = 'startseq'
        generated_words = []

        for _ in range(max_steps):
            # Convert current text to sequence
            sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='pre')

            # Predict next word probabilities
            yhat = model([image_features, sequence], training=False)
            probabilities = yhat.numpy().ravel()

            # Temperature scaling
            probabilities = np.exp(np.log(probabilities + 1e-9)/temperature)
            probabilities /= np.sum(probabilities)

            # Top-k sampling
            top_indices = np.argsort(probabilities)[-top_k:]
            top_probs = probabilities[top_indices] / np.sum(probabilities[top_indices])
            yhat_index = np.random.choice(top_indices, p=top_probs)

            word = index_to_words[yhat_index]

            # Stop conditions
            if word == 'endseq':
                break
            if word == 'xxxx':  # skip placeholder tokens
                continue
            # Avoid repeating the same word 3 times consecutively
            if len(generated_words) >= 2 and word == generated_words[-1] == generated_words[-2]:
                break

            generated_words.append(word)
            in_text += ' ' + word

        return ' '.join(generated_words)
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Error generating caption"

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Caption Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and data
    with st.spinner("Loading model and tokenizer..."):
        model, words_to_index, index_to_words, encodings = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model. Please check that all required files are present.")
        st.stop()
    
    # Check for encodings.pkl file
    if encodings is None:
        st.warning("⚠️ **encodings.pkl file not found!**")
        st.markdown("""
        To use the full functionality of this app, you need to download the `encodings.pkl` file.
        
        **Download Instructions:**
        1. Click on this link: [Download encodings.pkl](https://drive.google.com/file/d/1aPRfA7008147pp0Ni7SUKO-0JZHjRcCe/view?usp=drive_link)
        2. Download the file and place it in your project directory
        3. Refresh this page
        
        **Alternative method using command line:**
        ```bash
        pip install gdown
        gdown 1aPRfA7008147pp0Ni7SUKO-0JZHjRcCe
        ```
        """)
        st.markdown("---")
    else:
        st.success(f"✅ **encodings.pkl loaded successfully!** ({len(encodings)} image encodings available)")
    
    # Sidebar for parameters
    st.sidebar.markdown("## 🎛️ Generation Parameters")
    
    max_steps = st.sidebar.slider(
        "Maximum Words", 
        min_value=5, 
        max_value=50, 
        value=25, 
        help="Maximum number of words to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.7, 
        step=0.1,
        help="Controls randomness (lower = more focused, higher = more creative)"
    )
    
    top_k = st.sidebar.slider(
        "Top-K Sampling", 
        min_value=1, 
        max_value=20, 
        value=5, 
        help="Number of top probable words to sample from"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📤 Upload Image")
        
        # If encodings are available, provide option to use pre-encoded images
        if encodings is not None:
            st.markdown("### 🎯 Choose Image Source")
            image_source = st.radio(
                "Select image source:",
                ["Upload new image", "Use pre-encoded image"],
                help="Choose between uploading a new image or using a pre-encoded image from the dataset"
            )
            
            if image_source == "Use pre-encoded image":
                # Display list of available images
                image_names = list(encodings.keys())
                selected_image = st.selectbox(
                    "Select an image:",
                    image_names,
                    help="Choose from pre-encoded images"
                )
                
                if selected_image:
                    st.markdown("### 📷 Selected Image")
                    # Note: This assumes the images are in a specific directory
                    # You may need to adjust the path based on your setup
                    st.info(f"Selected: {selected_image}")
                    st.markdown("**Note:** To display the actual image, you'll need to have the image files in your project directory.")
                    
                    # Generate caption for pre-encoded image
                    if st.button("🎯 Generate Caption for Selected Image", type="primary"):
                        with st.spinner("Generating caption..."):
                            image_vector = encodings[selected_image].reshape((1, -1))
                            caption = generate_caption(
                                model, image_vector, words_to_index, index_to_words,
                                max_steps=max_steps, temperature=temperature, top_k=top_k
                            )
                            
                            # Display results
                            st.markdown("### 🎯 Generated Caption")
                            st.markdown(f'<div class="caption-box"><strong>{caption}</strong></div>', 
                                      unsafe_allow_html=True)
                            
                            # Show parameters used
                            st.markdown("### ⚙️ Parameters Used")
                            st.markdown(f"""
                            <div class="parameter-box">
                            <strong>Max Steps:</strong> {max_steps}<br>
                            <strong>Temperature:</strong> {temperature}<br>
                            <strong>Top-K:</strong> {top_k}
                            </div>
                            """, unsafe_allow_html=True)
        else:
            image_source = "Upload new image"
        
        # File uploader (only show if uploading new image)
        if image_source == "Upload new image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image to generate a caption"
            )
        else:
            uploaded_file = None
        
        if uploaded_file is not None:
            # Display uploaded image
            st.markdown("### 📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    # Preprocess image
                    image_features, display_image = preprocess_image(uploaded_file)
                    
                    if image_features is not None:
                        # Generate caption
                        caption = generate_caption(
                            model, image_features, words_to_index, index_to_words,
                            max_steps=max_steps, temperature=temperature, top_k=top_k
                        )
                        
                        # Display results
                        st.markdown("### 🎯 Generated Caption")
                        st.markdown(f'<div class="caption-box"><strong>{caption}</strong></div>', 
                                  unsafe_allow_html=True)
                        
                        # Show parameters used
                        st.markdown("### ⚙️ Parameters Used")
                        st.markdown(f"""
                        <div class="parameter-box">
                        <strong>Max Steps:</strong> {max_steps}<br>
                        <strong>Temperature:</strong> {temperature}<br>
                        <strong>Top-K:</strong> {top_k}
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## ℹ️ About This App")
        
        st.markdown("""
        This application uses a deep learning model to generate captions for images.
        
        ### How it works:
        1. **Upload an image** using the file uploader
        2. **Adjust parameters** in the sidebar to control caption generation
        3. **Click "Generate Caption"** to create a description
        
        ### Parameters:
        - **Maximum Words**: Controls the maximum length of generated captions
        - **Temperature**: Controls randomness (lower = more focused, higher = more creative)
        - **Top-K Sampling**: Limits word selection to top K most probable words
        
        ### Tips:
        - Try different temperature values for varied results
        - Lower temperature gives more consistent, focused captions
        - Higher temperature produces more creative, diverse captions
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Model Information")
        st.info("""
        **Model**: Pre-trained image captioning model
        **Input**: RGB images (224x224 pixels)
        **Output**: Natural language captions
        **Architecture**: CNN + LSTM/Transformer based
        """)

if __name__ == "__main__":
    main()
