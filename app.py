import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from PIL import Image
import io
import os
from keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

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
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .caption-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: none;
        font-size: 1.2rem;
        line-height: 1.6;
        text-align: center;
        font-weight: 500;
    }
    
    .parameter-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .image-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
        text-align: center;
    }
    
    .result-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        color: #2d5a3d;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stFileUploader > div {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stRadio > div {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_feature_extractor():
    """Load DenseNet121 for feature extraction"""
    try:
        # Load DenseNet121 base model (same as used for encoding)
        base_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
        return base_model
    except Exception as e:
        st.error(f"Error loading feature extractor: {str(e)}")
        return None

@st.cache_resource
def load_pneumonia_model():
    """Load MobileNetV2 pneumonia detection model"""
    try:
        pneumonia_model = load_model('MobileNetV2_model.keras', compile=False)
        return pneumonia_model
    except Exception as e:
        st.warning(f"Pneumonia detection model not found: {str(e)}")
        return None

@st.cache_resource
def load_model_and_data():
    """Load the trained model and tokenizer data with caching"""
    try:
        # Load trained model - try both possible names
        try:
            model = load_model('model.keras', compile=False)
        except:
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
    """Preprocess uploaded image using DenseNet121 feature extraction (same as used for encodings)"""
    try:
        # Load the feature extractor
        feature_extractor = load_feature_extractor()
        if feature_extractor is None:
            return None, None
        
        # Read image using PIL
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to (224, 224) - same as used for encoding
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Preprocess using DenseNet preprocessing (same as used for encoding)
        image_array = preprocess_input(image_array)
        
        # Extract features using DenseNet121 (same as used for encoding)
        features = feature_extractor.predict(image_array, verbose=0)
        
        return features, image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def preprocess_image_for_pneumonia(uploaded_file):
    """Preprocess uploaded image for pneumonia detection"""
    try:
        # Read image using PIL
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to (224, 224) for MobileNetV2
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize to [0, 1] range
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array, image
    except Exception as e:
        st.error(f"Error preprocessing image for pneumonia detection: {str(e)}")
        return None, None

def detect_pneumonia(pneumonia_model, image_array):
    """Detect pneumonia in the image"""
    try:
        if pneumonia_model is None:
            return None, None
        
        # Make prediction
        prediction = pneumonia_model.predict(image_array, verbose=0)
        
        # Get probability of pneumonia (assuming binary classification: 0=Normal, 1=Pneumonia)
        pneumonia_prob = prediction[0][0] if len(prediction[0]) == 1 else prediction[0][1]
        normal_prob = 1 - pneumonia_prob
        
        # Determine diagnosis
        if pneumonia_prob > 0.5:
            diagnosis = "Pneumonia Detected"
            confidence = pneumonia_prob
        else:
            diagnosis = "Normal (No Pneumonia)"
            confidence = normal_prob
        
        # Ensure confidence is between 0 and 1 and convert to Python float
        confidence = float(min(max(confidence, 0.0), 1.0))
        
        return diagnosis, confidence
    except Exception as e:
        st.error(f"Error in pneumonia detection: {str(e)}")
        return None, None

def generate_caption(model, image_features, words_to_index, index_to_words, 
                    max_steps=25, temperature=0.7, top_k=5, max_length=124):
    """
    Generate a caption for a given image feature vector.
    This matches the original Image_Caption function from your code.
    
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
        # Check if required mappings exist
        if not words_to_index or not index_to_words:
            return "Error: Missing word mappings"
        
        # For medical models, we might not have startseq, so we'll start with empty sequence
        # and let the model generate the first word
        in_text = 'startseq'  # Try the original approach first
        generated_words = []

        # If startseq is not in vocabulary, start with empty sequence
        if 'startseq' not in words_to_index:
            in_text = ''
            generated_words = []

        for _ in range(max_steps):
            try:
                # Convert current text to sequence
                if in_text:
                    sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
                else:
                    sequence = []
                
                # Pad sequence
                sequence = pad_sequences([sequence], maxlen=max_length, padding='pre')

                # Predict next word probabilities
                yhat = model([image_features, sequence], training=False)
                probabilities = yhat.numpy().ravel()

                # Check for valid probabilities
                if len(probabilities) == 0 or np.all(probabilities == 0):
                    break

                # Temperature scaling
                probabilities = np.exp(np.log(probabilities + 1e-9)/temperature)
                probabilities /= np.sum(probabilities)

                # Top-k sampling
                top_indices = np.argsort(probabilities)[-top_k:]
                top_probs = probabilities[top_indices] / np.sum(probabilities[top_indices])
                yhat_index = np.random.choice(top_indices, p=top_probs)

                # Check if index is valid
                if yhat_index not in index_to_words:
                    break
                    
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
                if in_text:
                    in_text += ' ' + word
                else:
                    in_text = word
                
            except Exception as step_error:
                print(f"Error in step: {step_error}")
                break

        if not generated_words:
            return "No caption generated - model may need pre-computed image features"
            
        return ' '.join(generated_words)
        
    except Exception as e:
        print(f"Error in generate_caption: {str(e)}")
        return f"Error generating caption: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Caption Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models and data silently
    model, words_to_index, index_to_words, encodings = load_model_and_data()
    pneumonia_model = load_pneumonia_model()
    
    if model is None:
        st.error("Failed to load models. Please check that all required files are present.")
        st.stop()
    
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
    
    # Simple image upload section
    st.markdown('<h2 class="sub-header">📤 Upload Your Image</h2>', unsafe_allow_html=True)
    
    # Initialize session state for results
    if 'generated_caption' not in st.session_state:
        st.session_state.generated_caption = None
    if 'display_image' not in st.session_state:
        st.session_state.display_image = None
    if 'image_name' not in st.session_state:
        st.session_state.image_name = None
    if 'pneumonia_diagnosis' not in st.session_state:
        st.session_state.pneumonia_diagnosis = None
    if 'pneumonia_confidence' not in st.session_state:
        st.session_state.pneumonia_confidence = None
    
    # Simple file uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload a chest X-ray image for analysis"
    )
    
    if uploaded_file is not None:
        st.session_state.display_image = Image.open(uploaded_file)
        st.session_state.image_name = uploaded_file.name
        
        # Show uploaded image
        st.image(st.session_state.display_image, caption=f"📷 {st.session_state.image_name}", width='stretch')
        
        # Analysis button
        if st.button("🎯 Analyze Image", type="primary", width='stretch'):
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Preprocess image for caption generation
                    image_features, _ = preprocess_image(uploaded_file)
                    
                    if image_features is not None:
                        # Generate caption
                        caption = generate_caption(
                            model, image_features, words_to_index, index_to_words,
                            max_steps=max_steps, temperature=temperature, top_k=top_k
                        )
                        st.session_state.generated_caption = caption
                        
                        # Detect pneumonia if model is available
                        if pneumonia_model is not None:
                            pneumonia_array, _ = preprocess_image_for_pneumonia(uploaded_file)
                            if pneumonia_array is not None:
                                diagnosis, confidence = detect_pneumonia(pneumonia_model, pneumonia_array)
                                st.session_state.pneumonia_diagnosis = diagnosis
                                st.session_state.pneumonia_confidence = confidence
                        
                        st.rerun()
                    else:
                        st.error("Failed to process the image. Please try again.")
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
    
    # Display results section
    if st.session_state.generated_caption is not None:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎯 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Display results in a clean layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display caption
            st.markdown("### 📝 Medical Description")
            st.markdown(f'<div class="caption-box"><strong>"{st.session_state.generated_caption}"</strong></div>', unsafe_allow_html=True)
        
        with col2:
            # Display pneumonia detection if available
            if st.session_state.pneumonia_diagnosis is not None:
                st.markdown("### 🏥 Pneumonia Detection")
                if "Pneumonia" in st.session_state.pneumonia_diagnosis:
                    st.error(f"🚨 **{st.session_state.pneumonia_diagnosis}**")
                else:
                    st.success(f"✅ **{st.session_state.pneumonia_diagnosis}**")
                
                # Show confidence
                confidence_percent = float(st.session_state.pneumonia_confidence) * 100
                progress_value = float(min(max(st.session_state.pneumonia_confidence, 0.0), 1.0))
                st.progress(progress_value)
                st.metric("Confidence", f"{confidence_percent:.1f}%")
        
        # Add a new analysis button
        if st.button("🔄 Analyze New Image", width='stretch'):
            st.session_state.generated_caption = None
            st.session_state.pneumonia_diagnosis = None
            st.session_state.pneumonia_confidence = None
            st.rerun()
    
    # Simple information section
    st.markdown("---")
    st.markdown("""
    ### 🏥 Medical Image Analysis
    This app analyzes chest X-ray images to:
    - **Generate medical descriptions** of the image content
    - **Detect pneumonia** with confidence scoring
    
    Upload a chest X-ray image above to get started.
    """)

if __name__ == "__main__":
    main()
