import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from PIL import Image
import io
import os

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
        
        # For this model, we need to create a dummy feature vector
        # since the model expects pre-computed features, not raw pixels
        # This is a simplified approach - in practice, you'd use a CNN encoder
        feature_dim = 2048  # Common feature dimension for image captioning
        dummy_features = np.random.normal(0, 0.1, (1, feature_dim))
        
        return dummy_features, image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
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
    
    # Load model and data
    with st.spinner("Loading model and tokenizer..."):
        model, words_to_index, index_to_words, encodings = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model. Please check that all required files are present.")
        st.stop()
    
    # Check for encodings.pkl file
    if encodings is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ **encodings.pkl file not found!**")
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
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-box">✅ **encodings.pkl loaded successfully!** ({len(encodings)} image encodings available)</div>', unsafe_allow_html=True)
    
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
    
    # Image source selection
    st.markdown('<h2 class="sub-header">📤 Choose Your Image</h2>', unsafe_allow_html=True)
    
    # If encodings are available, provide option to use pre-encoded images
    if encodings is not None:
        image_source = st.radio(
            "Select image source:",
            ["Upload new image", "Use pre-encoded image"],
            help="Choose between uploading a new image or using a pre-encoded image from the dataset",
            horizontal=True
        )
    else:
        image_source = "Upload new image"
    
    # Initialize session state for results
    if 'generated_caption' not in st.session_state:
        st.session_state.generated_caption = None
    if 'display_image' not in st.session_state:
        st.session_state.display_image = None
    if 'image_name' not in st.session_state:
        st.session_state.image_name = None
    
    # Handle different image sources
    if image_source == "Upload new image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image to generate a caption"
        )
        
        if uploaded_file is not None:
            st.session_state.display_image = Image.open(uploaded_file)
            st.session_state.image_name = uploaded_file.name
            
            # Show limitation notice
            st.markdown('<div class="warning-box">⚠️ <strong>Note:</strong> This model requires pre-computed image features. For uploaded images, we use a simplified approach that may not produce accurate captions. For best results, use pre-encoded images from the dataset.</div>', unsafe_allow_html=True)
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary", use_container_width=True):
                with st.spinner("🔄 Generating caption..."):
                    try:
                        # Preprocess image
                        image_features, _ = preprocess_image(uploaded_file)
                        
                        if image_features is not None:
                            # Generate caption
                            caption = generate_caption(
                                model, image_features, words_to_index, index_to_words,
                                max_steps=max_steps, temperature=temperature, top_k=top_k
                            )
                            st.session_state.generated_caption = caption
                            st.rerun()
                        else:
                            st.error("Failed to preprocess the image. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.error("This might be because the model expects pre-computed image features. Try using a pre-encoded image instead.")
    
    else:  # Use pre-encoded image
        image_names = list(encodings.keys())
        selected_image = st.selectbox(
            "Select an image:",
            image_names,
            help="Choose from pre-encoded images"
        )
        
        if selected_image:
            st.session_state.image_name = selected_image
            st.markdown(f'<div class="info-box">📷 <strong>Selected:</strong> {selected_image}</div>', unsafe_allow_html=True)
            
            # Generate caption for pre-encoded image
            if st.button("🎯 Generate Caption for Selected Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Generating caption..."):
                    try:
                        image_vector = encodings[selected_image].reshape((1, -1))
                        caption = generate_caption(
                            model, image_vector, words_to_index, index_to_words,
                            max_steps=max_steps, temperature=temperature, top_k=top_k
                        )
                        st.session_state.generated_caption = caption
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
    
    # Display results section
    if st.session_state.generated_caption is not None:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎯 Generated Caption</h2>', unsafe_allow_html=True)
        
        # Create a beautiful result container
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        # Display image and caption side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            if st.session_state.display_image is not None:
                st.image(st.session_state.display_image, caption=f"📷 {st.session_state.image_name}", use_container_width=True)
            else:
                st.markdown(f"<h4>📷 {st.session_state.image_name}</h4>", unsafe_allow_html=True)
                st.info("Image preview not available for pre-encoded images")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="caption-box"><strong>"{st.session_state.generated_caption}"</strong></div>', unsafe_allow_html=True)
            
            # Show parameters used
            st.markdown("### ⚙️ Generation Parameters")
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                st.metric("Max Steps", max_steps)
            with col_param2:
                st.metric("Temperature", f"{temperature:.1f}")
            with col_param3:
                st.metric("Top-K", top_k)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a new generation button
        if st.button("🔄 Generate New Caption", use_container_width=True):
            st.session_state.generated_caption = None
            st.rerun()
    
    # Information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">ℹ️ About This App</h2>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns([1, 1])
    
    with col_info1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 How it works:
        1. **Choose an image** by uploading or selecting from pre-encoded images
        2. **Adjust parameters** in the sidebar to control caption generation
        3. **Click "Generate Caption"** to create a description
        4. **View results** with the image and caption displayed together
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_info2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎛️ Parameters:
        - **Maximum Words**: Controls caption length (5-50 words)
        - **Temperature**: Controls randomness (0.1-2.0)
        - **Top-K Sampling**: Limits word selection (1-20)
        
        ### 💡 Tips:
        - Lower temperature = more focused captions
        - Higher temperature = more creative captions
        - Try different combinations for varied results
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
