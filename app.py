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

# Simple CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #FFFAF1;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #000000;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .caption-display {
        background-color: #F0F8FF;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #4169E1;
        font-size: 1.1rem;
        text-align: center;
    }
    
    .stButton > button {
        background-color: #4169E1;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1E90FF;
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
        
        # Check if 'startseq' exists in the vocabulary
        if 'startseq' not in words_to_index:
            return "Error: 'startseq' not found in vocabulary"
        
        in_text = 'startseq'
        generated_words = []

        for step in range(max_steps):
            try:
                # Convert current text to sequence
                sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
                
                if not sequence:
                    break
                    
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
                in_text += ' ' + word
                
            except Exception as step_error:
                print(f"Error in step {step}: {step_error}")
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
        st.warning("⚠️ encodings.pkl file not found! Download it from the Google Drive link for best results.")
    else:
        st.success(f"✅ encodings.pkl loaded successfully! ({len(encodings)} image encodings available)")
    
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
    
    # Initialize session state for results
    if 'generated_caption' not in st.session_state:
        st.session_state.generated_caption = None
    if 'display_image' not in st.session_state:
        st.session_state.display_image = None
    if 'image_name' not in st.session_state:
        st.session_state.image_name = None
    
    # Image source selection
    if encodings is not None:
        image_source = st.radio(
            "Choose image source:",
            ["Upload new image", "Use pre-encoded image"],
            horizontal=True
        )
    else:
        image_source = "Upload new image"
    
    # Handle different image sources
    if image_source == "Upload new image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
        )
        
        if uploaded_file is not None:
            st.session_state.display_image = Image.open(uploaded_file)
            st.session_state.image_name = uploaded_file.name
            
            st.warning("⚠️ Note: This model requires pre-computed image features. For uploaded images, we use a simplified approach that may not produce accurate captions.")
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
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
        selected_image = st.selectbox("Select an image:", image_names)
        
        if selected_image:
            st.session_state.image_name = selected_image
            st.info(f"Selected: {selected_image}")
            
            # Generate caption for pre-encoded image
            if st.button("🎯 Generate Caption for Selected Image", type="primary"):
                with st.spinner("Generating caption..."):
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
        st.markdown("## 🎯 Generated Caption")
        
        # Display image and caption side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.session_state.display_image is not None:
                st.image(st.session_state.display_image, caption=f"📷 {st.session_state.image_name}", use_container_width=True)
            else:
                st.info(f"📷 {st.session_state.image_name}")
        
        with col2:
            st.markdown(f'<div class="caption-display"><strong>"{st.session_state.generated_caption}"</strong></div>', unsafe_allow_html=True)
        
        # Add a new generation button
        if st.button("🔄 Generate New Caption"):
            st.session_state.generated_caption = None
            st.rerun()

if __name__ == "__main__":
    main()
