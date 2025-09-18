import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from PIL import Image
import os
import io
import requests
import gdown

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Image Captioning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to download encodings from Google Drive
def download_encodings_from_drive():
    """Download encodings.pkl from Google Drive if not present locally."""
    encodings_path = "encodings.pkl"
    
    if not os.path.exists(encodings_path):
        st.info("Encodings file not found locally. Downloading from Google Drive...")
        
        # Google Drive file ID from the shared link
        file_id = "1aPRfA7008147pp0Ni7SUKO-0JZHjRcCe"
        
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Downloading encodings file...")
            progress_bar.progress(25)
            
            # Method 1: Try gdown first
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={file_id}&export=download"
                gdown.download(url, encodings_path, quiet=False)
                progress_bar.progress(75)
                
            except Exception as gdown_error:
                st.warning(f"gdown failed: {gdown_error}. Trying alternative method...")
                progress_bar.progress(50)
                
                # Method 2: Direct download with session
                session = requests.Session()
                response = session.get(f"https://drive.google.com/uc?id={file_id}", stream=True)
                
                # Check if we got a confirmation page
                if 'download_warning' in response.text or response.status_code != 200:
                    # Try to get the confirmation token
                    for line in response.text.split('\n'):
                        if 'confirm=' in line and 'download' in line:
                            import re
                            confirm_token = re.search(r'confirm=([0-9A-Za-z_]+)', line)
                            if confirm_token:
                                confirm_token = confirm_token.group(1)
                                break
                    else:
                        # Try a different approach
                        response = session.get(
                            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
                            stream=True
                        )
                
                # Download the file
                if response.status_code == 200:
                    with open(encodings_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                else:
                    raise Exception(f"Failed to download: HTTP {response.status_code}")
            
            progress_bar.progress(90)
            
            # Validate the downloaded file
            if os.path.exists(encodings_path):
                file_size = os.path.getsize(encodings_path)
                
                if file_size < 1000:  # File too small, likely an error page
                    # Read first few bytes to check if it's HTML
                    with open(encodings_path, 'rb') as f:
                        first_bytes = f.read(100)
                    
                    if b'<' in first_bytes or b'html' in first_bytes.lower():
                        os.remove(encodings_path)  # Remove the invalid file
                        raise Exception("Downloaded file appears to be HTML, not a pickle file. The Google Drive link might require permission.")
                
                # Try to load the pickle file to validate it
                try:
                    with open(encodings_path, 'rb') as f:
                        pickle.load(f)
                    progress_bar.progress(100)
                    status_text.text("Download completed and validated!")
                    st.success(f"Successfully downloaded encodings.pkl ({file_size:,} bytes)")
                    
                except pickle.UnpicklingError:
                    os.remove(encodings_path)
                    raise Exception("Downloaded file is not a valid pickle file.")
                    
            else:
                raise Exception("File was not created after download attempt.")
                
        except Exception as e:
            st.error(f"Failed to download encodings file: {e}")
            st.error("Please try one of these solutions:")
            st.error("1. Make sure the Google Drive file is publicly accessible")
            st.error("2. Download the file manually from the Google Drive link")
            st.error("3. Place the encodings.pkl file in the same directory as this app")
            
            st.markdown("### Manual Download Instructions:")
            st.markdown("1. Go to: https://drive.google.com/file/d/1aPRfA7008147pp0Ni7SUKO-0JZHjRcCe/view?usp=sharing")
            st.markdown("2. Click 'Download' or 'Download anyway' if warned")
            st.markdown("3. Save the file as 'encodings.pkl' in your app directory")
            st.markdown("4. Refresh this page")
            
            st.stop()
    
    else:
        # File exists, validate it
        try:
            with open(encodings_path, 'rb') as f:
                pickle.load(f)
            st.sidebar.info("✅ Using cached encodings.pkl")
        except:
            st.error("Existing encodings.pkl file is corrupted. Please delete it and restart the app.")
            st.stop()
    
    return encodings_path

# Cache the model and data loading functions
@st.cache_resource
def load_model_and_data():
    """Load the trained model and preprocessing data."""
    try:
        # Load trained model
        model = load_model('model.keras', compile=False)
        
        # Download and load image features
        encodings_path = download_encodings_from_drive()
        features = pickle.load(open(encodings_path, "rb"))
        
        # Load tokenizer mappings
        with open("wordtoix.pkl", "rb") as f:
            words_to_index = pickle.load(f)
        
        with open("ixtoword.pkl", "rb") as f:
            index_to_words = pickle.load(f)
        
        return model, features, words_to_index, index_to_words
    except FileNotFoundError as e:
        st.error(f"Model or data files not found: {e}")
        st.error("Please ensure the following files are in your app directory:")
        st.error("- model_2.keras")
        st.error("- wordtoix.pkl") 
        st.error("- ixtoword.pkl")
        st.error("- encodings.pkl (will be downloaded automatically)")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.stop()

@st.cache_data
def load_ground_truth_captions():
    """Load ground truth captions if available."""
    try:
        # You'll need to replace this with your actual ground truth data loading
        # This is a placeholder - adjust based on your data structure
        with open("ground_truth_captions.pkl", "rb") as f:
            ground_truth = pickle.load(f)
        return ground_truth
    except:
        return None

def generate_caption(picture, model, words_to_index, index_to_words, 
                    max_length=124, max_steps=25, temperature=0.7, top_k=5):
    """
    Generate a caption for a given image feature vector.
    
    Args:
        picture (np.array): Precomputed image feature vector, shape (1, feature_dim)
        model: Trained caption generation model
        words_to_index: Dictionary mapping words to indices
        index_to_words: Dictionary mapping indices to words
        max_length (int): Maximum sequence length
        max_steps (int): Maximum number of words to generate
        temperature (float): Temperature for sampling (controls randomness)
        top_k (int): Number of top probable words to sample from
    
    Returns:
        str: Generated caption
    """
    in_text = 'startseq'
    generated_words = []
    
    for _ in range(max_steps):
        # Convert current text to sequence
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='pre')
        
        # Predict next word probabilities
        yhat = model([picture, sequence], training=False)
        probabilities = yhat.numpy().ravel()
        
        # Temperature scaling
        probabilities = np.exp(np.log(probabilities + 1e-9) / temperature)
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

def main():
    st.title("🏥 Medical Image Captioning System")
    st.markdown("Generate automated captions for chest X-ray images using deep learning")
    
    # Load model and data (will auto-download encodings if needed)
    with st.spinner("Loading model and data..."):
        model, features, words_to_index, index_to_words = load_model_and_data()
        ground_truth = load_ground_truth_captions()
    
    st.sidebar.success(f"✅ Model loaded successfully!")
    st.sidebar.info(f"📊 {len(features)} images in dataset")
    
    # Sidebar for parameters
    st.sidebar.header("Caption Generation Parameters")
    max_steps = st.sidebar.slider("Max Steps", min_value=10, max_value=50, value=25, 
                                  help="Maximum number of words to generate")
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1,
                                    help="Controls randomness (lower = more focused)")
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=5,
                              help="Number of top words to sample from")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Selection")
        
        # Method selection
        method = st.radio("Choose input method:", 
                         ["Select from dataset", "Upload custom image", "Random selection"])
        
        if method == "Select from dataset":
            if features:
                image_list = list(features.keys())
                selected_idx = st.selectbox("Select image index:", 
                                           range(len(image_list)), 
                                           index=min(2000, len(image_list)-1))
                pic = image_list[selected_idx]
                image_vector = features[pic].reshape((1, -1))
                
                # Display image info
                st.info(f"Selected image: {pic}")
                
                # Try to load and display the image
                images_dir = st.text_input("Images directory path:", 
                                          value="/kaggle/input/chest-xrays-indiana-university/images/images_normalized/")
                
                if os.path.exists(images_dir + pic):
                    img_array = plt.imread(images_dir + pic)
                    st.image(img_array, caption=f"Image: {pic}", use_column_width=True, clamp=True)
                else:
                    st.warning(f"Image file not found at: {images_dir + pic}")
                    st.info("The system will still generate captions using precomputed features.")
            else:
                st.error("No precomputed features available")
                
        elif method == "Random selection":
            if st.button("🎲 Generate Random Image"):
                if features:
                    random_idx = np.random.randint(0, len(features))
                    pic = list(features.keys())[random_idx]
                    image_vector = features[pic].reshape((1, -1))
                    st.session_state['selected_pic'] = pic
                    st.session_state['image_vector'] = image_vector
                    st.success(f"Randomly selected: {pic}")
                    
            if 'selected_pic' in st.session_state:
                pic = st.session_state['selected_pic']
                image_vector = st.session_state['image_vector']
                st.info(f"Current selection: {pic}")
        
        elif method == "Upload custom image":
            st.warning("⚠️ Note: Custom image upload requires feature extraction with the same model used during training.")
            uploaded_file = st.file_uploader("Choose an X-ray image...", 
                                            type=['jpg', 'jpeg', 'png', 'dcm'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.error("Feature extraction for custom images is not implemented in this demo. Please use dataset images.")
    
    with col2:
        st.header("📝 Generated Caption")
        
        if st.button("🚀 Generate Caption", type="primary"):
            if 'image_vector' in locals() or 'image_vector' in st.session_state:
                try:
                    current_vector = locals().get('image_vector') or st.session_state.get('image_vector')
                    current_pic = locals().get('pic') or st.session_state.get('selected_pic')
                    
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(
                            current_vector, model, words_to_index, index_to_words,
                            max_steps=max_steps, temperature=temperature, top_k=top_k
                        )
                    
                    st.success("Caption generated successfully!")
                    st.markdown(f"**Generated Caption:**")
                    st.markdown(f'> "{caption}"')
                    
                    # Show ground truth if available
                    if ground_truth and current_pic in ground_truth:
                        st.markdown("**Ground Truth Caption:**")
                        st.markdown(f'> "{ground_truth[current_pic]}"')
                    
                    # Show statistics
                    st.markdown("**Caption Statistics:**")
                    word_count = len(caption.split())
                    st.metric("Word Count", word_count)
                    
                except Exception as e:
                    st.error(f"Error generating caption: {e}")
            else:
                st.warning("Please select an image first!")
    
    # Additional information
    st.markdown("---")
    st.header("ℹ️ About This System")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Model Information:**
        - Deep learning model for medical image captioning
        - Trained on chest X-ray dataset
        - Uses attention mechanism for caption generation
        - Supports temperature-controlled sampling
        """)
    
    with info_col2:
        st.markdown("""
        **Parameters Guide:**
        - **Max Steps**: Maximum words in caption
        - **Temperature**: Lower = more focused, Higher = more creative
        - **Top-K**: Number of candidate words to consider
        """)
    
    # Display model stats if available
    if features:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Dataset Statistics")
        st.sidebar.metric("Total Images", len(features))
        if features:
            feature_dim = list(features.values())[0].shape[0] if features else "Unknown"
            st.sidebar.metric("Feature Dimension", feature_dim)

if __name__ == "__main__":
    main()
