"""
Streamlit Web App for Edge Detection
Upload an image and compare Sobel, Prewitt, and Canny edge detection results.
"""

import sys
import os

# Add current directory to Python path to find edge_detection module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
import numpy as np
from PIL import Image
from edge_detection import sobel_edge_detection, prewitt_edge_detection, canny_edge_detection
from edge_detection.utils import orientation_to_color

# Page configuration
st.set_page_config(
    page_title="Edge Detection Comparison",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Edge Detection Comparison Tool")
st.markdown("""
Upload an image to compare **Sobel**, **Prewitt**, and **Canny** edge detection algorithms.
All algorithms are implemented from scratch using NumPy.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Parameters")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Upload an image to process"
)

# Algorithm selection
st.sidebar.subheader("Algorithms")
show_sobel = st.sidebar.checkbox("Sobel", value=True)
show_prewitt = st.sidebar.checkbox("Prewitt", value=True)
show_canny = st.sidebar.checkbox("Canny", value=True)

# Sobel parameters
if show_sobel:
    st.sidebar.subheader("Sobel Parameters")
    sobel_blur = st.sidebar.checkbox("Apply Gaussian Blur", value=True, key="sobel_blur")
    sobel_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2, key="sobel_kernel")
    sobel_sigma = st.sidebar.slider("Blur Sigma", 0.5, 3.0, 1.0, 0.1, key="sobel_sigma")
    sobel_threshold = st.sidebar.slider("Threshold (0-255)", 0, 255, None, key="sobel_threshold")
    if sobel_threshold == 0:
        sobel_threshold = None

# Prewitt parameters
if show_prewitt:
    st.sidebar.subheader("Prewitt Parameters")
    prewitt_blur = st.sidebar.checkbox("Apply Gaussian Blur", value=True, key="prewitt_blur")
    prewitt_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2, key="prewitt_kernel")
    prewitt_sigma = st.sidebar.slider("Blur Sigma", 0.5, 3.0, 1.0, 0.1, key="prewitt_sigma")
    prewitt_threshold = st.sidebar.slider("Threshold (0-255)", 0, 255, None, key="prewitt_threshold")
    if prewitt_threshold == 0:
        prewitt_threshold = None

# Canny parameters
if show_canny:
    st.sidebar.subheader("Canny Parameters")
    canny_low_threshold = st.sidebar.slider("Low Threshold", 0, 200, 50, key="canny_low")
    canny_high_threshold = st.sidebar.slider("High Threshold", 0, 300, 150, key="canny_high")
    canny_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2, key="canny_kernel")
    canny_sigma = st.sidebar.slider("Blur Sigma", 0.5, 3.0, 1.0, 0.1, key="canny_sigma")

# Main content area
if uploaded_file is not None:
    try:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Display original image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("📷 Original Image")
            st.image(image)
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
        
        # Process button
        if st.button("🚀 Process Image", type="primary"):
            with st.spinner("Processing image..."):
                results = {}
                
                # Sobel edge detection
                if show_sobel:
                    try:
                        sobel_mag, sobel_x, sobel_y = sobel_edge_detection(
                            img_array,
                            blur=sobel_blur,
                            kernel_size=sobel_kernel_size,
                            sigma=sobel_sigma,
                            threshold=sobel_threshold
                        )
                        results['sobel'] = {
                            'magnitude': sobel_mag,
                            'gradient_x': sobel_x,
                            'gradient_y': sobel_y
                        }
                    except Exception as e:
                        st.error(f"Sobel processing error: {e}")
                
                # Prewitt edge detection
                if show_prewitt:
                    try:
                        prewitt_mag, prewitt_x, prewitt_y = prewitt_edge_detection(
                            img_array,
                            blur=prewitt_blur,
                            kernel_size=prewitt_kernel_size,
                            sigma=prewitt_sigma,
                            threshold=prewitt_threshold
                        )
                        results['prewitt'] = {
                            'magnitude': prewitt_mag,
                            'gradient_x': prewitt_x,
                            'gradient_y': prewitt_y
                        }
                    except Exception as e:
                        st.error(f"Prewitt processing error: {e}")
                
                # Canny edge detection
                if show_canny:
                    try:
                        canny_edges, canny_orientation = canny_edge_detection(
                            img_array,
                            low_threshold=canny_low_threshold,
                            high_threshold=canny_high_threshold,
                            kernel_size=canny_kernel_size,
                            sigma=canny_sigma,
                            return_orientation=True
                        )
                        # Create colored orientation map
                        canny_colored = orientation_to_color(canny_orientation, canny_edges)
                        results['canny'] = {
                            'edges': canny_edges,
                            'orientation': canny_orientation,
                            'colored': canny_colored
                        }
                    except Exception as e:
                        st.error(f"Canny processing error: {e}")
                
                # Display results
                if results:
                    st.success("✅ Processing complete!")
                    st.divider()
                    
                    # Comparison view - Magnitude/Edges
                    st.subheader("📊 Edge Detection Comparison")
                    
                    # Create columns based on number of algorithms
                    num_algorithms = len(results)
                    cols = st.columns(num_algorithms)
                    
                    col_idx = 0
                    if 'sobel' in results:
                        with cols[col_idx]:
                            st.markdown("### Sobel (Magnitude)")
                            st.image(results['sobel']['magnitude'], clamp=True)
                        col_idx += 1
                    
                    if 'prewitt' in results:
                        with cols[col_idx]:
                            st.markdown("### Prewitt (Magnitude)")
                            st.image(results['prewitt']['magnitude'], clamp=True)
                        col_idx += 1
                    
                    if 'canny' in results:
                        with cols[col_idx]:
                            st.markdown("### Canny (Edges)")
                            st.image(results['canny']['edges'], clamp=True)
                        col_idx += 1
                    
                    # Show colored orientation visualization
                    if 'canny' in results and 'colored' in results['canny']:
                        st.divider()
                        st.subheader("🎨 Edge Orientation Visualization")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("""
                            **Color Mapping by Edge Orientation:**
                            
                            Each color represents the gradient direction (edge orientation):
                            
                            - 🔴 **Red** (0°) → Horizontal edges (left-right)
                            - 🟡 **Yellow** (60°) → Diagonal edges
                            - 🟢 **Green** (120°) → Vertical edges (top-bottom)
                            - 🔵 **Cyan** (180°) → Horizontal edges (right-left)
                            - 🔵 **Blue** (240°) → Diagonal edges
                            - 🟣 **Magenta** (300°) → Vertical edges (bottom-top)
                            
                            **Color Cycle:** Red → Yellow → Green → Cyan → Blue → Magenta → Red
                            
                            Similar orientations have similar colors, making edge direction patterns easy to identify.
                            """)
                        with col2:
                            st.image(results['canny']['colored'], clamp=True)
                    
                    # Detailed views for Sobel and Prewitt
                    if 'sobel' in results or 'prewitt' in results:
                        st.divider()
                        st.subheader("🔍 Detailed Gradient Analysis")
                        
                        # Sobel details
                        if 'sobel' in results:
                            st.markdown("#### Sobel Gradients")
                            sobel_cols = st.columns(3)
                            with sobel_cols[0]:
                                st.markdown("**Magnitude**")
                                st.image(results['sobel']['magnitude'], clamp=True)
                            with sobel_cols[1]:
                                st.markdown("**Gradient X**")
                                st.image(results['sobel']['gradient_x'], clamp=True)
                            with sobel_cols[2]:
                                st.markdown("**Gradient Y**")
                                st.image(results['sobel']['gradient_y'], clamp=True)
                        
                        # Prewitt details
                        if 'prewitt' in results:
                            st.markdown("#### Prewitt Gradients")
                            prewitt_cols = st.columns(3)
                            with prewitt_cols[0]:
                                st.markdown("**Magnitude**")
                                st.image(results['prewitt']['magnitude'], clamp=True)
                            with prewitt_cols[1]:
                                st.markdown("**Gradient X**")
                                st.image(results['prewitt']['gradient_x'], clamp=True)
                            with prewitt_cols[2]:
                                st.markdown("**Gradient Y**")
                                st.image(results['prewitt']['gradient_y'], clamp=True)
                    
                    # Side-by-side comparison
                    st.divider()
                    st.subheader("⚖️ Side-by-Side Comparison")
                    
                    comparison_images = []
                    comparison_labels = []
                    
                    if 'sobel' in results:
                        comparison_images.append(results['sobel']['magnitude'])
                        comparison_labels.append("Sobel")
                    if 'prewitt' in results:
                        comparison_images.append(results['prewitt']['magnitude'])
                        comparison_labels.append("Prewitt")
                    if 'canny' in results:
                        comparison_images.append(results['canny']['edges'])
                        comparison_labels.append("Canny")
                    
                    if len(comparison_images) > 1:
                        # Create a horizontal comparison
                        comp_cols = st.columns(len(comparison_images))
                        for idx, (img, label) in enumerate(zip(comparison_images, comparison_labels)):
                            with comp_cols[idx]:
                                st.markdown(f"**{label}**")
                                st.image(img, clamp=True)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.exception(e)

else:
    # Show instructions when no image is uploaded
    st.info("👆 Please upload an image to get started!")
    
    # Show example or instructions
    st.markdown("""
    ### How to use:
    1. **Upload an image** using the file uploader above
    2. **Select algorithms** you want to compare in the sidebar
    3. **Adjust parameters** for each algorithm (optional)
    4. **Click "Process Image"** to see the results
    
    ### Features:
    - ✅ Compare multiple edge detection algorithms side-by-side
    - ✅ Adjustable parameters for each algorithm
    - ✅ View gradient components (X, Y, Magnitude)
    - ✅ Real-time processing with visual feedback
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Edge Detection from Scratch | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)

