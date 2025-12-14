# Edge Detection from Scratch

A comprehensive implementation of edge detection algorithms using only NumPy, built from scratch for educational purposes.

## Features

- **Sobel Edge Detection**: Classic gradient-based edge detection algorithm
- **Prewitt Edge Detection**: Simple gradient-based edge detection algorithm (similar to Sobel)
- **Canny Edge Detection**: Advanced multi-stage edge detection with non-maximum suppression and hysteresis
- **Pure NumPy Implementation**: All algorithms implemented from scratch without using OpenCV or similar libraries
- **Modular Architecture**: Clean, well-organized code structure

## Project Structure

```
edge_detection/
├── __init__.py          # Package initialization
├── sobel.py             # Sobel edge detection implementation
├── prewitt.py           # Prewitt edge detection implementation
├── canny.py             # Canny edge detection implementation
└── utils.py             # Utility functions (Gaussian blur, convolution, etc.)

main.py                  # Example usage and demonstration
streamlit_app.py         # Interactive web app for edge detection comparison
requirements.txt         # Python dependencies
README.md               # This file
```

## Installation

1. Clone or download this repository
2. Create and activate a virtual environment (recommended):

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import numpy as np
from edge_detection import sobel_edge_detection, prewitt_edge_detection, canny_edge_detection

# Load your image (grayscale or RGB)
image = np.array(...)  # Your image as numpy array

# Sobel edge detection
magnitude, grad_x, grad_y = sobel_edge_detection(
    image,
    blur=True,
    kernel_size=5,
    sigma=1.0,
    threshold=None  # Optional threshold for binary output
)

# Prewitt edge detection
magnitude, grad_x, grad_y = prewitt_edge_detection(
    image,
    blur=True,
    kernel_size=5,
    sigma=1.0,
    threshold=None  # Optional threshold for binary output
)

# Canny edge detection
edges = canny_edge_detection(
    image,
    low_threshold=50,
    high_threshold=150,
    kernel_size=5,
    sigma=1.0
)
```

### Running the Example

```bash
# Run with default sample image
python main.py

# Run with your own image
python main.py path/to/your/image.jpg
```

### Running the Streamlit Web App

For an interactive web interface to compare all edge detection algorithms:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser where you can:
- Upload images interactively
- Compare Sobel, Prewitt, and Canny side-by-side
- Adjust parameters in real-time
- View gradient components (X, Y, Magnitude)

## Algorithms

### Sobel Edge Detection

The Sobel operator uses two 3x3 kernels to compute the gradient in the x and y directions:

- **Gradient X**: Detects vertical edges
- **Gradient Y**: Detects horizontal edges
- **Magnitude**: Combined edge strength

The Sobel kernels use weighted coefficients (2 in the center row/column) which gives more weight to the center pixels, making it more accurate than Prewitt.

**Parameters:**
- `blur`: Apply Gaussian blur before edge detection (default: True)
- `kernel_size`: Size of Gaussian kernel (default: 5)
- `sigma`: Standard deviation for Gaussian (default: 1.0)
- `threshold`: Optional threshold for binary output (default: None)

### Prewitt Edge Detection

The Prewitt operator is similar to Sobel but uses simpler 3x3 kernels with uniform weights:

- **Gradient X**: Detects vertical edges
- **Gradient Y**: Detects horizontal edges
- **Magnitude**: Combined edge strength

Prewitt is simpler than Sobel (no weighted coefficients) but is less accurate. It's faster and less sensitive to noise.

**Parameters:**
- `blur`: Apply Gaussian blur before edge detection (default: True)
- `kernel_size`: Size of Gaussian kernel (default: 5)
- `sigma`: Standard deviation for Gaussian (default: 1.0)
- `threshold`: Optional threshold for binary output (default: None)

### Canny Edge Detection

The Canny algorithm is a multi-stage process:

1. **Gaussian Blur**: Reduce noise
2. **Gradient Calculation**: Compute magnitude and direction using Sobel operators
3. **Non-Maximum Suppression**: Thin edges by keeping only local maxima
4. **Double Thresholding**: Classify pixels as strong, weak, or non-edges
5. **Hysteresis**: Connect weak edges to strong edges

**Parameters:**
- `low_threshold`: Lower threshold for edge detection (default: 50)
- `high_threshold`: Upper threshold for edge detection (default: 150)
- `kernel_size`: Size of Gaussian kernel (default: 5)
- `sigma`: Standard deviation for Gaussian (default: 1.0)

## Implementation Details

All algorithms are implemented from scratch using only NumPy:

- **Convolution**: Manual 2D convolution implementation
- **Gaussian Blur**: Custom Gaussian kernel generation and application
- **Non-Maximum Suppression**: Direction-based edge thinning
- **Hysteresis**: Depth-first search for edge connectivity

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Visualization (for examples)
- `Pillow`: Image loading (optional, for loading image files)

## Educational Purpose

This implementation is designed for educational purposes to understand:
- How edge detection algorithms work internally
- Image processing fundamentals
- NumPy array operations
- Convolution operations

## License

This project is for educational purposes.


