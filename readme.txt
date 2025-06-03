# 🦵 Knee Ligament Detection using MRI Images

This Streamlit application allows users to upload two knee MRI scans and automatically detects ligament-like structures using edge detection and texture analysis.

## 🚀 Features

- Upload and validate two MRI images.
- Detect grayscale dominance and edge density.
- Perform Local Binary Pattern (LBP) analysis for texture.
- Highlight ligament-like regions based on contours.
- User-friendly UI with custom CSS for enhanced UX.

## 🧠 How It Works

1. **Upload** two MRI images.
2. **Validate** the images to ensure they resemble grayscale MRI scans.
3. **Edge Detection** using Canny and Gaussian Blur.
4. **Contour Analysis** filters potential ligament-like shapes based on:
   - Area
   - Circularity
   - Aspect Ratio
5. **Visualization** of original, edge-detected, and annotated images.

## 🖼️ Example Output

- Original Image  
- Edge Detection  
- Ligament Detection Result  
- Text-based insights (success/warning/error)

## 📦 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
