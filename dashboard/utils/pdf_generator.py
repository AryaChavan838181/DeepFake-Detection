import os
import io
import cv2
import librosa
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image

def build_pdf_report(file_info, results, output_stream):
    """
    file_info: dict, e.g. {'name': 'video.mp4', 'type': 'video'}
    results: dict containing 'scores' and 'plots' streams
    """
    c = canvas.Canvas(output_stream, pagesize=A4)
    width, height = A4
    
    # Background and Title
    c.setFillColorRGB(0.1, 0.1, 0.12)
    c.rect(0, 0, width, height, fill=1)
    
    c.setFillColorRGB(1.0, 0.84, 0.0)
    c.setFont("Courier-Bold", 24)
    c.drawString(40, height - 50, "DFDETECTIVE FORENSIC REPORT")
    
    c.setFont("Courier", 12)
    c.setFillColorRGB(0.8, 0.8, 0.8)
    c.drawString(40, height - 75, f"FILE: {file_info['name']}")
    c.drawString(40, height - 90, f"TYPE: {file_info['type'].upper()}")
    
    # Verdict
    y_pos = height - 130
    conf = results['confidence'] * 100
    is_deepfake = results['is_deepfake']

    if is_deepfake:
        c.setFillColorRGB(1.0, 0.3, 0.3)
        c.setFont("Courier-Bold", 16)
        c.drawString(40, y_pos, "[ ALERT ] DEEPFAKE DETECTED")
        c.setFont("Courier", 12)
        c.drawString(40, y_pos - 15, f"{conf:.1f}% Synthetic Probability")
    else:
        c.setFillColorRGB(0.0, 1.0, 0.67)
        c.setFont("Courier-Bold", 16)
        c.drawString(40, y_pos, "[ VERIFIED ] AUTHENTIC CONTENT")
        c.setFont("Courier", 12)
        c.drawString(40, y_pos - 15, f"{conf:.1f}% Authentic Confidence")

    y_pos -= 50
    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.setFont("Courier-Bold", 14)
    c.drawString(40, y_pos, "--- ENSEMBLE METRICS ---")
    y_pos -= 20
    
    scores = results.get('scores', {})
    for k, v in scores.items():
        c.setFont("Courier", 12)
        c.drawString(40, y_pos, f"{k}: {v:.4f}")
        y_pos -= 15
        
    y_pos -= 20
    c.setFont("Courier-Bold", 14)
    c.drawString(40, y_pos, "--- FORENSIC EVIDENCE ---")
    y_pos -= 20
    
    # Render Plots
    from reportlab.lib.utils import ImageReader
    plots = results.get('plots', [])
    for plot in plots:
        if y_pos < 200:
            c.showPage()
            c.setFillColorRGB(0.1, 0.1, 0.12)
            c.rect(0, 0, width, height, fill=1)
            y_pos = height - 50
            
        c.setFillColorRGB(1.0, 0.84, 0.0)
        c.setFont("Courier", 12)
        c.drawString(40, y_pos, plot['title'])
        y_pos -= 10
        
        # plot['stream'] is an io.BytesIO containing valid PNG/JPEG
        # We need to render it onto PDF
        if plot['stream']:
            try:
                img_reader = ImageReader(plot['stream'])
                # Maintain aspect ratio
                iw, ih = img_reader.getSize()
                aspect = ih / float(iw)
                pw = 400
                ph = pw * aspect
                c.drawImage(img_reader, 50, y_pos - ph, width=pw, height=ph)
                y_pos -= (ph + 30)
            except Exception as e:
                c.drawString(40, y_pos, f"Failed to render plot: {e}")
                y_pos -= 20
        else:
            y_pos -= 10
            
    c.save()
    return True
