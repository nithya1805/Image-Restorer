import cv2
import numpy as np

# -------------------------------------------------------
# OCR QUALITY THRESHOLDS (starting values)
# -------------------------------------------------------

THRESHOLDS = {
    "brightness_min": 90,
    "brightness_max": 200,
    "contrast_min": 35,
    "noise_max": 12,
    "sharpness_min": 80
}


# -------------------------------------------------------
# IMAGE QUALITY METRICS
# -------------------------------------------------------

def measure_brightness(gray):
    return float(np.mean(gray))


def measure_contrast(gray):
    return float(np.std(gray))


def measure_noise(gray):
    """Estimate noise using Gaussian residual."""
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - smooth.astype(np.float32)
    return float(np.std(residual))


def measure_sharpness(gray):
    """Tenengrad sharpness score."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


# -------------------------------------------------------
# MAIN QUALITY CHECK
# -------------------------------------------------------

def check_preprocessing_needed(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness = measure_brightness(gray)
    contrast = measure_contrast(gray)
    noise = measure_noise(gray)
    sharpness = measure_sharpness(gray)

    issues = {}

    issues["brightness"] = not (
        THRESHOLDS["brightness_min"]
        <= brightness
        <= THRESHOLDS["brightness_max"]
    )

    issues["contrast"] = contrast < THRESHOLDS["contrast_min"]
    issues["noise"] = noise > THRESHOLDS["noise_max"]
    issues["sharpness"] = sharpness < THRESHOLDS["sharpness_min"]

    preprocessing_needed = any(issues.values())

    return {
        "preprocessing": "YES" if preprocessing_needed else "NO",
        "metrics": {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "noise": round(noise, 2),
            "sharpness": round(sharpness, 2)
        },
        "issues": issues
    }