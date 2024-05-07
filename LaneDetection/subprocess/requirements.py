import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import cv2
except ImportError:
    install("opencv-python")
try:
    import numpy as np
except ImportError:
    install("numpy")
try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")