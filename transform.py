import numpy as np

def read_raw(file_path):
    width = 512
    height = 512
    channels = 204
    total = width * height * channels

    img = np.fromfile(file_path, dtype=np.uint16)
    if img.size != total:
        raise ValueError("The file does not match the expected dimensions.")
    img = img.reshape((height, width, channels))
    return img

def save_raw(file_path, image):
    image.astype(np.uint8).tofile(file_path)
    print(f"Successfully saved the image at {file_path}")

def rotate(image, angle):
    angle = angle % 360

    if angle == 0:
        return image
    elif angle == 90:
        return np.transpose(image, (1, 0, 2))[:, ::-1, :]
    elif angle == 180:
        return image[::-1, ::-1, :]
    elif angle == 270:
        return np.transpose(image, (1, 0, 2))[::-1, :, :]
    
def mirror(image, axis):
    #left to right
    if axis == 'x':
        return image[:, ::-1, :]
    #up to down
    elif axis == 'y':
        return image[::-1, :, :]
    else:
        raise ValueError("Axis must be 'x' or 'y'")
