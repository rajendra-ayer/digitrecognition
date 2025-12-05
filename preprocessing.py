from PIL import Image, ImageOps
import numpy as np

def preprocess(img):
    img = img.convert("L")
    arr = np.array(img)

    threshold = 250
    mask = arr < threshold
    if mask.any():
        ys, xs = np.where(mask)
        arr = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
        img = Image.fromarray(arr)

    w, h = img.size
    size = max(w, h)
    new_img = Image.new("L", (size, size), 255)
    new_img.paste(img, ((size - w)//2, (size - h)//2))

    new_img = new_img.resize((28, 28), Image.LANCZOS)

    arr = np.array(new_img)
    if arr.mean() > 127:
        new_img = ImageOps.invert(new_img)

    arr = np.array(new_img).astype("float32") / 255.0
    arr = arr.reshape(28, 28, 1)
    return arr
