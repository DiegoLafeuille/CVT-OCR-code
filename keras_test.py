import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import keras_ocr

plt.style.use('ggplot')

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


img = cv2.imread('images/cvt_spectrometer_zoomed.jpg')
img = ResizeWithAspectRatio(img, height= 720)
rois = cv2.selectROIs("Select ROIs", img)
cv2.destroyWindow("Select ROIs")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# print(rois)

# Apply OCR
pipeline = keras_ocr.pipeline.Pipeline()


roi_dfs = []
for i, roi in enumerate(rois):
    roi_img = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    results = pipeline.recognize([roi_img])
    result = results[0]
    if result == '':
        result = 'No text'
    img_id = f"{i}th ROI"
    roi_df = pd.DataFrame(result, columns=['text', 'bbox'])
    
    box_adjust = np.array([[roi[0], roi[1]],[roi[0], roi[1]],[roi[0], roi[1]],[roi[0], roi[1]]])  
    for box in roi_df['bbox']:
        box += box_adjust
    
    roi_dfs.append(roi_df)
kerasocr_df = pd.concat(roi_dfs)


# Plot
fig, axs = plt.subplots(1, 1, figsize=(15, 10))
keras_results = kerasocr_df[['text','bbox']].values.tolist()
print(kerasocr_df.iloc[0]['bbox'])
keras_results = [(x[0], np.array(x[1])) for x in keras_results]
keras_ocr.tools.drawAnnotations(img, 
                                keras_results, ax=axs)
plt.show()



