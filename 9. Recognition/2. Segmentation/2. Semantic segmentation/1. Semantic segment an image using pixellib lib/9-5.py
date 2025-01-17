from pixellib.semantic import semantic_segmentation
import cv2 as cv

seg = semantic_segmentation()
seg.load_ade20k_model('deeplabv3_xception65_ade20k.h5')


img_fname = "bus_1713.png"
seg.segmentAsAde20k(img_fname, output_image_name='image_new.jpg')
info1, img_segmented1 = seg.segmentAsAde20k(img_fname)
info2, img_segmented2 = seg.segmentAsAde20k(img_fname, overlay=True)

cv.imshow('Image original', cv.imread(img_fname))
cv.imshow('Image segmentation', img_segmented1)
cv.imshow('Image segmentation overlayed', img_segmented2)

cv.waitKey()
cv.destroyAllWindows()
