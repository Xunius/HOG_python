# HOG_python

Vectorized Histogram of Orientated Gradients (HOG) feature extraction using Python (numpy+scipy)

This is a python implementation of Histogram of Orientated Gradients (HOG)
using
[skimage's](http://scikit-image.org/docs/0.13.x/auto_examples/features_detection/plot_hog.html)
as a reference, with faster speed, particularly when applied with a sliding
windows method.

Processing a single image of size (512x512), it shows a speed gain of about
20 % wrt skimage. If the input images all come in the same size, it is possible
to compute an indices array for the 1st input image, and feed the indices to
subsequent images to boost the speed gain to ~ 30%.

The speed advantage really gets revealed when applying a sliding window method
on an input image. The vectorized implementation returns all HOG features for
all sliding windows at a single scale in one go. Compared with a native sliding
window + skimage approach, this can speed up by 20 -- 30 times. A single image
extraction is a special case of a sliding window extraction with the window size
being the same as the image.

The key idea is to convert the histogram computation to a 3D convolution
and to use FFT algorithms to leverage the speed. In a native sliding window
method there is always a lot of repeated computations. In the case of HOG extraction,
one only needs to compute the HOGs via FFT-convolution for once and pick out
pixels at correct locations to form a feature vector.

I've tried to write a Fortran version but unfortunately my Fortran knowledge is
perhaps too poor and it ends up being slower than the numpy+scipy version.
If you are interested and good at Fortran maybe you can help optimize it
and see how much more this can be further sped up.

I'm less sure about extending the vectorization to cover multiple scales
of an image, as the scaled convolution doesn't always equal to the
convolution of scaled image.



