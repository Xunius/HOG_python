'''Creates histogram of orientated gradients (HOG) feature descriptor.

Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2018-03-12 21:33:10.
'''


#--------Import modules-------------------------
import numpy
from scipy.signal import fftconvolve
import time
from skimage.feature import hog as skhog
from skimage import color



#----------------Utility functions----------------
def getLine(beta,x,y):
    '''Get a line of pixels given slope and a point'''
    if abs(beta)>=1:
        xhat=y/beta
        result=numpy.where((x>=xhat) & (x<=xhat+1),1,0)
    else:
        yhat=x*beta
        result=numpy.where((y>=yhat) & (y<=yhat+1),1,0)
    return result

def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.

    <arr>: ndarray of rank 2.
    <sub_shape>: tuple of length 2, window size: (ny, nx).
    <stride>: int, stride of windows.

    Return <subs>: strided window view.
    
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides
    m1,n1=arr.shape
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride,1+(n1-n2)//stride,m2,n2)
    strides=(stride*s0,stride*s1,s0,s1)
    subs=numpy.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def asStride2(arr,sub_shape):
    '''Get non-overlap sub-matrices of an ndarray.

    <arr>: ndarray of rank 3.
    <sub_shape>: window shape (wy, wx).

    Return <rows>: sub-matrices. Number of rows=ny//wy*wy, number of columns=
                   nx//wx*wx (ignore imcomplete windows).
    '''
    wy,wx=sub_shape
    ny=arr.shape[0]//wy
    nx=arr.shape[1]//wx
    arr=arr[:ny*wy,:nx*wx,:]
    cols=numpy.array(numpy.array_split(arr,nx,axis=1))
    rows=numpy.array(numpy.array_split(cols,ny,axis=1))

    return rows

def conv3D(var,kernel):
    '''3D convolution using scipy.signal.convolve
    This seems to be the fastest among the 3.
    '''
    return fftconvolve(var,kernel,mode='valid')

def conv3D2(var,kernel):
    '''3D convolution by sub-matrix summing.
    NOTE: kernel is all ones, so the convolution is a summation of n shifted
    sub-matrices, n=sum(kernel).
    '''
    ny,nx,nz=var.shape
    ky,kx,kz=kernel.shape

    result=0
    for ii in range(kernel.size):
        yi,xi=numpy.unravel_index(ii,(ky,kx))
        slabii=var[yi:ny-ky+yi+1:1, xi:nx-kx+xi+1:1,:]
        result+=slabii

    return result

def conv3D3(var,kernel):
    '''3D convolution by strided view.
    '''
    aa=asStride(var,kernel.shape,1)
    #return numpy.tensordot(aa,kernel,axes=((2,3),(0,1)))
    return aa.sum(axis=(2,3))

def getGradients(img,ndim):
    '''Compute gradients and gradient magnitudes.'''

    gy=numpy.gradient(img,axis=0)
    gx=numpy.gradient(img,axis=1)
    weights=numpy.sqrt(gy**2+gx**2)
    if ndim==2:
        theta=numpy.arctan2(gy,gx)*180./numpy.pi%180.
    elif ndim==3:
        #-------Select from channel with max weights-------
        theta3=numpy.arctan2(gy,gx)*180./numpy.pi%180.
        maxgrads=numpy.argmax(weights,axis=2)
        theta=numpy.zeros([img.shape[0],img.shape[1]])
        for ii in range(3):
            theta=numpy.where(maxgrads==ii,theta3[:,:,ii],theta)
        weights=numpy.max(weights,axis=2)

    return theta,weights

def normalizeBlock(block,method,eps=1e-5):
    '''Normalize hog blocks, for hog()'''
    if method=='l1':
        norm=numpy.sum(numpy.abs(block))+eps
        result=block/norm
    elif method=='l2':
        norm=numpy.sqrt(numpy.sum(block**2)+eps**2)
        result=block/norm
    elif method=='l1-sqrt':
        norm=numpy.sum(numpy.abs(block))+eps
        result=numpy.sqrt(block/norm)

    return result

def getNormalizeFunc(method,eps=1e-5):
    '''Get a function to normalize hog blocks, for slideWindowHOG()'''
    if method=='l1':
        norm_func=lambda x:x/(numpy.abs(x).sum(axis=(1,2,3),keepdims=True)+eps)
    elif method=='l2':
        norm_func=lambda x:x/(numpy.sqrt((x**2).sum(axis=(1,2,3),keepdims=True))+eps**2)
    elif method=='l1-sqrt':
        norm_func=lambda x:numpy.sqrt(x/(numpy.abs(x).sum(axis=(1,2,3),keepdims=True))+eps)
    return norm_func


#--------------Extract hog from image--------------
def hog(img,orientations=9,pixels_per_cell=(8,8),
        cells_per_block=(3,3),block_norm='l1',visualise=False,
        transform_sqrt=False,feature_vector=True,verbose=True):
    '''Extract HOG features from an image. Standard version.

    <img>: 2d or 3d image array.
    <orientations>: int, number of bins in gradient orientation histogram
                    computation.
    <pixels_per_cell>: (cy, cx) number of pixels in a cell.
    <cells_per_block>: (by, bx) number of cells in a normalization block.
    <block_norm>: str, block normalization method.
                  'l1': normalize by l-1 norm.
                  'l2': normalize by l-2 norm.
                  'l1-sqrt': normalize by l-1 norm, followed by square root.
    <visualise>: bool, whether to return a visualiztion of the HOG.
    <transform_sqrt>: bool, whether to normalize the input image by
                      sqrt or not.
    <feature_vector>: bool, if True, return feature vector as a flatten vector.
    
    NOTE: this is a basic implementation of the HOG algorithm, following the
          steps of a standard HOG without vectorization, and works
          on a single window in a single image.

          For fast computations of HOG features with sliding windows on an
          image, use slideWindowHOG().

          For fast computations of HOG features on a single image, can
          use slideWindowHOG() with a window size same as input image.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2018-03-08 10:05:35.
    '''

    #-------------------Check inputs-------------------
    ndim=numpy.ndim(img)
    if ndim!=2 and ndim!=3:
        raise Exception("<img> needs to be 2D or 3D.")
    if not isinstance(orientations,(int,long)) or orientations<=0:
        raise Exception("<orientations> needs to be an positive int.")
    if not block_norm in ['l1','l2','l1-sqrt']:
        raise Exception("<block_norm> needs to be one of ['l1','l2','l1-sqrt'].")

    #-----------apply power law compression-----------
    if transform_sqrt:
        img=numpy.sqrt(img)

    if ndim==2:
        height,width=img.shape
    else:
        height,width,channels=img.shape
    cy,cx=pixels_per_cell
    by,bx=cells_per_block

    # Get gradient orientations in [0,180]
    theta,weights=getGradients(img,ndim)

    #-----Compute histogram of gradients in cells-----
    ny=height//cy
    nx=width//cx

    #----------------Loop though cells----------------
    hists=[]
    for ii in range(ny):
        histsii=[]
        for jj in range(nx):
            thetaij=theta[ii*cy:(ii+1)*cy, jj*cx:(jj+1)*cx]
            weightsij=weights[ii*cy:(ii+1)*cy, jj*cx:(jj+1)*cx]
            histij,binedges=numpy.histogram(thetaij,bins=orientations,
                    range=(0,180),weights=weightsij)
            histij=histij/cy/cx
            histsii.append(histij)
        hists.append(histsii)
    hists=numpy.array(hists)

    #--------Create visualization of gradients--------
    if visualise:

        # NOTE: the choice of gradient angles would introduce some differences
        # to the resultant plot. 
        #thetas=0.5*(binedges[:-1]+binedges[1:])
        thetas=binedges[:-1]
        thetas=numpy.tan((thetas+90.)/180.*numpy.pi)
        hog_image=numpy.zeros([height,width])
        xcell=numpy.arange(cx)
        ycell=numpy.arange(cy)
        xcell,ycell=numpy.meshgrid(xcell-xcell.mean(),ycell-ycell.mean())

        for ii in range(ny):
            for jj in range(nx):
                for kk,tkk in enumerate(thetas):
                    linekk=getLine(tkk,xcell,ycell)*hists[ii,jj,kk]
                    hog_image[ii*cy:(ii+1)*cy, jj*cx:(jj+1)*cx]+=linekk

    #--------------Normalize over blocks--------------
    feature=[]
    for ii in range(ny-(by-1)):
        fsii=[]
        for jj in range(nx-(bx-1)):
            blockij=hists[ii:ii+by, jj:jj+bx, :]
            blockij=normalizeBlock(blockij,block_norm)
            fsii.append(blockij)
        feature.append(fsii)

    feature=numpy.array(feature)

    if feature_vector:
        feature=feature.flatten()

    if visualise:
        return feature,hog_image, hists
    else:
        return feature, hists


#---Extract hog from sliding windows of an image---
def slideWindowHOG(img,window_size=(128,64),stride=4,
        indices=None,return_indices=True,
        orientations=9,pixels_per_cell=(8,8),
        cells_per_block=(3,3),block_norm='l1',
        transform_sqrt=False,feature_vector=True,verbose=True):
    '''Extract HOG features from an image. Vectorized version.

    <img>: 2d or 3d image array.
    <window_size>: size of sliding window (wy, wx).
    <stride>: int, strides in moving sliding window.
    <indices>: 1d array or None, indices to index the convolution to get values
               of cells in all sliding windows. If None, compute the
               indices. If the input images all have same shape, can
               compute indices for the 1st image and apply to all subsequent
               calls to speed up.
    <return_indices>: bool, whether to return <indices>.
    <orientations>: int, number of bins in gradient orientation histogram
                    computation.
    <pixels_per_cell>: (cy, cx) number of pixels in a cell.
    <cells_per_block>: (by, bx) number of cells in a normalization block.
    <block_norm>: str, block normalization method.
                  'l1': normalize by l-1 norm.
                  'l2': normalize by l-2 norm.
                  'l1-sqrt': normalize by l-1 norm, followed by square root.
    <transform_sqrt>: bool, whether to normalize the input image by
                      sqrt or not.
    <feature_vector>: bool, if True, return feature vector as a flatten vector.
    
    NOTE: this version computes HOG features from sliding windows in an image.
    Can also be used to extract HOG from a single window in an image, by
    setting the window_size same as image size (not including channel dimension).

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2018-03-08 10:17:06.
    '''

    #-------------------Check inputs-------------------
    ndim=numpy.ndim(img)
    if ndim!=2 and ndim!=3:
        raise Exception("<img> needs to be 2D.")
    if not isinstance(orientations,(int,long)) or orientations<=0:
        raise Exception("<orientations> needs to be an positive int.")
    if not block_norm in ['l1','l2','l1-sqrt']:
        raise Exception("<block_norm> needs to be one of ['l1','l2','l1-sqrt'].")

    #-----------apply power law compression-----------
    if transform_sqrt:
        img=numpy.sqrt(img)

    if ndim==2:
        height,width=img.shape
    else:
        height,width,channels=img.shape
    wy,wx=window_size
    cy,cx=pixels_per_cell
    by,bx=cells_per_block

    # Get gradient orientations in [0,180]
    theta,weights=getGradients(img,ndim)
    binedges=numpy.linspace(0,180.,orientations+1)

    #--------------Compute 3d convolution--------------
    #NOTE: for orientations=9, this layer-by-layer 3d convolution actually
    #turns out faster than a true 3d convolution
    kernel=numpy.ones(pixels_per_cell)
    convs=numpy.zeros((height-cy+1,width-cx+1)+(orientations,))
    for ii in range(orientations):
        countii=numpy.where((theta>=binedges[ii]) & (theta<binedges[ii+1]),1,0)
        convii=fftconvolve(countii*weights,kernel,mode='valid')
        #convii=conv3D3(countii*weights,kernel)
        convs[:,:,ii]=convii/cy/cx

    # number of complete cells in a window
    m1=window_size[0]//pixels_per_cell[0]
    m2=window_size[1]//pixels_per_cell[1]
    # number of complete windows in image
    n1=(height-wy)//stride+1
    n2=(width-wx)//stride+1

    #---------------Get strided indices---------------
    if indices is None:
        indices=[]

        #--------Indices in a cell, fixed--------
        dy0=numpy.arange(0,pixels_per_cell[0]*m1,pixels_per_cell[0])
        dx0=numpy.arange(0,pixels_per_cell[1]*m2,pixels_per_cell[1])
        dz0=numpy.arange(orientations)
        dx0,dy0,dz0=numpy.meshgrid(dx0,dy0,dz0)

        #------Indices in a cell in 1st window, fixed------
        conts=(dx0*convs.shape[2]+dy0*convs.shape[1]*convs.shape[2]+dz0).flatten()
        conts=numpy.repeat(conts[None,:],n1*n2,axis=0)

        #---Indices in a cell for all strided windows, dynamic--
        idys,idxs=numpy.unravel_index(numpy.arange(n1*n2),(n1,n2))
        dyis=stride*idys*convs.shape[1]*convs.shape[2]
        dxjs=stride*idxs*convs.shape[2]

        #-----------------Raveled indices-----------------
        indices=conts+(dyis+dxjs)[:,None]

        '''
        for ii in range(n1*n2):
            #idy,idx=numpy.unravel_index(ii,(n1,n2))
            idy=ii//n2
            idx=ii%n2
            #dy=dy0+stride*idy
            #dx=dx0+stride*idx
            #dx,dy,dz=numpy.meshgrid(dx,dy,dz0)
            #idxii=dy*convs.shape[1]*convs.shape[2]+dx*convs.shape[2]+dz
            idxii=dy0s+stride*idy*convs.shape[1]*convs.shape[2]+\
                    dx0s+stride*idx*convs.shape[2]+ dz0
            idxii=idxii.flatten()
            indices.append(idxii)
        '''

    #indices=numpy.array(indices)
    hists=numpy.take(convs,indices).reshape([n1*n2,m1,m2,orientations])

    #--------------Normalize over blocks--------------
    b1=m1-by+1
    b2=m2-bx+1
    feature=numpy.empty((n1*n2,b1,b2)+tuple(cells_per_block)+(orientations,))
    norm_func=getNormalizeFunc(block_norm)

    for ii in range(b1*b2):
        idy,idx=numpy.unravel_index(ii,(b1,b2))
        blockii=hists[:,idy:idy+by, idx:idx+bx, :]
        blockii=norm_func(blockii)
        feature[:,idy,idx,:,:,:]=blockii

    if feature_vector:
        feature=feature.reshape((-1),numpy.prod(feature.shape[1:]))
    if return_indices:
        return feature,indices
    else:
        return feature






#-------------Main---------------------------------
if __name__=='__main__':

    #--------------------Read image--------------------
    from skimage import data
    img=data.astronaut()
    img2=color.rgb2gray(img) # convert to gray because in this version (0.13.0)
    # skimage doesn't support 3D images yet.
    print 'image shape',img2.shape

    #--------------Get HOG using skimage--------------
    #NOTE in this version of skimage, pixels_per_cell is (cx,cy).
    #In newer versions (0.14dev at least), it changes to (cy,cs). 
    t1=time.time()
    fd,hog_image,hists=skhog(img2,orientations=9,pixels_per_cell=(8,16),
            transform_sqrt=True,
            feature_vector=False,
            block_norm='L1',
            cells_per_block=(3,3),visualise=True)
    t2=time.time()
    print 'time of skimage hog',t2-t1

    #----------Get HOG using a standard hog()----------
    t1=time.time()
    fd2,hog_image2,hists2=hog(img2,orientations=9,pixels_per_cell=(16,8),
            cells_per_block=(3,3),block_norm='l1',feature_vector=False,
            transform_sqrt=True,
            visualise=True)
    t2=time.time()
    print 'time of hog()',t2-t1

    #----------Get HOG using a vectorized hog(), compute indices-
    t1=time.time()
    fd3,indices=slideWindowHOG(img2,window_size=img2.shape[:2],orientations=9,
            pixels_per_cell=(16,8),
            stride=4,indices=None,
            return_indices=True,
            cells_per_block=(3,3),block_norm='l1',feature_vector=False,
            transform_sqrt=True)
    t2=time.time()
    print 'time of slideWindowHOG() on single window',t2-t1

    #----------Get HOG using a vectorized hog(), give pre-computed indices-
    t1=time.time()
    fd4=slideWindowHOG(img2,window_size=img2.shape[:2],orientations=9,
            pixels_per_cell=(16,8),
            stride=4,indices=indices,
            return_indices=False,
            cells_per_block=(3,3),block_norm='l1',feature_vector=False,
            transform_sqrt=True)
    t2=time.time()
    print 'time of slideWindowHOG() on single window with indices given',t2-t1

    #---------------Sliding window tests---------------
    window_size=(200,200)
    stride=4

    ny=(img2.shape[0]-window_size[0])//stride+1
    nx=(img2.shape[1]-window_size[1])//stride+1

    t1=time.time()
    for kk in range(ny):
        for jj in range(nx):
            boxjk=img2[kk*stride:kk*stride+window_size[0],\
                    jj*stride:jj*stride+window_size[1]]

            fdkj=skhog(boxjk,orientations=9,
                    pixels_per_cell=(16,8),
                    transform_sqrt=True,
                    feature_vector=True,
                    cells_per_block=(3,3),
                    visualise=False)
    t2=time.time()

    print 'time of native sliding window + skimage hog()', t2-t1

    t1=time.time()
    fd3=slideWindowHOG(img2,window_size=window_size,orientations=9,
            pixels_per_cell=(16,8),
            stride=4,indices=None,
            return_indices=False,
            cells_per_block=(3,3),block_norm='l1',feature_vector=False,
            transform_sqrt=True)
    t2=time.time()
    print 'time of slideWindowHOG() on sliding window',t2-t1
 
    """
    import cProfile
    cProfile.run('''fd3,indices=slideWindowHOG(img2,orientations=9,pixels_per_cell=(16,8),
            cells_per_block=(1,1),block_norm='l1',feature_vector=False,
            transform_sqrt=True)
    ''')


    #-----------------------Plot-----------------------
    import matplotlib.pyplot as plt
    figure=plt.figure(figsize=(12,10),dpi=100)

    ax1=figure.add_subplot(2,2,1)
    ax1.imshow(img2,cmap=plt.cm.gray)

    ax2=figure.add_subplot(2,2,2)
    ax2.imshow(hog_image,cmap=plt.cm.gray)

    ax3=figure.add_subplot(2,2,3)
    ax3.imshow(hog_image2,cmap=plt.cm.gray)

    plt.show(block=False)
    """




        

    

