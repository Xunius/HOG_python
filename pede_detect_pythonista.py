'''Test pedestrian detection.

Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2018-02-22 15:24:46.
'''




#--------Import modules-------------------------
import os
import numpy
import hog_pythonista
import appex
from PIL import Image






def pyramid(image,scale=1.2,min_size=(30,30)):
    yield image
    scale=float(scale)

    while True:
        hh=int(image.size[0]/scale)
        ww=int(image.size[1]/scale)
        if hh<min_size[0] or ww<min_size[1]:
            break
        image=image.resize((hh,ww))
        yield image


def slideMatch2(img,stride,theta,intercept,params):
    box_size=params['box_size']
    orientations=params['orientations']
    pixels_per_cell=params['pixels_per_cell']
    cells_per_block=params['cells_per_block']
    block_norm=params['block_norm']
    transform_sqrt=params['transform_sqrt']

    scale=1.1
    pys=pyramid(img,scale=scale,min_size=box_size)

    result2=[]
    for ii,pii in enumerate(pys):
        pii=numpy.array(pii)
        hii,wii=pii.shape
        scaleii=scale**ii
        print('imgii',ii)

        nx=(wii-box_size[1])//stride+1
        ny=(hii-box_size[0])//stride+1
        print(ny,nx,box_size,stride,orientations,pixels_per_cell,cells_per_block,block_norm,transform_sqrt)
        features=hog_pythonista.slideWindowHOG(pii,window_size=box_size,
                stride=stride,
                indices=None,
                return_indices=False,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm=block_norm,
                transform_sqrt=transform_sqrt)

        print(features.shape)
        yhats=numpy.dot(features,theta[0])+intercept
        idx=numpy.where(yhats>=1.0)[0]
        idys,idxs=numpy.unravel_index(idx,(ny,nx))
        resultii=numpy.zeros([len(idx),5])
        resultii[:,0]=yhats[idx]
        resultii[:,1]=idxs*stride*scaleii
        resultii[:,2]=idys*stride*scaleii
        resultii[:,3]=(idxs*stride+box_size[1])*scaleii
        resultii[:,4]=(idys*stride+box_size[0])*scaleii

        result2.append(resultii)

    result2=numpy.concatenate(result2,axis=0)

    return result2



def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    # Malisiewicz et al.
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = numpy.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        #overlap = (w * h) / area[idxs[:last]]
        aa=numpy.minimum(area[i],area[idxs[:last]])
        overlap=(w*h)/aa

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last],
                numpy.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def non_max_suppression_fast2(boxes, overlapThresh, weight, final_threshold):
    # if there are no boxes, return an empty list
    # Malisiewicz et al.
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    pc = boxes[:,0]
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    #idxs = numpy.argsort(y2)
    idxs=numpy.argsort(pc)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        aa=numpy.minimum(area[i],area[idxs[:last]])
        #overlap = (w * h) / area[idxs[:last]]
        overlap = (w * h) / aa

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last],
                numpy.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    if not weight:
        results=boxes[pick][:,1:].astype("int")
    else:
        results=[]
        for i in pick:
            xx1 = numpy.maximum(x1[i], x1)
            yy1 = numpy.maximum(y1[i], y1)
            xx2 = numpy.minimum(x2[i], x2)
            yy2 = numpy.minimum(y2[i], y2)

            # compute the width and height of the bounding box
            w = numpy.maximum(0, xx2 - xx1 + 1)
            h = numpy.maximum(0, yy2 - yy1 + 1)
            aa=numpy.minimum(area[i],area)
            #overlap = (w * h) / area[idxs[:last]]
            overlap = (w * h) / aa

            bb=numpy.where(overlap>overlapThresh)[0]

            #-Omit detections with overlapping boxes less than-
            # threshold
            if len(bb)<final_threshold:
                continue

            x1mean=pc[bb].dot(x1[bb])/numpy.sum(pc[bb])
            x2mean=pc[bb].dot(x2[bb])/numpy.sum(pc[bb])
            y1mean=pc[bb].dot(y1[bb])/numpy.sum(pc[bb])
            y2mean=pc[bb].dot(y2[bb])/numpy.sum(pc[bb])

            boxii=numpy.array([x1mean,y1mean,x2mean,y2mean]).astype('int')
            results.append(boxii)

    return results


#-------------Main---------------------------------
if __name__=='__main__':

    #--------------------Read image--------------------
    if not appex.is_running_extension():
        print('Running in Pythonista app, using test image...')
        img=Image.open('test:Mandrill')
    else:
        img=appex.get_image()
        
    STRIDE=4
	
    HOG_PARAMETERS={
      'box_size'        : (128,64),
      'orientations'    : 9,
      'pixels_per_cell' : (8,8),
      'cells_per_block' : (2,2),
      'block_norm'      : 'l1',
      'transform_sqrt'  : True
    }

    img=img.convert('L')
    width2=min(400,img.size[1])
    height2=int(float(img.size[0])/img.size[1]*width2)
    img=img.resize((width2,height2))
    #img=numpy.array(img)
    print(img.size)

    #-----------------Load parameters-----------------
    folder=os.path.expanduser('~/Documents')
    abpath_in=os.path.join(folder,'testnpdata.npz')
    npdata=numpy.load(abpath_in)
    theta=npdata['theta']
    intercept=npdata['intercept']
    
    import time
    t1=time.time()
    rects=slideMatch2(img,STRIDE,theta,intercept,HOG_PARAMETERS)
    t2=time.time()
    print('time',t2-t1)
    print(rects)

    rects2=non_max_suppression_fast2(rects,0.35,True,15)
    #rects2=non_max_suppression_fast(rects,0.35)
    #rects2=numpy.array(rects)[:,1:]
    print(rects2)

    #-------------------Plot------------------------
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    figure=plt.figure(figsize=(12,10),dpi=100)
    ax=figure.add_subplot(111)

    img=numpy.array(img)
    ax.imshow(img)

    for ii, boxii in enumerate(rects2):
        x0,y0,x1,y1=boxii
        rectii=patches.Rectangle((x0,y0),x1-x0,y1-y0,fill=False)
        ax.add_patch(rectii)

    plt.show()

        

