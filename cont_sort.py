#!/usr/bin/env python
# coding: utf-8

# In[ ]:


im=cv2.imread(r'tablesample.png')
img=im.copy()
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
def draw_contour(img, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the countour number on the image
    cv2.putText(img, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 155, 255), 2)
    # return the image with the contour number drawn on it
    return img
def extr_table(img):
    im1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh_value = cv2.threshold(im1,150,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((1,1),np.uint8)
    dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, boundingbox)= sort_contours(contours, method="top-to-bottom")
    i=0
    cont=[]
    for (a,c) in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        if  h>50: 
        # loop over the (now sorted) contours and draw them
            draw_contour(img, c, i)#draw the contour number for each contour
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,150,255),2)
            cont.append([x,y,w,h])
            i+=1
    for i in range(len(cont)-1,-1,-1):#iterate through obtained contours to get each region
        ROI = img[cont[i][1]:cont[i][1]+cont[i][3],cont[i][0]:cont[i][0]+cont[i][2]]#im[y:y+h, x:x+w]
        ROI=cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        ROI=cv2.adaptiveThreshold(ROI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,1001,11)
        kernel1=np.ones((1,80), np.uint8)
        kernel2=np.ones((80,1), np.uint8)
        morphed_vertical=cv2.morphologyEx(ROI, cv2.MORPH_CLOSE, kernel1)
        morphed_horizontal=cv2.morphologyEx(ROI, cv2.MORPH_CLOSE, kernel2)
        dst=cv2.add(ROI,(255-morphed_vertical))
        new_dst=cv2.add(dst,(255-morphed_horizontal))
        roi=pytesseract.image_to_string(new_dst,lang="eng",config="--psm 6").replace('\n','')  
        roi=roi.lower()
        
    return img
extr_table(img)

