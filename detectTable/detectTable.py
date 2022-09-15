import cv2
import numpy as np
from PIL import Image
import pandas as pd
def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] -
                    thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                    line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x) - int(5*thin_thresh), max(x) + int(5*thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

def group_v_lines(v_lines, thin_thresh):
        new_v_lines = []
        while len(v_lines) > 0:
            thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
            lines = [line for line in v_lines if thresh[0] -
                    thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
            v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                    line[0][0] or line[0][0] > thresh[0] + thin_thresh]
            y = []
            for line in lines:
                y.append(line[0][1])
                y.append(line[0][3])
            y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
            new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
        return new_v_lines
def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b
def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1   
def get_bottom_right(right_points, bottom_points, points):
    for right in right_points:
        for bottom in bottom_points:
            for i in range (-1,2):
                if [right[0] + i, bottom[1]] in points:
                    return right[0] + i, bottom[1]
                if [right[0], bottom[1] + i] in points:
                    return right[0], bottom[1] + i
    return None, None
def detectTable(img):
    kernel = np.array([[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel) # applying the sharpening kernel to the input image.

    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #inverting the image 
    img_bin = 255-img_bin

    kernel_len = np.array(img).shape[1]//110
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)
    h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 30, maxLineGap=250)
    new_horizontal_lines = group_h_lines(h_lines, kernel_len)

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)
    v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 30, maxLineGap=250)
    new_vertical_lines = group_v_lines(v_lines, kernel_len)
    
    points = []
    for hline in new_horizontal_lines:
        x1A, y1A, x2A, y2A = hline
        for vline in new_vertical_lines:
            x1B, y1B, x2B, y2B = vline

            line1 = [np.array([x1A, y1A]), np.array([x2A, y2A])]
            line2 = [np.array([x1B, y1B]), np.array([x2B, y2B])]

            x, y = seg_intersect(line1, line2)
            if x1A <= x <= x2A and y1B <= y <= y2B:
                points.append([int(x), int(y)])
    cells = []
    for point in points:
        left, top = point
        right_points = sorted(
            [p for p in points if p[0] > left and (top-1 <= p[1] <=top+1)], key=lambda x: x[0])
        bottom_points = sorted(
            [p for p in points if p[1] > top and (left-1 <= p[0] <= left+1)], key=lambda x: x[1])

        right, bottom = get_bottom_right(
            right_points, bottom_points, points)
        if right and bottom:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cells.append([left, top, right, bottom])
    return img, cells
def tableRestruction(cells):
    row=[]
    column=[]
    j=0
    #Sorting the cells to their respective row and column
    for i in range(len(cells)):         
        if(i==0):
            column.append(cells[i])
            previous=cells[i]    
        
        else:
            if(previous[1]-1<=cells[i][1]<=previous[1]+1):
                column.append(cells[i])
                previous=cells[i]            
                
                if(i==len(cells)-1):
                    row.append(column)        
                
            else:
                row.append(column)
                column=[]
                previous = cells[i]
                column.append(cells[i])
    countcol = 0
    for i in range(len(row)):
        if len(row[i]) > countcol:
            countcol = len(row[i])

        #Retrieving the center of each column
    left_p = [int(row[i][j][0]) for j in range(len(row[i]))]

    left_p=np.array(left_p)
    left_p.sort()
    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(left_p-row[i][j][0])
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
    return finalboxes, len(row), countcol
def extractTable(img, detector):
    img2, cells = detectTable(img)
    # cv2.imshow('a', img2)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 
    finalboxes, no_row, no_collum = tableRestruction(cells)
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=''
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    x1,y1,x2,y2 = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = img[y1-3:y2+3, x1-3:x2+3]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=2)
                    # cv2.imshow("window_name", erosion)
                    # cv2.waitKey(0) 
                    # #closing all open windows 
                    # cv2.destroyAllWindows() 
                    finalimg = Image.fromarray(np.uint8(erosion))
                    out = detector.predict(finalimg)
                    print(out)
                    inner = inner +" "+ out
                outer.append(inner)
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(no_row, no_collum))
    data = dataframe.style.set_properties(align="left")
    return data