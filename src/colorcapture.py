#!/usr/bin/python3
'''
    colorcapture.py

    Capture L*a*b* color of a captured image.

    Note: Picam has a stripe on the right of the image! And it appears on our graph! 
    Graph is shrunk by 3px to remove it...
    https://www.raspberrypi.org/forums/viewtopic.php?t=227701
'''
import cv2
import time
import numpy as np
from scipy.signal import savgol_filter
import peakutils
import base64

def wavelength_to_rgb(nm):
    #from: https://www.codedrome.com/exploring-the-visible-spectrum-in-python/
    #returns RGB vals for a given wavelength
    gamma = 0.8
    max_intensity = 255
    factor = 0
    rgb = {"R": 0, "G": 0, "B": 0}
    if 380 <= nm <= 439:
        rgb["R"] = -(nm - 440) / (440 - 380)
        rgb["G"] = 0.0
        rgb["B"] = 1.0
    elif 440 <= nm <= 489:
        rgb["R"] = 0.0
        rgb["G"] = (nm - 440) / (490 - 440)
        rgb["B"] = 1.0
    elif 490 <= nm <= 509:
        rgb["R"] = 0.0
        rgb["G"] = 1.0
        rgb["B"] = -(nm - 510) / (510 - 490)
    elif 510 <= nm <= 579:
        rgb["R"] = (nm - 510) / (580 - 510)
        rgb["G"] = 1.0
        rgb["B"] = 0.0
    elif 580 <= nm <= 644:
        rgb["R"] = 1.0
        rgb["G"] = -(nm - 645) / (645 - 580)
        rgb["B"] = 0.0
    elif 645 <= nm <= 780:
        rgb["R"] = 1.0
        rgb["G"] = 0.0
        rgb["B"] = 0.0

    if 380 <= nm <= 419:
        factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
    elif 420 <= nm <= 700:
        factor = 1.0
    elif 701 <= nm <= 780:
        factor = 0.3 + 0.7 * (780 - nm) / (780 - 700)

    if rgb["R"] > 0:
        rgb["R"] = int(max_intensity * ((rgb["R"] * factor) ** gamma))
    else:
        rgb["R"] = 0

    if rgb["G"] > 0:
        rgb["G"] = int(max_intensity * ((rgb["G"] * factor) ** gamma))
    else:
        rgb["G"] = 0

    if rgb["B"] > 0:
        rgb["B"] = int(max_intensity * ((rgb["B"] * factor) ** gamma))
    else:
        rgb["B"] = 0

    return (rgb["R"], rgb["G"], rgb["B"])

def snapshot():
    # Get a frame from the graph, and write it to disk
    ret, graphdata = get_graph()
    print("wavelenghts", graphdata[1]) #wavelengths
    print("values", graphdata[2]) #intensities

    if ret:
        # store the spectrum as image
        now = time.strftime("%d-%m-%Y-%H:%M:%S")
        imgname = "FF-spectrum-" + now + ".jpg"
        cv2.imwrite(imgname, cv2.cvtColor(graphdata[0], cv2.COLOR_RGB2BGR))
        print("Written image", imgname)
        # store the spectrum values
        fname = 'FF-' + now + '.csv'
        f = open(fname ,'w')
        f.write('Wavelength,Intensity\r\n')
        for x in zip(graphdata[1], graphdata[2]):
            f.write(str(x[0])+','+str(x[1])+'\r\n')
        f.close()
        print("Written data file", fname)				

# return a success flag and 320x240 RGB captured frame
def get_frame():
    global vid
    if vid.isOpened():
        ret, frame = vid.read()
        if ret:
            # Return a boolean success flag and the current frame converted to BGR
            frame = cv2.resize(frame, (320, 240))                # resize the image
            cv2.line(frame,(0,120),(320,120),(255,255,255),1)    # draw an horizontal line in the middle
            return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # convert from BGR to RGB
        else:
            return (ret, None)
    else:
        return (ret, None)

'''
lasers are 532nm and 650nm
calibration ((604, 650), (391, 532))
pxrange 213 nmrange 118
pxpermn 1.805084745762712 nmperpx 0.5539906103286385
'''

def get_graph():
    global vid
    intensity = np.zeros([636], dtype=np.int32) #[0] * 636 #array for intensity data...full of zeroes
    mindist = 50 #minumum distance between peaks    
    thresh = 20 #Threshold

    if not vid.isOpened():
        print("in get_graph. Video not opened.")
        return (ret, None)

    ret, frame = vid.read()
    if not ret:
        print("in get_graph. Error reading a frame.")
        return (ret, None)

    #Process the data...
    #Why 636 pixels? see notes on picam at beginning of file!
    piwidth = 636
    image = frame
    bwimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows,cols = bwimage.shape
    #create a blank image for the data
    graph = np.zeros([255,piwidth,3], dtype=np.uint8)
    graph.fill(255) #fill white

    # Display  vertical lines 
    #calculate the ranges
    pxrange = 213 #abs(calibration[0][0] - calibration[1][0]) #how many px between points 1 and 2?
    nmrange = 118 #abs(calibration[0][1] - calibration[1][1]) #how many nm between points 1 and 2?
    #how many pixels per nm?
    pxpernm = pxrange/nmrange   #1.805084745762712
    #how many nm per pixel?
    nmperpx = nmrange/pxrange   #0.5539906103286385
    #how many nm is zero on our axis
    zero = 315.3896 #calibration[0][1] - (calibration[0][0]/pxpernm)
    scalezero = zero #we need this unchanged duplicate of zero for later!
    prevposition = 0
    textoffset = 12
    font = cv2.FONT_HERSHEY_SIMPLEX

    #vertical lines
    for i in range(piwidth):
        position = round(zero)
        if position != prevposition: #because of rounding, sometimes we draw twice. Lets fix tht!
            # we could have grey lines for subdivisions???S
            if position%10==0:
                cv2.line(graph,(i,15),(i,255),(200,200,200),1)
            if position%50==0:
                cv2.line(graph,(i,15),(i,255),(0,0,0),1)
                cv2.putText(graph,str(position)+'nm',(i-textoffset,12),font,0.4,(0,0,0),1, cv2.LINE_AA)
        zero += nmperpx
        prevposition = position
    #horizontal lines
    for i in range (255):
        if i!=0 and i%51==0: #suppress the first line then draw the rest...
            cv2.line(graph,(0,i),(piwidth,i),(100,100,100),1)
    
    #now process the data...
    halfway = int(rows/2) #halfway point to select a row of pixels from
    
    #pull out single row of data and store in a self.intensity array
    #Why -4 pixels? see notes on picam at beginning of file!
    for i in range(cols-4):
        data = bwimage[halfway, i]
        intensity[i] = data

        #if self.holdpeaks == True:
        #    if data > intensity[i]:
        #        intensity[i] = data
        #else:
        #   self.intensity[i] = data

    #if self.holdpeaks == False:
    #    #do a little smoothing of the data
    #    self.intensity = savgol_filter(self.intensity,17,int(self.savpoly))
    #intensity = intensity.astype(int)

    #now draw the graph
    #for each index, plot a verital line derived from int
    #use waveleng_to_rgb to false color the data.
    wavelengthdata = []
    index=0
    for i in intensity:
        wavelength = (scalezero+(index/pxpernm))
        wldata = round(wavelength,1)
        wavelength = round(wavelength)
        wavelengthdata.append(wldata)
        rgb = wavelength_to_rgb(wavelength)
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        #(start x,y) (end x,y) (color) thickness
        #origin is top left.
        cv2.line(graph, (index,255), (index,255-i), (r,g,b), 1)
        cv2.line(graph, (index,254-i), (index,255-i), (0,0,0), 1,cv2.LINE_AA)
        index+=1

    #find peaks and label them
    thresh = int(thresh) #make sure the data is int.
    indexes = peakutils.indexes(intensity, thres=thresh/max(intensity), min_dist=mindist)
    for i in indexes:
        height = intensity[i]
        height = 245 - height
        wavelength = int(scalezero+(i/pxpernm))
        cv2.rectangle(graph,((i-textoffset)-2,height+3),((i-textoffset)+45,height-11),(255,255,0),-1)
        cv2.rectangle(graph,((i-textoffset)-2,height+3),((i-textoffset)+45,height-11),(0,0,0),1)
        cv2.putText(graph,str(wavelength)+'nm',(i-textoffset,height),font,0.4,(0,0,0),1, cv2.LINE_AA)

    #################################################################
    graphdata = []
    graphdata.append(graph)
    graphdata.append(wavelengthdata)
    graphdata.append(intensity)
    return (ret, graphdata)

def main():
    global vid
    # open the video camera for capture
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    vid.set(cv2.CAP_PROP_FPS, 25)
    if not vid.isOpened():
        print ("Error in opening camera. Exiting...")
        exit()
    
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Capture ({height} x {width})")

    ret, frame = get_frame()
    if not ret:
        print("Error in capture the frame")
    else:
        print("Frame captured", frame.shape)

    cv2.imshow('Captured image', frame)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    snapshot()
if __name__=="__main__":
    main()