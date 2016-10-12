import cv2
import numpy as np
import sys

def showImages():
    cv2.imshow('Optical Flow',frame)
    cv2.moveWindow('Optical Flow', 640, 330)
    cv2.imshow('Horizontal Component', horz)
    cv2.moveWindow('Horizontal Component', 0, 0)
    cv2.imshow('Vertical Component', vert)
    cv2.moveWindow('Vertical Component', 640, 0)
    cv2.imshow('Path of Car in relation to Camera Motion', car_path)
    cv2.moveWindow('Path of Car in relation to Camera Motion', 0, 330)

def printRatio():
    print("The speed of the car is ", ratio, "times the speed of the panning of the camera")

def printSpeeds():
	print("Car Speed:   ", speed_car, " m/s")
	print("Barn Speed:  ", speed_barn, " m/s")
	print("total speed: ", (speed_car + speed_barn), " m/s")

def calcDist(x1, y1, x2, y2):
    xHold = x1 - x2
    yHold = y1 - y2
    xSquare = xHold ** 2
    ySquare = yHold ** 2
    return (xSquare + ySquare) ** 0.5

# Create list of names here from .jpeg images
list_names = ['0000000' + str(i+1) + '.jpg' for i in range(9)]
list_names1 = ['000000' + str(i+10) + '.jpg' for i in range(90)]
list_names2 = ['00000' + str(i+100) + '.jpg' for i in range(153)]
list_names = list_names + list_names1
list_names = list_names + list_names2

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = .2,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create colors for tracking elements
color = np.array([[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]])
color_circle = np.array([[255,0,255],[255,0,255],[255,0,255],[255,0,255],[255,0,255]])

# Read in the first frame
frame1 = cv2.imread(list_names[0])
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#establish points to track
p0 = cv2.goodFeaturesToTrack(prvs, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(frame1)
#create frame to display car path
car_path = np.zeros_like(frame1)


counter = 1 			#keep track of frame number
headLightDist_old = 0;	#old distance betwen headlights
headLightDiff = 0;		#difference in distance between headlights
reFindPoints = False	#for reastablishing tracking points
barn_old = [0,0]		#old barn location
head1_old = [0,0]		#old headlight location
tree = True				#for first detection of trees in front of car
average_speed = [0,0]


#Explanation
print("BACKGROUND INFORMATION")
print("For the Horizontal and Verticle Components image boxes, white space indicates positive ")
print("movement, grey indicates negative movement, and black idicates no movement of individual pixels.") 
print("For the image box displaying the Path of the Car in relation to the Motion of the Camera, ")
print("it is important to note that the resulting line gets lighter with time. Furthermore, movement ")
print("to the right indicates the car is moving faster than the panning of the camera, movement to")
print("the left indicates the Camera is moving faster than the car and verticle movement is a result of the")
print("(x,y) location of the car in relation to the rest of the frame. \nEverything else if fairly self explanatory.")
print("This program will progress frame by frame giving certain readings after each frame.")
print("To move through the frames, click on one of the image boxes and press any key.")
print("Hold down a key for continous flow.\nEnjoy!\n\n")
#sys.stdout.flush()
cv2.waitKey(1000)

print("...skipping output of frame 1 to initialize internal values...\n")


# Until we reach the end of the list...
while counter < len(list_names):
    if counter != 1:
        print("\nFrame Number", counter)
    for i in range(5):
        color[i][0] += (counter/100)
        color[i][1] += (counter/20)

    # Read the next frame in
    frame2 = cv2.imread(list_names[counter])
    #convert colr frame to gray
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    #When tracking points are lost due to tree obstruction
    #need to refind tracking point locations
    if reFindPoints:
    	p0 = cv2.goodFeaturesToTrack(next, mask = None, **feature_params)
    	reFindPoints = False

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

    # Select good points from 3d array
    good_new = p1[st==1]
    good_old = p0[st==1]

    if len(good_new) == 1:
    	reFindPoints = True

    # Calculate optical flow between the two frames
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Normalize horizontal and vertical components
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')

  	#cycle through each detected point
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()	#(a,b) is location of detected point in new frame
        c,d = old.ravel()	#(c,d) is location of detected point in prev frame
        if i == 0:
            head2 = [a,b] 		#for when the barn point travels out of the image
            barn_new = [a,b]	#store barn location
        elif i == 1:
            head1 = [a,b]	#maintain headlight location
        elif i == 2:
            head2 = [a,b]
        if counter < 155 or counter > 174:
        	mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        	if i == len(good_new)-1:
        		car_path = cv2.line(car_path, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame2,(a,b),4,color_circle[i].tolist(),-1)

    headLightDist_new = calcDist(head1[0], head1[1], head2[0], head2[1])
    headLightDiff = headLightDist_new - headLightDist_old
    if counter > 1 and counter < 155:
    	print("Change in distance between headlights: ", headLightDiff, "pixels")
    if abs(headLightDiff) > 1 and tree:
        if counter != 1:
        	print("WATCH OUT THERE IS A TREE!")
        	tree = False;

    showImages()

    if counter < 121: #this is the last frame we can compare the barn and the car
    	barn_change = calcDist(barn_new[0], barn_new[1], barn_old[0], barn_old[1])
    	car_change =  calcDist(head1[0], head1[1], head1_old[0], head1_old[1])
    	ratio = abs(car_change / barn_change)
    	if counter > 1:
    		printRatio()
    	#assuming 100 frames per second
    	pix_velocity_car = car_change / .01  #pixels/second
    	pix_velocity_barn = barn_change / .01
    	#average distance between ehadlights for mid sized sedan = 5.5 ft = 167.64 cm in 2007
    	speed_car = pix_velocity_car * (1.6764 / headLightDist_new) #meters/second
    	speed_barn = pix_velocity_barn * (1.6764 / headLightDist_new)
    	if counter > 1:
    	    printSpeeds()
    	    average_speed[0] = average_speed[0] + speed_car + speed_barn
    	    average_speed[1] = average_speed[1] + 1

    if (counter > 121 and counter < 155) or counter > 174:
   	    car_change =  calcDist(head1[0], head1[1], head1_old[0], head1_old[1])
   	    print("Car has shifted by",car_change,"pixels")


    if counter > 1:
    	cv2.waitKey(0)

    barn_old = barn_new
    headLightDist_old = headLightDist_new
    head1_old = head1


    # Change - Make next frame previous frame
    prvs = next.copy()

    # Increment counter to go to next frame
    counter += 1

    # Now update the previous frame and previous points
    p0 = good_new.reshape(-1,1,2)

print("Average Speed of the Car was",average_speed[0]/average_speed[1],"mph")
cv2.destroyAllWindows()
