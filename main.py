import cv2
import numpy as np
import math

print("package ok")

def stackImages(scale,imgArray):
    """
    Helper function to stack images horizontally and vertically for debugging.
    Allows viewing the Original, Mask, and Canny output in one window.
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def get_color_info(h, s, v):
    """
    Classifies the color of a shape based on HSV values and assigns a Priority Score.

    Args:
        h, s, v: Hue, Saturation, Value of the pixel.
    Returns:
        tuple: (ColorName, Score)
        - Red (Severe): Score 3
        - Yellow (Mild): Score 2
        - Green (Safe): Score 1
        - Camps (Pink/Blue/Grey): Score 0
    """

    #  Grey Camp Check
    if s < 40: return "Grey", 0


    # RED (Severe)
    if (h < 10 or h > 165):
        return "Red", 3 # Severe Status

    elif ( h > 125 ):
        return "Pink", 0  # Pink Camp

    # YELLOW (Mild)
    elif 20 < h < 35: return "Yellow", 2

    # GREEN (Safe)
    elif 35 < h < 85: return "Green", 1

    # BLUE (Camp)
    elif 90 < h < 115: return "Blue", 0

    return "Unknown", 0



#Function created for segmentation of land and sea by adding overlays on both the terrain
def segment_terrain(image_path):
    """
        Main pipeline to process the UAV image.
        Steps:
        1. Segment Land vs Ocean using HSV Masks.
        2. Detect Geometric Shapes (Survivors/Camps).
        3. Calculate Priority based on Shape and Color.
        4. Calculate Euclidean Distance between Survivors and Camps.
    """

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the path.")
        return

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Min and max HSV values tuned for creating sea mask - made a hsv_Detector file for getting these values
    lower_blue = np.array([77, 124, 0])
    upper_blue = np.array([179, 255, 255])

    # Min and max HSV values tuned for creating Land mask - made a hsv_Detector file for getting these values
    lower_land = np.array([0, 0, 0])
    upper_land = np.array([71, 255, 169])

    #MASKS CREATED ACC TO OUR TUNED HSV VALUES FOR LAND AND SEA
    mask_ocean = cv2.inRange(hsv_img, lower_blue, upper_blue)
    mask_land = cv2.inRange(hsv_img, lower_land, upper_land)

    #cleaning up the noise in the created mask so that unnessesary dots can be erased out

    kernel = np.ones((5, 5), np.uint8)

    #Smaller kernel for smaller black dots
    kernel2 = np.ones((2, 2), np.uint8)
    mask_ocean = cv2.morphologyEx(mask_ocean, cv2.MORPH_OPEN, kernel)
    mask_ocean = cv2.morphologyEx(mask_ocean, cv2.MORPH_CLOSE, kernel2)
    mask_land = cv2.morphologyEx(mask_land, cv2.MORPH_OPEN, kernel)

    output_img = img.copy()
    #overlays
    output_img[mask_ocean > 0] = [255, 0, 0]  # Pure Blue overlay
    output_img[mask_land > 0] = [0, 255, 255]  # Yellow overlay

    #creating a small shade of orinigal image on the mask
    alpha = 0.6  # Transparency factor
    final_output = cv2.addWeighted(output_img, alpha, img, 1 - alpha, 0)

    imgStack = stackImages(0.5 , ([img , final_output], [mask_ocean , mask_land]))

    cv2.imshow("Segmentation", imgStack)

    # Lists to store detected objects for distance calculation later
    camps = []
    casualties = []


    def getContours(img):

        """
        Extracts contours, classifies shapes, detects colors, and populates lists.
        """

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        casualty_id_counter = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(area)

            # # Area Filter: Ignore tiny noise (<100) and huge terrain boundaries
            if 100 < area < 1550:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)

                peri = cv2.arcLength(cnt, True)

                # Shape Approximation
                # peri is the perimeter. epsilon is the accuracy parameter.
                # Higher epsilon (0.05) ignores jagged pixel edges (noise).
                epsilon = 0.05 * peri
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                print(len(approx))

                objCor = len(approx)

                x, y, w, h = cv2.boundingRect(approx)

                objectType = "None"
                shape_score = 0

                # shape detection
                if objCor == 3:
                    objectType = "Triangle"
                    shape_score = 2

                elif objCor == 4:
                    objectType = "Square"
                    shape_score = 1

                #Use of Solidarity for detemining star and circles as edges detected for them were no constant and accurate
                elif objCor > 4:
                    # Calculate Solidity
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)

                    # Prevent division by zero
                    if hull_area == 0:
                        solidity = 0
                    else:
                        solidity = float(area) / hull_area

                    # A Circle is "fat" (High Solidity). A Star is "spiky" (Low Solidity).
                    if solidity > 0.90:
                        objectType = "Circle"
                    else:
                        objectType = "Star"
                        shape_score = 3


                #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(imgContour, objectType,(x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            #(0, 0, 0), 2)

            elif 1550 < area < 10000:
                """
                Same code but for detection of circles as there size is bigger 
                """

                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)


                peri = cv2.arcLength(cnt, True)
                epsilon = 0.05 * peri
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                print(len(approx))

                objCor = len(approx)

                x, y, w, h = cv2.boundingRect(approx)

                objectType = "None"
                shape_score = 0

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)

                # Prevent division by zero
                if hull_area == 0:
                    solidity = 0
                else:
                    solidity = float(area) / hull_area

                # A Circle is "fat" (High Solidity). A Star is "spiky" (Low Solidity).
                if solidity > 0.90:
                    objectType = "Circle"
                else:
                    objectType = "Star"
                    shape_score = 3

            if objectType != "None":
                cX = x + w // 2
                cY = y + h // 2

                # We pick the pixel at the exact center
                pixel = hsv_img[cY, cX]
                hue, sat, val = pixel[0], pixel[1], pixel[2]

                # Getting Status from our get_color_info function
                color_name, color_score = get_color_info(hue, sat, val)

                if objectType == "Circle":
                    # Store it in the camps list
                    camps.append({
                        'color': color_name,  # Pink, Blue, or Grey
                        'center': (cX, cY)
                    })
                    # Draw Camp Label
                    cv2.putText(imgContour, f"{color_name} Camp", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                                1)

                    # IF IT IS A CASUALTY (Person)
                else:
                    priority = shape_score * color_score
                    # Store it in the casualties list
                    casualties.append({
                        'id': casualty_id_counter,
                        'type': objectType,
                        'status': color_name,
                        'priority': priority,
                        'center': (cX, cY),
                        'box': (x, y, w, h),  # Store box for drawing later
                        'distances': {}  # Empty dictionary to store distances later
                    })
                    casualty_id_counter += 1

                    # Label: "Star Red"
                    label = f"{objectType} {color_name}"
                    cv2.putText(imgContour, label, (x, y - 5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

            # Draw and Label
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            total_priority = shape_score * color_score




            # Label: "P:9" (Only for casualties)
            if total_priority > 0:
                p_label = f"P:{total_priority}"
                cv2.putText(imgContour, p_label, (x, y + h + 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)



    imgContour = final_output.copy()
    shapeImg = final_output.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    blurred = cv2.bilateralFilter(imgGray, 9, 75, 75)
    imgCanny = cv2.Canny(blurred,50, 100)
    kernel = np.ones((3, 3), np.uint8)
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDial)

    print(f" Found {len(camps)} Camps and {len(casualties)} Casualties")

    for person in casualties:
        p_center = person['center']

        for camp in camps:
            c_center = camp['center']
            c_name = camp['color']

            # EUCLIDEAN DISTANCE FORMULA
            # dist = sqrt( (x2-x1)^2 + (y2-y1)^2 )
            dist = math.sqrt((c_center[0] - p_center[0]) ** 2 + (c_center[1] - p_center[1]) ** 2)

            # Storing this distance inside the person's dictionary
            person['distances'][c_name] = int(dist)

        # Printing for Debugging
        print(f"Person {person['id']} ({person['type']}) P:{person['priority']} Distances = {person['distances']}")

    '''
    for person in casualties:
        # Find the name of the closest camp just for visualization
        if person['distances']:
            closest_camp_color = min(person['distances'], key=person['distances'].get)

            # Find that camp's coordinates to draw a line
            for camp in camps:
                if camp['color'] == closest_camp_color:
                    start_pt = person['center']
                    end_pt = camp['center']
                    # Drawing a thin grey line to show we calculated it
                    cv2.line(imgContour, start_pt, end_pt, (150, 150, 150), 2)
                    
    '''
    camp_capacities = {
        'Pink': 3,
        'Blue': 4,
        'Grey': 2
    }

    casualties.sort(key=lambda x: x['priority'], reverse=True)

    assigned_count = 0
    total_priority_rescued = 0

    for person in casualties:
        sorted_distances = sorted(person['distances'].items(), key=lambda x: x[1])

        person_assigned = False

        # This way we try to assign to the nearest camp first
        for camp_color, dist in sorted_distances:

            # Checking if this camp has space
            if camp_capacities.get(camp_color, 0) > 0:
                # ASSIGN IT TO THEM
                person['assigned_to'] = camp_color
                camp_capacities[camp_color] -= 1  # Decrease capacity
                person_assigned = True

                # Update metrics
                assigned_count += 1
                total_priority_rescued += person['priority']

                print(
                    f"Success: Person {person['id']} ({person['type']}, P:{person['priority']}) = Assigned to {camp_color} Camp (Dist: {dist})")

                # Visuals: Draw line to the ASSIGNED camp
                target_camp = None
                for c in camps:
                    if c['color'] == camp_color:
                        target_camp = c
                cv2.line(imgContour, person['center'], target_camp['center'], (0, 255, 0), 2)

                break  # To stop checking other camps for this person

        if not person_assigned:
            print(f"Failure: Person {person['id']} could not be assigned (All camps are full)")
            person['assigned_to'] = "None"


    num_casualties = len(casualties)
    if num_casualties > 0:
        rescue_ratio = total_priority_rescued / num_casualties
    else:
        rescue_ratio = 0

    print(f"\n     FINAL METRICS     ")
    print(f"Total Priority Rescued: {total_priority_rescued}")
    print(f"Rescue Ratio (Pr): {rescue_ratio:.2f}")


    imgStack = stackImages(0.8, ([img, imgGray, blurred],
                                 [imgCanny, imgContour, imgDial]))
    cv2.imshow("Final Rescue Plan", imgStack)



    #imgBlank = np.zeros_like(img)


    '''
    cv2.imshow('Original', img)
    cv2.imshow('Ocean Mask', mask_ocean)
    cv2.imshow('Land Mask', mask_land)
    cv2.imshow('Segmented Overlay', final_output)'''

    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

segment_terrain('task_images/9.png')





