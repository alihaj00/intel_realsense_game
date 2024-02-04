import math
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import socket






import socket



def count_fingers(landmarks,i):

  # Check if hand landmarks are detected
  if landmarks:
    if landmarks[i]:
      hand_landmarks = landmarks[i]
      # Extract landmark positions
      landmarks = []
      for point in hand_landmarks.landmark:
        landmarks.append((point.x, point.y, point.z))

      # Count fingers based on landmark positions
      finger_count = 0

      # Thumb (check if x-coordinate of thumb tip is to the left of the x-coordinate of the thumb IP)
      if landmarks[4][0] < landmarks[3][0]:
        finger_count += 1

      # Index finger
      if landmarks[8][1] < landmarks[6][1]:
        finger_count += 1

      # Middle finger
      if landmarks[12][1] < landmarks[10][1]:
        finger_count += 1

      # Ring finger
      if landmarks[16][1] < landmarks[14][1]:
        finger_count += 1

      # Little finger
      if landmarks[20][1] < landmarks[18][1]:
        finger_count += 1

      return finger_count

  else:
    return 0  # No hand detected



def main():

  host, port = "127.0.0.1", 25001
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  print("waiting for connection to be made")
  try:
    sock.connect((host, port))
  except Exception as e:
    print("start printing")
    print(e)
    print("end printing")
  receivedData = sock.recv(1024).decode("UTF-8")  # receiveing data in Byte fron C#, and converting it to String
  print("connection been made")
  direction = [0, 0]  # Vector2   x = 0, y = 0

  ##et up the formatting for our OpenCV window that displays the output of our Hand Detection and Tracking
  font = cv2.FONT_HERSHEY_SIMPLEX
  org = (20, 100)
  fontScale =0.8
  color = (255, 0, 0)
  thickness = 1

  # ====== Realsense ======
  realsense_ctx = rs.context()
  connected_devices = []  # List of serial numbers for present cameras
  for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
  device = connected_devices[0]  # In this example we are only using one camera
  pipeline = rs.pipeline()
  config = rs.config()
  background_removed_color = 153  # Grey

  # ====== Mediapipe ======
  mpHands = mp.solutions.hands
  hands = mpHands.Hands()
  mpDraw = mp.solutions.drawing_utils





  # ====== Enable Streams ======
  config.enable_device(device)
  # # For worse FPS, but better resolution:
  # stream_res_x = 1280
  # stream_res_y = 720
  # # For better FPS. but worse resolution:
  stream_res_x = 640
  stream_res_y = 480
  stream_fps = 30
  config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
  config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
  profile = pipeline.start(config)
  align_to = rs.stream.color
  align = rs.align(align_to)

  # ====== Get depth Scale ======
  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")
  # ====== Set clipping distance ======
  clipping_distance_in_meters = 2
  clipping_distance = clipping_distance_in_meters / depth_scale
  print(f"\tConfiguration Successful for SN {device}")


  finger_count_right, finger_count_left = 0, 0
  is_right_hand_closed, is_left_hand_closed = 0, 0



  right_hand_depth=0
  left_hand_depth=0
  hand_diff=0
  bow_power=0
  y_angel=0
  angel=0
  x_left ,x_right=0,0
  xplace=0
  X_place=0

  leftHandOnScreen=0
  rightHandOnScreen=0
  y=0
  x=0
  while True:

    start_time = dt.datetime.today().timestamp()  # Necessary for FPS calculations
    # Get and align frames
    # Process images
    # Process hands
    # Display FPS
    # Display images
    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    if not aligned_depth_frame or not color_frame:
      continue


    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image, 1)
    color_image = np.asanyarray(color_frame.get_data())

    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                                  background_removed_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = cv2.flip(color_image, 1)
    color_image = cv2.flip(color_image, 1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(color_images_rgb)
    if results.multi_hand_landmarks:
      number_of_hands = len(results.multi_hand_landmarks)
      i = 0

      leftHandOnScreen=0
      rightHandOnScreen=0
      for handLms in results.multi_hand_landmarks:

        mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
        org2 = (20, org[1] + (20 * (i + 1)))
        hand_side_classification_list = results.multi_handedness[i]
        hand_side = hand_side_classification_list.classification[0].label

        if (number_of_hands == 1):
          if hand_side == "Left":
            leftHandOnScreen = 1
          if hand_side == "Right":
            rightHandOnScreen = 1
        if number_of_hands==2:
          rightHandOnScreen=1
          leftHandOnScreen=1

        #print(rightHandOnScreen)
        #print("\n")
        #print(leftHandOnScreen)

        # middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]

        wrist_place = results.multi_hand_landmarks[i].landmark[0]
        x = int(wrist_place.x * len(depth_image_flipped[0]))
        y = int(wrist_place.y * len(depth_image_flipped))
        if x >= len(depth_image_flipped[0]):
          x = len(depth_image_flipped[0]) - 1
        if y >= len(depth_image_flipped):
          y = len(depth_image_flipped) - 1

        if (x<0):
          x=0
        if x>640:
          x=640

        depth = aligned_depth_frame.get_distance(x, y)



        #print(f"x={dx},y={dy},z={dz}")
        if hand_side== "Right":

          finger_count_right = count_fingers(results.multi_hand_landmarks,i)

        if hand_side == "Left":

          finger_count_left = count_fingers(results.multi_hand_landmarks,i)

        mfk_distance = depth_image_flipped[y, x] * depth_scale  # meters
        mfk_distance_feet = mfk_distance * 3.281  # feet

        #images = cv2.putText(images,f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
        #images = cv2.putText(images, f"{hand_side} Hand Distance: x={dx} y={dy} z= {dz} ", org2, font, fontScale, color,thickness, cv2.LINE_AA)
        images = cv2.putText(images, f" Hand Distance:{hand_diff} ", org2, font, fontScale, color,thickness, cv2.LINE_AA)
        if (hand_side=="Right"):
          right_hand_depth= mfk_distance
          x_right = x
        if (hand_side=="Left"):
          left_hand_depth=mfk_distance
          x_left = x
        if (right_hand_depth>0 and left_hand_depth>0):
          hand_diff=(right_hand_depth-left_hand_depth)
        if (x_left>0 and x_right>0 and abs(x_right-x_left)<120):
          xplace=(x_left+x_right)/2
        i += 1

        if (hand_side=="Left" and y>0):
          y_angel=y


        images = cv2.putText(images, f"Hands: {number_of_hands}  rightisclosed={is_right_hand_closed}   leftisclosed={is_left_hand_closed}", org, font, fontScale, color, thickness, cv2.LINE_AA)
      else:
        images = cv2.putText(images, "No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)
      # Display FPS
      time_diff = dt.datetime.today().timestamp() - start_time
      fps = int(1 / time_diff)
      org3 = (20, org[1] + 60)
      images = cv2.putText(images, f"FPS: {fps}  angel = {angel}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
      name_of_window = 'SN: ' + str(device)


      # Display images

      cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
      cv2.imshow(name_of_window, images)
      key = cv2.waitKey(5)

      '''try to send to server but the server didnt take matrix it only take string need to fix that'''
      #    print(color_images_rgb)
      #   s.send(color_images_rgb)

      # Press esc or 'q' to close the image window
      if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break
    else:
      # Display images
      name_of_window = 'SN: ' + str(device)

      cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
      cv2.imshow(name_of_window, images)
      key = cv2.waitKey(5)
    is_left_hand_closed, is_right_hand_closed = 0, 0

    if (finger_count_left < 4):
      is_left_hand_closed = 1

    if (finger_count_right < 4):
      is_right_hand_closed = 1
    if leftHandOnScreen == 0:
      is_left_hand_closed = -1
    if rightHandOnScreen == 0:
      is_right_hand_closed = -1

    # print(f"right hand is closed{is_right_hand_closed}\n")
    # print(f"left hand is closed{is_left_hand_closed}")

    # checking the handdistance

    bow_power = int(1000 * (hand_diff / 0.3))
    if (bow_power > 1000):
      bow_power = 1000
    if (bow_power < 0):
      bow_power = 0


    angel = ((240 - y_angel) * 60 / 240)
    angel = round(angel, 2)
    if (angel < -30):
      angel = -50

    if (xplace > 0 and xplace < 640 and is_right_hand_closed!=-1 and is_left_hand_closed!=-1):
      X_place = -(340 - xplace) / 300
      X_place = int(X_place * 1000)
      if X_place < -1000:
        X_place = -1000
      if X_place > 1000:
        X_place = 1000

    angel = int(angel)
    send_data = str(angel) + "," + str(bow_power) + "," + str(int(is_right_hand_closed)) + "," + str(
      int(is_left_hand_closed)) + "," + str(int(X_place))
    print(f"data = {send_data} ")
    # angel_string = str(angel)
    sock.sendall(send_data.encode("UTF-8"))  # Converting string to Byte, and sending it to C#
    receivedData = sock.recv(1024).decode("UTF-8")  # receiveing data in Byte fron C#, and converting it to String
    print(receivedData)



  print(f"Exiting loop for SN: {device}")
  print(f"Application Closing.")
  pipeline.stop()
  print(f"Application Closed.")
  sock.close()

  '''note we have depth {depth}
  angel {angel}
  x,y,z{dx,dy,dz}
  hand closed or not {is_hand_closed(bool )}'''


if __name__ == "__main__":
  main()


