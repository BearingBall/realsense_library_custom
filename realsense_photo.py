import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pickle

def takePhoto(fileName):
    pipeline = rs.pipeline()
    config = rs.config()
    
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    #print("Depth Scale is: " , depth_scale)
    
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    i=0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
    
            if not aligned_depth_frame or not color_frame:
                continue
    
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
    
            # Remove background - Set pixels further than clipping_distance to grey
            #grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), color_image, color_image)
            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            if key == 32:
                file = open(fileName+"_"+str(i)+".configs", "wb")
                configs = [bg_removed.shape[0], bg_removed.shape[1] , bg_removed.shape[2], depth_image.shape[0], depth_image.shape[1 ], depth_scale]
                pickle.dump(configs, file)
                file.close
                file = open(fileName+"_"+str(i)+".color", "wb")
                pickle.dump(bg_removed, file)
                file.close()
                file = open(fileName+"_"+str(i)+".depth", "wb")
                pickle.dump(depth_image, file)
                file.close()
                print("taked "+str(i))
                i+=1

    finally:
        pipeline.stop()

        
def showPhoto(fileName):
    i=0
    isRefresh = True
    point = np.zeros(4)
    currentLabel = 0
    labels = ["floor" , "roof", "wall", "door", "obstacle"]
    while True:
        if isRefresh:
            isRefresh = False
            file = open(fileName+"_"+str(i)+".configs", "rb")
            configs = pickle.load(file)
            file.close()
            file = open(fileName+"_"+str(i)+".color", "rb")
            bg_removed = pickle.load(file)
            file.close()
            file = open(fileName+"_"+str(i)+".depth", "rb")
            depth_image = pickle.load(file)
            file.close()
            
            depth_scale = configs[5]
            clipping_distance_in_meters = 1 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale
    
            bg_removed = bg_removed.reshape((int(configs[0]),int(configs[1]),int(configs[2])))
            depth_image = depth_image.reshape((int(configs[3]),int(configs[4]),1))
    
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            images = images.astype(np.uint8)
        
        def draw_circle(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point[0] = x
                point[1] = y
            if event == cv2.EVENT_LBUTTONUP:
                point[2] = x
                point[3] = y
                tmp = np.zeros(5)
                tmp[0] = point[0]
                tmp[1] = point[1]
                tmp[2] = point[2]
                tmp[3] = point[3]
                tmp[4] = currentLabel
                param.append(tmp)
                
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Align Example',draw_circle, configs)
        
        images = np.hstack((bg_removed, depth_colormap))
        images = images.astype(np.uint8)
        for j in range(6, len(configs)):
            cv2.rectangle(images,(int(configs[j][0]),int(configs[j][1])),(int(configs[j][2]),int(configs[j][3])), (255,0,0),1)
            cv2.putText(images, labels[int(configs[j][4])] , (int(min(configs[j][0],configs[j][2] )),int(min(configs[j][1],configs[j][3]))+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        for j in range(len(labels)):
            cv2.putText(images, labels[j] , (650, 40+j*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 if j == currentLabel else 255, 0, 255),2)
        cv2.putText(images, fileName+"_"+str(i) , (900, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        cv2.imshow('Align Example',  images)
        
        key = cv2.waitKey(1)
        #if (not key == -1):
         #   print(key)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            file = open(fileName+"_"+str(i)+".configs", "wb")
            pickle.dump(configs, file)
            file.close()
            break
        if (key == 46)and(os.path.isfile(fileName+"_"+str(i+1)+".configs")):
            file = open(fileName+"_"+str(i)+".configs", "wb")
            pickle.dump(configs, file)
            file.close()
            i+=1
            isRefresh = True
        if (key == 44)and(os.path.isfile(fileName+"_"+str(i-1)+".configs")):
            file = open(fileName+"_"+str(i)+".configs", "wb")
            pickle.dump(configs, file)
            file.close()
            i-=1
            isRefresh = True
        if (key < 54 and key > 48):
            currentLabel = key-49
        if (key == 122 and len(configs) >6):
            configs.pop()