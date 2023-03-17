# import files
import cv2

# use for only cascadePath
cascadePath = "data/face_data/haarcascade_frontalface_default.xml"

# cv2.CAP_DSHOW to remove warning  # create a video capture object which is helpful to capture videos through webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# this code use for only ip camera
webcam_ip = "192.168.1.5:8080" # Setup Your ip camera ip.
web_cam_address = f"https://{webcam_ip}/video"
cam.open(web_cam_address)