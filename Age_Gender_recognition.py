import cam as cam
import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="data/gender_age/opencv_face_detector.pbtxt"
faceModel="data/gender_age/opencv_face_detector_uint8.pb"
ageProto="data/gender_age/age_deploy.prototxt"
ageModel="data/gender_age/age_net.caffemodel"
genderProto="data/gender_age/gender_deploy.prototxt"
genderModel="data/gender_age/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-3)', '(4-7)', '(8-14)', '(15-20)','(21-24)' ,'(25-32)', '(33-37)', '(38-45)','(46-52)','(53-60)','(61-70)','(70-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

from cam_setup import cam

padding=20


global det_age, det_gen
det_age = []
det_gen = []


def result():
    while True:
        hasFrame, frame = cam.read()
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                        :min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            gen = gender
            # print(gen)
            det_gen.append(gen)

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
            ag = age[1:-1]
            # print(ag)
            det_age.append(ag)




            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Detecting age and gender", resultImg)
        k=cv2.waitKey(10)  # & 0xff  # Press 'ESC' for exiting cam
        if k == 27:
            break
        # cv2.waitKey(10)
        # break
        # import output1

    cam.release()
    cv2.destroyAllWindows()

result()





