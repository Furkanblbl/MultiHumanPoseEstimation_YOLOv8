from ultralytics import YOLO
import cv2
import numpy as np
import time



class PoseEstimationModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_people(self, frame):
        """
        input: every frame
        output: object of YOLO that detect people
        """
        results = self.model.track(frame, persist=True) # Detect people using YOLO pattern
        return results 
    

    @staticmethod
    def get_person_boxes_and_keypoints(frame, results):
        """
        Returns the tiles and keypoints of each detected person

        input: Yolo object
        outputs: 
            persons_boxes : Keeps the box values of each person
            persons_keypoints_cordinates: coordinates of the joint areas of each person
        
        """
        persons_boxes = [] # for 2 person person_boxes= [[],[]]
        persons_keypoints_cordinates = []


        for result in results:
            kpts = result.keypoints
            bxs = result.boxes
        
            for keypoint, box in zip(kpts, bxs):
                #keypoint, A class for storing and manipulating detection keypoints.
                #box, A class for storing and manipulating detection boxes.
                box.is_track = True # To open box.ids, is track = True.
                if box.conf[0]>0.5:
    
                    height = box.xywh[0][3]
                    widht = box.xywh[0][2]
                    oran = height / widht
                    persons_boxes.append(box)

                    cv2.rectangle(frame,(int(box.xyxy[0][0]),int(box.xyxy[0][1])),( int(box.xyxy[0][2]),int(box.xyxy[0][3])), (0,0,255), 2, cv2.LINE_AA)
                    
                    
                    cordinates = keypoint.xy # coordinates of the joint areas of a person 
                    persons_keypoints_cordinates.append(cordinates)

                    for cordinate in cordinates: 
                        for tmp in range(len(cordinate)): # len(cordinate) is the 17 keypoints of joint areas
                            cv2.circle(frame, (int(cordinate[tmp][0]), int(cordinate[tmp][1])), 5, (0, 255, 0), -1) #draw keypoints for a person

        return persons_boxes, persons_keypoints_cordinates

        

class Person:
    def __init__(self, person_id):
        self.person_id = person_id
        self.prev_time = 0
        self.current_time = time.time()
        self.currentState = "other"
        self.status = {"squatting":0, "lie down":0, "running":0, "other":0}
        self.prev_box = None


class ActionAnalyzer:
    def __init__(self):
        pass

    def analyze_action(self, person:Person, person_keypoints_cordinates, boxes ):
        """
        detects the situation a person is in
        inputs:
            person:Person : Person class object
            person_keypoints_cordinates :  coordinates of the joint areas of each person
            boxes : ultralytics.engine.results.Boxes object
        output:

        """
        for cordinate, box in zip(person_keypoints_cordinates, boxes):
            height = box.xywh[0][3]
            width = box.xywh[0][2]
            ration = height / width
            
            """
            https://alimustoofaa.medium.com/yolov8-pose-estimation-and-pose-keypoint-classification-using-neural-net-pytorch-98469b924525
            for example :
                cordinate[6] is right shoulder.
                cordinate[6][0] is right shoulder's x axis.
            """
            self.right_knee_angle = ActionAnalyzer.calculate_angle(int(cordinate[12][0]), int(cordinate[12][1]), int(cordinate[14][0]), int(cordinate[14][1]), int(cordinate[16][0]), int(cordinate[16][1])) 
            self.left_knee_angle = ActionAnalyzer.calculate_angle(int(cordinate[11][0]), int(cordinate[11][1]), int(cordinate[13][0]), int(cordinate[13][1]), int(cordinate[15][0]),int(cordinate[15][1]))
            self.right_hip_angle = ActionAnalyzer.calculate_angle(int(cordinate[6][0]), int(cordinate[6][1]), int(cordinate[12][0]), int(cordinate[12][1]), int(cordinate[14][0]),int(cordinate[14][1]))
            
            self.right_elbow_angle = ActionAnalyzer.calculate_angle(int(cordinate[6][0]), int(cordinate[6][1]), int(cordinate[8][0]), int(cordinate[8][1]), int(cordinate[10][0]),int(cordinate[10][1]))
            self.left_elbow_angle = ActionAnalyzer.calculate_angle(int(cordinate[5][0]), int(cordinate[5][1]), int(cordinate[7][0]), int(cordinate[7][1]), int(cordinate[9][0]),int(cordinate[9][1]))

            status =  ""
            time_ = time.time()

            if person.prev_time == 0:
                person.prev_time = person.current_time
            
            if (self.left_knee_angle<110) and (self.right_knee_angle<110) and (self.right_hip_angle<95) :  #cokme pozisyonu
                status = "squatting"
                
            elif (abs(int(cordinate[5][1])-int(cordinate[11][1])) < 60) and (ration < 1.0) :        
                status = "lie down"
           
            elif(self.left_elbow_angle < 130 or self.right_elbow_angle < 130  ) and ration > 2.0 and (self.left_knee_angle < 150 or self.right_knee_angle < 150):
                status = "running"
            
            else:
                status = "other"

            
            if person.currentState != status:
                person.status[person.currentState] += person.current_time - person.prev_time 
                person.prev_time = person.current_time
            # else:
            #     person.status[person.currentState] = y - x

            person.currentState = status
            return time_
        
        

    @staticmethod
    def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y):  # 11 13 15 points shoulder elbow wrist
        """
        Calculating the angle between given joints
        input:
            a_x, a_y = first joint axis
            b_x, b_y = middle joint axis
            c_x, c_y = third joint axis
        
        output :  angle 
        """
        a_x = np.array(a_x) # first
        a_y = np.array(a_y)
        b_x = np.array(b_x) # mid
        b_y = np.array(b_y)
        c_x = np.array(c_x) # end points 
        c_y = np.array(c_y)

        radians = np.arctan2(c_y-b_y, c_x-b_x) - np.arctan2(a_y-b_y, a_x-b_x) 
        angle = np.abs(radians*180.0/np.pi)  # calculate our radians for pur particular joint and angle 
        
        if angle > 180.0:
            angle = 360 - angle

        return angle



class VideoProcessor:
    def __init__(self, model_path, video_source):
        self.pose_model = PoseEstimationModel(model_path)
        self.video_source = video_source
        self.persons = {} 

    def process_video(self):
  
        cap = cv2.VideoCapture(self.video_source) # play video
        action_analyzer = ActionAnalyzer() #object of ActionAnalyzer class

        codec = cv2.VideoWriter_fourcc(*'XVID')
        frame_rate = 30.0  
        video_boyutu = (int(cap.get(3)), int(cap.get(4))) 
        video_kaydedici = cv2.VideoWriter('kaydedilenN_video.avi', codec, frame_rate, video_boyutu)

        while True:
            ret, frame = cap.read() # read video as frame by frame

            if not ret:
                break

            # Detect people with YOLO pattern
            results = self.pose_model.detect_people(frame)


            # Take the boxes belonging to the people on the frame
            persons_boxes,  persons_keypoints_cordinates = self.pose_model.get_person_boxes_and_keypoints(frame, results)

            
            for box, person_keypoints_cordinates in zip(persons_boxes,persons_keypoints_cordinates):
                person_id = box.id.item() # assign id to each person
                if person_id not in self.persons:
                    self.persons[person_id] = Person(person_id) # create Person object and store in dict
                    print(self.persons)

                person = self.persons[person_id] #return <__main__.Person object at 0x7fcdb1965040>
    
                # Analyze action
                action = action_analyzer.analyze_action(person, person_keypoints_cordinates, box)
                person.current_time = action 

                cv2.putText(
                    frame,
                    (f"Person {person_id}: {person.status} for {person.status[person.currentState]} seconds."),
                    (int(box.xyxy[0][0]), int(box.xyxy[0][1] -20 )),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )

                print(f"Person {person_id}: {person.status}")
           

            video_kaydedici.write(frame)
            # If you want to show the image you can print it on the screen
            cv2.imshow('YOLO Output', frame)


            # 'End the loop by pressing the 'q' key
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        # Cleanup operations after video processing is finished
        cap.release()
        video_kaydedici.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    model_path = "model/yolov8n-pose.pt"
    video_source = "videos/media4.mp4"

    video_processor = VideoProcessor(model_path, video_source)
    video_processor.process_video()




























