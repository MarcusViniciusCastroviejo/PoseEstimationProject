import cv2
import numpy as np
import time
import mediapipe as mp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class PoseDetector:

    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        # Initial parameters
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Mediapipe parameters
        self.mpPose = mp.solutions.pose
        self.results = None
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     True, self.smooth_segmentation,
                                     self.min_detection_confidence, self.min_tracking_confidence)

        # Position
        self.landmarks = []
        """ [0] Head
            [1] left shoulder       [2] right shoulder      [3] left elbow      [4] right elbow
            [5] left hand           [6] right hand          [7] left hip        [8] right hip
            [9] left knee           [10] right knee         [11] left ankle     [12] right ankle
            [13] left heel          [14] right heel         [15] left foot      [16] right foot
            [17] hip center         [18] neck center
        """
        self.angles = []
        self.labelAngles = ["Knee", "Ankle", "Hip", "Spine"]
        self.poseConnections = [(1, 3), (3, 5), (2, 4), (4, 6),
                                (1, 7), (7, 9), (9, 11), (11, 13),
                                (11, 15), (13, 15), (2, 8), (8, 10),
                                (10, 12), (12, 14), (12, 16), (14, 16),
                                (1, 2), (7, 8), (17, 18), (18, 0)]

        # Parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontSize = None
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        self.gray = (200, 200, 200)

        # Dimension
        self.headRadius = None
        self.factor = None

        # Time
        self.init = True
        self.initTime = time.time()
        self.prevTime = self.initTime
        self.curTime = None
        self.realDuration = None
        self.appDuration = None
        self.Time = []

        # Kinematics
        self.dt = None
        self.prevLm = None
        self.position = []
        self.vel = []
        self.vel_prev = [0, 0]
        self.acc = []
        self.fps = 0

    # ----------------------------------------------- Main Functions ------------------------------------------------ #
    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lms = PoseDetector.findLandmarks(self, img)
        return lms

    def findLandmarks(self, img):
        ear = []
        height, width, _ = img.shape

        if self.results.pose_landmarks:
            self.prevLm = self.landmarks.copy()
            self.landmarks.clear()
            self.landmarks.append((0, 0))
            for i, lm in enumerate(self.results.pose_landmarks.landmark):
                if i in [7, 8]:
                    ear.append((round(lm.x * width), round(lm.y * height)))
                if (11 <= i <= 16) or (23 <= i <= 32):
                    self.landmarks.append((round(lm.x * width), round(lm.y * height)))

            self.landmarks[0] = (round(0.5 * (ear[0][0] + ear[1][0])), round(0.5 * (ear[0][1] + ear[1][1])))
            self.landmarks.append((round(0.5 * (self.landmarks[7][0] + self.landmarks[8][0])),    # Hip center (17)
                                   round(0.5 * (self.landmarks[7][1] + self.landmarks[8][1]))))
            self.landmarks.append((round(0.5 * (self.landmarks[1][0] + self.landmarks[2][0])),      # Neck center (18)
                                   round(0.5 * (self.landmarks[1][1] + self.landmarks[2][1]))))

            if self.init:
                self.init = False
                self.prevLm = self.landmarks.copy()
                PoseDetector.autoSize(self, img)

        return self.landmarks

    def drawPosition(self, img):
        for pair in self.poseConnections:
            x1, y1 = self.landmarks[pair[0]]
            x2, y2 = self.landmarks[pair[1]]
            pt1, pt2 = (x1, y1), (x2, y2)

            if pair == (17, 18) or pair == (18, 0):
                cv2.line(img, pt1, pt2, self.green, 2, cv2.LINE_AA)
            else:
                cv2.line(img, pt1, pt2, self.green, 4, cv2.LINE_AA)

        for i, lm in enumerate(self.landmarks):
            if i == 0:
                cv2.circle(img, (lm[0], lm[1]), self.headRadius, self.green, cv2.FILLED)
            elif i in [17, 18]:
                pass
            else:
                cv2.circle(img, (lm[0], lm[1]), 10, self.green, cv2.FILLED)
                if i % 2 == 0:
                    cv2.circle(img, (lm[0], lm[1]), 6, self.blue, cv2.FILLED)
                else:
                    cv2.circle(img, (lm[0], lm[1]), 6, self.red, cv2.FILLED)
        # mp_draw = mp.solutions.drawing_utils
        # mp_draw.plot_landmarks(self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

    def putAngle(self, img, FPS=True):
        PoseDetector.setAngle(self)

        [_, h], _ = cv2.getTextSize("Left", self.font, 1.5*self.fontSize, 2)
        box = cv2.getTextSize("Angle = 999.9", self.font, self.fontSize, 1)
        # cv2.rectangle(img, (0, 0), (round(self.factor*img.shape[1]), 2*h + 6*box[0][1]), self.gray, cv2.FILLED)

        leftOrg = [10, round(1.5*h)]
        rightOrg = [leftOrg[0] + box[0][0] + 20, round(1.5*h)]

        cv2.putText(img, "Left", leftOrg, self.font, 1.5*self.fontSize, self.red, 2, cv2.LINE_AA)
        cv2.putText(img, "Right", rightOrg, self.font, 1.5*self.fontSize, self.blue, 2, cv2.LINE_AA)

        for i, ang in enumerate(self.angles):
            txt = (self.labelAngles[i] + " = " + str(round(ang[0], 1)),
                   self.labelAngles[i] + " = " + str(round(ang[1], 1)))
            leftOrg[1] += 2 * box[0][1]
            rightOrg[1] += 2*box[0][1]
            cv2.putText(img, txt[0], rightOrg, self.font, self.fontSize, self.blue, 1, cv2.LINE_AA)
            cv2.putText(img, txt[1], leftOrg, self.font, self.fontSize, self.red, 1, cv2.LINE_AA)
        if FPS:
            txt = str(int(self.fps))
            height = img.shape[0]
            [w, h], _ = cv2.getTextSize("99", self.font, 1.5*self.fontSize, 2)
            orgTxt = (10, height-h)
            orgFps = (w+15, height-h)
            cv2.putText(img, txt, orgTxt, self.font, 1.5*self.fontSize, self.green, 2)
            cv2.putText(img, "fps", orgFps, self.font, self.fontSize, self.green, 2)

    # ----------------------------------------------- Angle Functions ----------------------------------------------- #
    def setAngle(self):
        self.angles.clear()
        knee = (PoseDetector.findAnglePts(self, 10, 8, 12), PoseDetector.findAnglePts(self, 9, 7, 11))
        self.angles.append(knee)

        ankle = [PoseDetector.findAngleAxs(self, 12, 10, theta=0), PoseDetector.findAngleAxs(self, 11, 9, theta=0)]
        for i, ank in enumerate(ankle):
            if ank > 90:
                ankle[i] = 180 - ank
        self.angles.append(ankle)

        hip = (PoseDetector.findAnglePts(self, 8, 2, 10), PoseDetector.findAnglePts(self, 7, 1, 9))
        self.angles.append(hip)

        spine = (PoseDetector.findAngleAxs(self, 17, 18, theta=-np.pi/2),
                 PoseDetector.findAngleAxs(self, 17, 18, theta=-np.pi/2))
        self.angles.append(spine)

        # ankle = (PoseDetector.findAnglePts(self, 12, 10, 16) - PoseDetector.findAnglePts(self, 16, 12, 14),
        #          PoseDetector.findAnglePts(self, 11, 9, 15) - PoseDetector.findAnglePts(self, 15, 11, 13))
        # self.angles.append(ankle)

    def findAnglePts(self, idRef, id1, id2):
        ptRef, pt1, pt2 = np.array(self.landmarks[idRef]), np.array(self.landmarks[id1]), np.array(self.landmarks[id2])
        vec1, vec2 = (pt1-ptRef), (pt2-ptRef)
        theta = np.arccos(np.dot(vec1, vec2) /
                          (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return np.degrees(theta)

    def findAngleAxs(self, idRef, id1, theta):
        ptRef, pt1 = np.array(self.landmarks[idRef]), np.array(self.landmarks[id1])
        vec = (pt1-ptRef)
        u = np.array([np.cos(theta), np.sin(theta)])
        phi = np.arccos(np.dot(vec, u) / np.linalg.norm(vec))
        return np.degrees(phi)

    def findDist(self, pt1, pt2):
        pt1, pt2 = np.array(pt1), np.array(pt2)
        dist = np.linalg.norm(pt2-pt1)
        return dist

    # --------------------------------------------- Dimension Functions --------------------------------------------- #
    def get_optimal_font_scale(self, text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale / 10, thickness=1)
            new_width = textSize[0][0]
            if new_width <= width:
                return scale / 10, textSize[0]

    def autoSize(self, img):
        leftLegVisibility = self.results.pose_landmarks.landmark[27].visibility
        rightLegVisibility = self.results.pose_landmarks.landmark[28].visibility

        if not leftLegVisibility < 0.8 and rightLegVisibility < 0.8:
            neckSize = PoseDetector.findDist(self, self.landmarks[18], self.landmarks[0])
            self.headRadius = min(round(0.59*neckSize), 130)
        else:
            leftLegSize = PoseDetector.findDist(self, self.landmarks[7], self.landmarks[11])
            rightLegSize = PoseDetector.findDist(self, self.landmarks[8], self.landmarks[12])
            if leftLegVisibility >= 0.8 and rightLegVisibility >= 0.8:
                legSize = 0.5 * (leftLegSize + rightLegSize)
                head = round(legSize)
            elif leftLegVisibility >= 0.8:
                head = round(leftLegSize)
            else:                                                                       # if rightLegVisibility >= 0.8
                head = round(rightLegSize)
            self.headRadius = min(round(0.2 * head), 30)

        h, w, _ = img.shape
        offset = 30                                             # 10px on the left + 20px separating left/right = 30px
        if h/1080 <= w/1920:
            self.factor = 0.33
        else:
            self.factor = 0.5
        width = round(0.5 * (self.factor * w - offset))
        self.fontSize, _ = PoseDetector.get_optimal_font_scale(self, "Angle = 999.9", width)

    # --------------------------------------------- Kinematics Functions -------------------------------------------- #
    def time(self):
        #
        self.curTime = time.time()
        self.Time.append(self.curTime - self.initTime)
        dt_app = self.curTime - self.prevTime

        # Computes Apparent duration so far and the dt according to the real video duration
        self.appDuration = self.curTime - self.initTime
        timeFactor = self.realDuration / self.appDuration
        self.dt = timeFactor * dt_app

        # Frames per Second
        self.fps = 1 / dt_app
        self.prevTime = self.curTime

    def kinematics(self):                                                                    # s, px, px/s, px/s²
        # Computes Velocity
        self.position.append(self.landmarks[16])
        ds = np.array(self.landmarks[16]) - np.array(self.prevLm[16])
        vel = ds / self.dt
        self.vel.append(vel)

        # Computes Acceleration
        dv = np.array(vel) - np.array(self.vel_prev)
        acc = dv / self.dt
        self.acc.append(acc)

        # Something
        self.vel_prev = vel

        PoseDetector.plotsXVA(self)

        return self.Time, self.position, self.vel, self.acc

    def setRealDuration(self, duration):
        self.realDuration = duration

    def plotsXYT(self):
        Xs, Ys, T = [], [], []
        if len(self.position) == 1:
            plt.ion()
            self.fig, self.ax = plt.subplots(3,1, constrained_layout=True)

            self.pltXT = self.ax[0].scatter(Xs, T, s=20, color='blue')
            self.ax[0].set_ylabel("Time")
            self.ax[0].set_xlabel("Width X")

            self.pltYT = self.ax[1].scatter(T, Ys, s=20, color='blue')
            self.ax[1].set_ylabel("Height Y")
            self.ax[1].set_xlabel("Time")

            self.pltYX = self.ax[2].scatter(Xs, Ys, s=20, color='blue')
            self.ax[2].set_ylabel("Vertical Position Y")
            self.ax[2].set_xlabel("Horizontal Position X")

            plt.draw()

        for p in self.position:
            Xs.append(p[0] - self.position[0][0])
            Ys.append(-p[1] + self.position[0][1])
        T = self.Time

        Xmin, Xmax = min(Xs)-25, max(Xs)+25
        Ymin, Ymax = min(Ys)-25, max(Ys)+25
        Tmax = max(T)+10

        self.ax[0].set_ylim(0, Tmax)
        self.ax[0].set_xlim(Xmin, Xmax)

        self.ax[1].set_ylim(Ymin, Ymax)
        self.ax[1].set_xlim(0, Tmax)

        self.ax[2].set_ylim(Ymin, Ymax)
        self.ax[2].set_xlim(Xmin, Xmax)

        self.pltXT.set_offsets(np.c_[Xs, T])
        self.pltYT.set_offsets(np.c_[T, Ys])
        self.pltYX.set_offsets(np.c_[Xs, Ys])
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def plotsXVA(self):
        Xs, vel, acc, T = [], [], [], []
        if len(self.position) == 1:
            plt.ion()
            self.fig, self.ax = plt.subplots(3,1, constrained_layout=True)

            self.pltXT = self.ax[0].scatter(T, Xs, s=10, color='blue')
            self.ax[0].set_ylabel("Width X")
            self.ax[0].set_xlabel("Time")

            self.pltVT = self.ax[1].scatter(T, vel, s=10, color='blue')
            self.ax[1].set_ylabel("Velocity X")
            self.ax[1].set_xlabel("Time")

            self.pltAT = self.ax[2].scatter(T, acc, s=10, color='blue')
            self.ax[2].set_ylabel("Acceleration X")
            self.ax[2].set_xlabel("Time")

            plt.draw()

        for p in self.position:
            Xs.append(p[0] - self.position[0][0])
        for v in self.vel:
            vel.append(v[0])
        for a in self.acc:
            acc.append(a[0])
        T = self.Time

        Xmin, Xmax = min(Xs)-100, max(Xs)+100
        Vmin, Vmax = min(vel)-100, max(vel)+100
        Amin, Amax = min(acc)-100, max(acc)+100
        Tmax = max(T)+10

        self.ax[0].set_ylim(Xmin, Xmax)
        self.ax[0].set_xlim(0, Tmax)

        self.ax[1].set_ylim(Vmin, Vmax)
        self.ax[1].set_xlim(0, Tmax)

        self.ax[2].set_ylim(Amin, Amax)
        self.ax[2].set_xlim(0, Tmax)

        self.pltXT.set_offsets(np.c_[T, Xs])
        self.pltVT.set_offsets(np.c_[T, vel])
        self.pltAT.set_offsets(np.c_[T, acc])
        self.fig.canvas.draw_idle()
        plt.pause(0.01)


def main():
    test = PoseDetector()
    path = input("Path: ")
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            _ = test.findPose(img)
            test.drawPosition(img)
            test.putAngle(img)

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
