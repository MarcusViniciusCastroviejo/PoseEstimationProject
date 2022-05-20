import cv2
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PoseModule2 import PoseDetector
from StackedImages import stackImages


class PoseEstimation:

    def __init__(self, mode, capture, save, chParams=True, blank=False):
        # ------------------------------------------------ Arguments ------------------------------------------------ #
        self.mode = mode                                        # (0) image                    # (1) video
        self.capture = capture                                  # (True) record live           # (False) file
        self.save = save                                        # (True) save file created     # (False) just visualise
        self.chParams = chParams                                # (True) smart size            # (False) original size
        self.blank = blank                                      # (True) background            # (False) original

        # ---------------------------------------------- Declarations ----------------------------------------------- #
        # Paths
        self.path = None
        self.newPath = None
        self.dojoPath = "C:\\Users\\mvcas\\PycharmProjects\\OpenCV\\Resources\\Simple_Dojo.jpg"

        # New Paths Settings
        self.windowName = "Video"
        self.directory = "C:\\Users\\mvcas\\PycharmProjects\\OpenCV\\SavedVideos"
        self.extension = ".mp4"

        # Images
        self.img = None
        self.imgBackground = None
        self.stackedImage = None
        self.imgSave = None
        self.background = None
        self.orgSize = [int(), int()]
        self.size = [int(), int()]

        # Save Settings
        self.file = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Scale factor
        self.scale = 0.75

        # Detector
        self.detector = None
        # self.detector = PoseDetector()
        # time.sleep(0.5)

    # ---------------------------------------------- Setting Functions ---------------------------------------------- #

    def ifBlank(self):
        # Blank (background)
        if self.blank:
            options = "[0]Preto \t\t [1] Dojo \t\t [2] Outro \n>> "
            name = int(input("Escolha seu background: \n" + options))
            if name == 0:                                                                           # Black background
                w, h = round(self.scale * 1920), round(self.scale * 1080)
                self.background = np.zeros([w, h, 3], np.uint8)
            elif name == 1:                                                                         # Dojo background
                self.background = cv2.imread(self.dojoPath)
            else:                                                                                   # Any background
                self.background = cv2.imread(input("Digite o caminho da imagem de fundo: \n>> "))

            # What to see
            options = "[0] Fundo com background \t\t [1] Arquivo stacked \t\t [2] Dois arquivos\n>> "
            if self.save:
                txt = "Qual arquivo você quer visualizar/salvar? \n" + options
            else:
                txt = "Qual arquivo você quer visualizar? \n" + options
            self.file = int(input(txt))

    def ifSave(self):
        # Save (new Path)
        if self.save:
            name = input("Digite o nome do novo arquivo: \n>> ")
            # Default
            self.newPath = self.directory + name + self.extension
            # Save background
            if self.blank:
                if self.file == 2:  # Stacked File
                    self.newPath = self.directory + name + "_stacked" + self.extension
                elif self.file == 3:  # Two files
                    self.newPath = (self.directory + name + "_original" + self.extension,
                                    self.directory + name + "_background" + self.extension)

    def begin(self):
        # Configuration
        self.path = input("Digite o caminho do Arquivo: \n>> ")
        PoseEstimation.ifBlank(self)
        PoseEstimation.ifSave(self)

        self.detector = PoseDetector()

        PoseEstimation.video(self)

    # ----------------------------------------------- Main Functions ------------------------------------------------ #

    def video(self):
        # Declarations
        save_out, _save_out = None, None

        cap = cv2.VideoCapture(self.path)
        size, fps = PoseEstimation.capSize(self, cap)
        print(size)

        # Pre-Save
        if self.save:
            # Two files
            if self.file == 3:
                size = PoseEstimation.smartDim(self, size)
                save_out = cv2.VideoWriter(self.newPath[0], self.fourcc, fps, size)
                _save_out = cv2.VideoWriter(self.newPath[1], self.fourcc, fps, size)

            # Stacked file size (Horizontal/Vertical)
            elif self.file == 2:
                if size[0] / 1920 > size[1] / 1080:
                    size = PoseEstimation.smartDim(self, (size[0], 2*size[1]))
                else:
                    size = PoseEstimation.smartDim(self, (2*size[0], size[1]))
                save_out = cv2.VideoWriter(self.newPath, self.fourcc, fps, size)

            # Default
            else:
                size = PoseEstimation.smartDim(self, size)
                save_out = cv2.VideoWriter(self.newPath, self.fourcc, fps, size)

        # Main Loop
        while cap.isOpened():
            ret, self.img = cap.read()
            # Reading status
            if ret:
                PoseEstimation.size(self)
                if not self.blank:
                    PoseEstimation.original(self)
                else:
                    PoseEstimation.blank(self)

                self.detector.time()
                t, x, v, a = self.detector.kinematics()

                # Display
                PoseEstimation.display(self)

                # Save
                if self.save:
                    PoseEstimation.imageSave(self)
                    if self.file == 3:
                        save_out.write(self.imgSave[0])
                        _save_out.write(self.imgSave[1])
                    else:
                        save_out.write(self.imgSave)

                # Stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Finished reading video!")
                break

        # PoseEstimation.plot2D(self, t, x, v, a)

        # Terminate
        if save_out is not None:
            save_out.release()
        if _save_out is not None:
            _save_out.release()
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------------------- Assistance Functions --------------------------------------------- #

    def original(self):
        if not self.chParams:
            return
        size = PoseEstimation.smartDim(self, self.size)
        self.img = cv2.resize(self.img, size, interpolation=cv2.INTER_AREA)

        lm = self.detector.findPose(self.img)
        self.detector.drawPosition(self.img)
        self.detector.putAngle(self.img)

    def blank(self):
        self.imgBackground = self.background.copy()

        if self.file == 0:
            size = PoseEstimation.smartDim(self, self.size)
            self.img = cv2.resize(self.img, size)
            lm = self.detector.findPose(self.img)

            self.imgBackground = cv2.resize(self.imgBackground, size)
            self.detector.drawPosition(self.imgBackground)
            self.detector.putAngle(self.imgBackground)

        else:
            stackOrientation = self.size[0] / 1920 > self.size[1] / 1080
            if stackOrientation:
                stacksize = PoseEstimation.smartDim(self, (self.size[0], 2 * self.size[1]))
                size = (stacksize[0], round(0.5*stacksize[1]))
            else:
                stacksize = PoseEstimation.smartDim(self, (2 * self.size[0], self.size[1]))
                size = (round(0.5*stacksize[0]), stacksize[1])

            self.img = cv2.resize(self.img, size)
            lm = self.detector.findPose(self.img)
            self.detector.drawPosition(self.img)
            self.detector.putAngle(self.img)

            self.imgBackground = self.background.copy()
            self.imgBackground = cv2.resize(self.imgBackground, size)
            self.detector.drawPosition(self.imgBackground)
            self.detector.putAngle(self.imgBackground)

            # Stacked image configuration
            if stackOrientation:
                self.stackedImage = stackImages(1, ([self.img], [self.imgBackground]))                     # Vertical
            else:
                self.stackedImage = stackImages(1, [self.img, self.imgBackground])                         # Horizontal

    def display(self):
        if self.file is None:
            cv2.imshow("Video", self.img)
        elif self.file == 0:
            cv2.imshow("Video", self.imgBackground)
        else:
            cv2.imshow("Stacked Video", self.stackedImage)

    def imageSave(self):
        if self.file == 0:
            self.imgSave = self.img
        elif self.file == 1:
            self.imgSave = self.imgBackground
        elif self.file == 2:
            self.imgSave = self.stackedImage
        elif self.file == 3:
            self.imgSave = (self.img, self.imgBackground)

    # --------------------------------------------- Secondary Functions --------------------------------------------- #

    def smartDim(self, size):
        if not self.chParams:
            return
        wfactor = 1920. / size[0]
        hfactor = 1080. / size[1]
        factor = min(wfactor, hfactor)
        width = round(factor * self.scale * size[0])
        height = round(factor * self.scale * size[1])
        return width, height

    def capSize(self, cap):
        # Video Properties
        width, height= round(cap.get(3)), round(cap.get(4))

        # Real Duration
        frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(5)
        realDuration = frameCount/fps
        self.detector.setRealDuration(realDuration)

        # Dimension
        size = PoseEstimation.smartDim(self, (width, height))

        # Output
        return size, fps

    def size(self):
        height, width, _ = self.img.shape
        self.size = (width, height)

    # ----------------------------------------------- Extra / Testing ----------------------------------------------- #

    def plot3D(self, t, s, v, a):
        x, y = [], []
        for p in s:
            x.append(p[0] - s[0][0])
            y.append(-p[1] + s[0][1])
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.scatter3D(x, t, y, '-o', color="green")
        ax.plot3D(x, t, y, color="green")
        ax.set_xlabel('Position X')
        ax.set_ylabel('Time')
        ax.set_zlabel('Posotion Y')
        plt.show()

    def plot2D(self, t, s, v, a):
        x, y, = [], []
        fig, ax = plt.subplots(3,1, constrained_layout=True)

        for p in s:
            x.append(p[0] - s[0][0])
            y.append(-p[1] + s[0][1])

        ax[0].plot(x, t, 'ob-', markersize=5)
        ax[0].set_ylabel("Time")
        ax[0].set_xlabel("Width X")

        ax[1].plot(t, y, 'ob-', markersize=5)
        ax[1].set_ylabel("Height Y")
        ax[1].set_xlabel("Time")

        ax[2].plot(x, y, 'ob-', markersize=5)
        ax[2].set_ylabel("Vertical Position Y")
        ax[2].set_xlabel("Horizontal Position X")

        plt.show()
