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
        self.dojoPath = "C:/Users/mvcas/PycharmProjects/OpenCV/Resources/Simple_Dojo.jpg"

        # New Paths Settings
        self.windowName = None
        self.directory = None
        self.extension = None

        # Images
        self.img = None
        self.imgBackground = None
        self.stackedImage = None
        self.imgDisplay = None
        self.background = None
        self.orgSize = [int(), int()]
        self.size = [int(), int()]

        # Video Settings
        self.cap = None
        self.capDevice = None

        # Save Settings
        self.file = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = None

        # Scale factor
        self.scale = 0.9

        # Detector
        self.detector = None
        # self.detector = PoseDetector()
        # time.sleep(0.5)

    # ---------------------------------------------- Setting Functions ---------------------------------------------- #
    def ifMode(self):
        # (Path configuration and Window Name)
        # Image
        if self.mode == 0:
            self.windowName = "Image"
            self.directory = "C:/Users/mvcas/PycharmProjects/OpenCV/SavedImages/"
            self.extension = ".jpg"
        # Video
        elif self.mode == 1:
            self.windowName = "Video"
            self.directory = "C:/Users/mvcas/PycharmProjects/OpenCV/SavedVideos/"
            self.extension = ".mp4"

    def ifCapture(self):
        # (Device, Orientation, Path/IP)
        # Capture
        if self.capture:
            self.capDevice = int(input("Escolha o dispositivo de câmera: \n"
                                       "[0] Webcam \t\t [1] Phone \n>> "))
            if self.capDevice == 1:  # Phone (Wifi IP)
                ip = input("Digite o Endereço de IP do wifi no celular: \n>> ")
                self.path = "https://" + ip + ":8080/video"
        # Not capture
        else:
            self.path = input("Digite o caminho do arquivo: \n>> ")

    def ifBlank(self):
        # Blank (background)
        if self.blank:
            name = int(input("Escolha seu background: \n"
                             "[0] Preto \t\t [1] Dojo \t\t [2] Outro \n>> "))
            if name == 0:  # Black background
                w, h = round(self.scale * 1920), round(self.scale * 1080)
                self.background = np.zeros([w, h, 3], np.uint8)
            elif name == 1:  # Dojo background
                self.background = cv2.imread(self.dojoPath)
            else:  # Any background
                self.background = cv2.imread(input("Digite o caminho da imagem de fundo: \n>> "))

            # What to save
            if self.save:
                self.file = int(input("Qual arquivo você quer salvar? \n"
                                      "[0] Fundo original \t\t [1] Fundo com background \t\t "
                                      "[2] Arquivo stacked \t\t [3] Dois arquivos\n>> "))

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
        PoseEstimation.ifMode(self)
        PoseEstimation.ifCapture(self)
        PoseEstimation.ifBlank(self)
        PoseEstimation.ifSave(self)

        self.detector = PoseDetector()

        # Image
        if self.mode == 0:
            # Capture
            if self.capture:
                PoseEstimation.imageCapture(self)
            # Not Capture
            else:
                PoseEstimation.image(self)
        # Video
        elif self.mode == 1:
            PoseEstimation.video(self)

    # ----------------------------------------------- Main Functions ------------------------------------------------ #
    def image(self):
        # Image
        self.img = cv2.imread(self.path)
        self.img = PoseEstimation.imgChange(self, self.img)
        PoseEstimation.size(self)
        _ = self.detector.findPose(self.img)
        self.detector.drawPosition(self.img)
        self.detector.putAngle(self.img)

        # Blank
        PoseEstimation.blank(self)
        # Display
        PoseEstimation.display(self)
        # Stop
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

        # Save
        if self.save:
            PoseEstimation.imageDisplay(self)
            if self.file == 3:
                cv2.imwrite(self.newPath[0], self.imgDisplay[0])
                cv2.imwrite(self.newPath[1], self.imgDisplay[1])
            else:
                cv2.imwrite(self.newPath, self.imgDisplay)

        # Terminate
        cv2.destroyAllWindows()

    def imageCapture(self):
        # Capture

        # Main Loop
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            # Reading status
            if not ret:
                print("[404] Error reading camera!")
                break
            else:
                PoseEstimation.size(self)
                # Image
                self.img = PoseEstimation.imgChange(self, self.img)
                PoseEstimation.size(self)
                _ = self.detector.findPose(self.img)
                self.detector.drawPosition(self.img)
                self.detector.putAngle(self.img)

                # Blank
                PoseEstimation.blank(self)
                # Display
                PoseEstimation.display(self)
                # Stop
                if cv2.waitKey(1) & 0xFF == ord('a'):
                    break

        # Captured Image
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

        # Save
        if self.save:
            PoseEstimation.imageDisplay(self)
            if self.file == 3:
                cv2.imwrite(self.newPath[0], self.imgDisplay[0])
                cv2.imwrite(self.newPath[1], self.imgDisplay[1])
            else:
                cv2.imwrite(self.newPath, self.imgDisplay)

        # Terminate
        self.cap.release()
        cv2.destroyAllWindows()

    def video(self):
        # Declarations
        save_out, _save_out = None, None
        # Capture
        PoseEstimation.capture(self)

        # Pre-Save
        if self.save:
            # Two files
            if self.file == 3:
                save_out = cv2.VideoWriter(self.newPath[0], self.fourcc, self.fps, self.size)
                _save_out = cv2.VideoWriter(self.newPath[1], self.fourcc, self.fps, self.size)
            else:
                # Stacked file size (Horizontal/Vertical)
                if self.file == 2:
                    if self.size[0] / 1920 > self.size[1] / 1080:
                        self.size = PoseEstimation.smartDim(self, (self.size[0], 2*self.size[1]))
                    else:
                        self.size = PoseEstimation.smartDim(self, (2*self.size[0], self.size[1]))
                # Default
                save_out = cv2.VideoWriter(self.newPath, self.fourcc, self.fps, self.size)

        # Main Loop
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            # Reading status
            if not ret:
                print("[404] Error reading video!")
                break
            else:
                PoseEstimation.size(self)

                if self.blank:
                    PoseEstimation.blank(self)
                else:
                    # Image
                    self.img = PoseEstimation.imgChange(self, self.img)
                    lm = self.detector.findPose(self.img)
                    self.detector.drawPosition(self.img)
                    self.detector.putAngle(self.img)

                self.detector.time()
                t, x, v, a = self.detector.kinematics()
                # Display
                PoseEstimation.display(self)

                # Save
                if self.save:
                    PoseEstimation.imageDisplay(self)
                    if self.file == 3:
                        save_out.write(self.imgDisplay[0])
                        _save_out.write(self.imgDisplay[1])
                    else:
                        save_out.write(self.imgDisplay)

                # Stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        # PoseEstimation.plot2D(self, t, x, v, a)
        PoseEstimation.plot3D(self, t, x, v, a)
        # Terminate
        if save_out is not None:
            save_out.release()
        if _save_out is not None:
            _save_out.release()
        self.cap.release()
        cv2.destroyAllWindows()

    # -------------------------------------------- Assistance Functions --------------------------------------------- #
    def capture(self):
        # Capture
        if self.capture:
            if self.capDevice == 0:
                self.cap = cv2.VideoCapture(0)  # Webcam
            elif self.capDevice == 1:
                self.cap = cv2.VideoCapture(self.path)  # Phone
            # Setting the Brightness
            self.cap.set(10, 130)

        # Not Capture
        else:
            self.cap = cv2.VideoCapture(self.path)

        # Dimension (for pre-save)
        self.size = PoseEstimation.capSize(self)

    def imgChange(self, img):
        if not self.chParams:
            return
        height, width, _ = img.shape
        dim = PoseEstimation.smartDim(self, (width, height))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def blank(self):
        if not self.blank:
            return
        stackOrientation = self.orgSize[0] / 1920 > self.orgSize[1] / 1080

        if stackOrientation:
            stacksize = PoseEstimation.smartDim(self, (self.orgSize[0], 2 * self.orgSize[1]))
            size = (stacksize[0], round(0.5*stacksize[1]))
        else:
            stacksize = PoseEstimation.smartDim(self, (2 * self.orgSize[0], self.orgSize[1]))
            size = (round(0.5*stacksize[0]), stacksize[1])
        self.size = size

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
            self.stackedImage = stackImages(1, ([self.img], [self.imgBackground]))                         # Vertical
        else:
            self.stackedImage = stackImages(1, [self.img, self.imgBackground])                             # Horizontal

    def display(self):
        if self.blank:
            cv2.imshow("Stacked " + self.windowName, self.stackedImage)
        else:
            cv2.imshow(self.windowName, self.img)

    def imageDisplay(self):
        if self.file == 0:
            self.imgDisplay = self.img
        if self.file == 1:
            self.imgDisplay = self.imgBackground
        if self.file == 2:
            self.imgDisplay = self.stackedImage
        if self.file == 3:
            self.imgDisplay = (self.img, self.imgBackground)

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

    def capSize(self):
        width, height, fps = round(self.cap.get(3)), round(self.cap.get(4)), round(self.cap.get(5))
        frameCount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        realDuration = frameCount/fps
        self.detector.setRealDuration(realDuration)
        self.fps = fps
        # Dimension
        size = PoseEstimation.smartDim(self, (width, height))
        self.orgSize = size
        return size

    def size(self):
        height, width, _ = self.img.shape
        self.size = (width, height)

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
