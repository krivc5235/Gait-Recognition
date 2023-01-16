import sys
import pickle

import numpy as np
from PyQt4 import QtCore, QtGui, uic
import numpy as n
from PIL import Image
from PIL import ImageQt
import cv2
import os

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


#Loading the UI window
qtCreatorFile = "GaitUI.ui" 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

def cpickle(filename, data, compress=False):
    fo = open(filename, "wb")
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)
    fo.close()

def unpickle(filename):
    fo = open(filename, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

class GaitDemo(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        Set initial parameters here.
        Note that the demo window size is 1366*768, you can edit this via Qtcreator.
        In this demo, we take 20 frames of profiles to generate a GEI. You can edit this number by your self.
        '''
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.showFullScreen()
        self.setupUi(self)
        self.center()
        self.capture = cv2.VideoCapture(0)# Edit this default num to 1 or 2, if you have multiple cameras.
        #self.fgbg = cv2.createBackgroundSubtractorMOG()
        self.currentFrame=n.array([])
        self.firstFrame=None
        self.register_state = False
        self.recognition_state = False
        self.save_on = False
        self.pca=PCA(n_components=0.99)
        self.clf = LogisticRegression(C=.01,solver ='lbfgs',multi_class='auto',max_iter=250)
        self.regression_classifier = None
        self.gei_fix_num = 20
        
        #Set two window for raw video and segmentation.
        self.video_lable=QtGui.QLabel(self.centralwidget)
        self.seg_label = QtGui.QLabel(self.centralwidget)
        self._timer = QtCore.QTimer(self)
        self.video_lable.setGeometry(50,100, 512, 384)
        self.seg_label.setGeometry(600,100, 512, 384)
        self.load_dataset()
        self._timer.timeout.connect(self.play)
        
        #Waiting for you to push the button.
        self.register_2.clicked.connect(self.register_show)
        self.recognize.clicked.connect(self.recognition_show)
        self.updater.clicked.connect(self.update_bk)
        self.save_gei.clicked.connect(self.save_gei_f)
        self._timer.start(27)

        self.update()

    def save_gei_f(self):
        '''
        Waiting the save button.
        '''
        self.save_on = True
        self.label_4.setText('State: Saving')

    def register_show(self):
        '''
        To record the GEI into gait database.
        '''
        self.register_state = True
        self.recognition_state = False
        self.label_4.setText('State: Registering')
        self.gei_current = n.zeros((128,88), n.single)
        self.numInGEI = 0

    def load_dataset(self):
        '''
        Load gait database if existing.
        '''
        self.data_path = './GaitData'
        if os.path.exists(self.data_path):
            dic = unpickle(self.data_path)
            self.num = dic['num']
            self.gei = dic['gei']
            self.name = dic['name']
            dejanski = self.gei[:len(self.name), :, :]
            imena = list(set(self.name))
            if len(imena) > 1:
                dejanski = np.array(dejanski)
                train_data = dejanski.reshape(dejanski.shape[0], 128*88)
                train_data = shuffle(train_data)
                train_pca = self.pca.fit_transform(train_data)
                self.regression_classifier = self.clf.fit(train_pca, np.array(self.name))

        else:
            self.num = 0
            self.gei = n.zeros([100,128,88],n.uint8)
            self.name = []
            dic = {'num':self.num, 'gei':self.gei, 'name':self.name}
            cpickle(self.data_path, dic, compress=False)
        self.id_num.setText('%d' %self.num)
        self.label_4.setText('State: Running')

    def recognition_show(self):
        '''
        Working now and just recognizing the one in front of this camera.
        '''
        self.recognition_state = True
        self.register_state = False
        self.gei_current = n.zeros((128,88), n.single)
        self.numInGEI = 0
        self.label_4.setText('State: Recognition')

    def center(self):
        frameGm = self.frameGeometry()
        centerPoint = QtGui.QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def update_bk(self):
        '''
        If you moved the camera.
        '''
        self.label_4.setText('State: Running')
        self.firstFrame = self.FrameForUpdate

    def play(self):
        '''
        Main program.
        '''
        ret, frame=self.capture.read() #Read video from a camera.
        if(ret==True):
            frame = cv2.resize(frame,(512,384))
            #Apply background subtraction method.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            if self.firstFrame is None:
                self.firstFrame = gray # Set this frame as the background.
            frameDelta = cv2.absdiff(self.firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
            self.FrameForUpdate = gray 
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            thresh = n.array(thresh)
            max_rec=0
            
            #Find the max box.
            for c in cnts:
                if cv2.contourArea(c) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)

                if w>25 and h>50:
                    if max_rec<w*h:
                        max_rec = w*h
                        (x_max, y_max, w_max, h_max) = cv2.boundingRect(c)
            #If exist max box.
            if max_rec>0:
                cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
                if x_max>20: #To ignore some regions which contain parts of human body.
                    if self.register_state or self.recognition_state:
                        nim = n.zeros([thresh.shape[0]+10,thresh.shape[1]+10],n.single) # Enlarge the box for better result.
                        nim[y_max+5:(y_max + h_max+5),x_max+5:(x_max + w_max+5)] = thresh[y_max:(y_max + h_max),x_max:(x_max + w_max)]
                        offsetX = 20
                        # Get coordinate position.
                        ty, tx = (nim >100).nonzero()
                        sy, ey = ty.min(), ty.max()+1
                        sx, ex = tx.min(), tx.max()+1
                        h = ey - sy
                        w = ex - sx
                        if h>w:# Normal human should be like this, the height shoud be greater than width.
                            # Calculate the frame for GEI
                            cx = int(tx.mean())
                            cenX = h/2
                            start_w = int((h-w)/2)

                            if max(cx-sx,ex-cx)<cenX:
                                start_w = int(cenX - (cx-sx))
                            tim = n.zeros((h,h), n.single)
                            tim[:,start_w:start_w+w] = nim[sy:ey,sx:ex]
                            rim = Image.fromarray(n.uint8(tim)).resize((128,128), Image.ANTIALIAS)
                            tim = n.array(rim)[:,offsetX:offsetX+88]
                            if self.numInGEI<self.gei_fix_num:
                                self.gei_current += tim # Add up until reaching the fix number.
                            self.numInGEI += 1
                            
                        if  self.numInGEI>self.gei_fix_num:
                            if self.save_on:
                                #Save the GEI.
                                self.gei[self.num,:,:] = self.gei_current/self.gei_fix_num
                                Image.fromarray(n.uint8(self.gei_current/self.gei_fix_num)).save('./gei/gei%02d%s.jpg'%(self.num,self.id_name.toPlainText()))
                                self.name.append(self.id_name.toPlainText())
                                self.num +=1
                                self.id_num.setText('%d' %self.num)
                                dic = {'num':self.num, 'gei':self.gei, 'name':self.name}
                                cpickle(self.data_path, dic, compress=False)
                                self.save_on = False
                                self.load_dataset()
                                self.label_4.setText('State: Saved!')
                            elif self.recognition_state:
                                #Recognition.
                                self.gei_query = self.gei_current/(self.gei_fix_num)

                                if self.regression_classifier is not None:
                                    slika = np.array(self.gei_query)
                                    slika = slika.reshape(128*88)
                                    slika = slika.reshape(1, -1)
                                    slika_test = self.pca.transform(slika)

                                    id_rec = self.regression_classifier.predict(slika_test)
                                    id_rec = id_rec[0]
                                else:
                                    score = n.zeros(self.num)
                                    self.gei_to_com = n.zeros([128,88],n.single)
                                    for q in range(self.num):
                                        self.gei_to_com = self.gei[q,:,:]
                                        score[q]=n.exp(-(((self.gei_query[:]-self.gei_to_com[:])/(128*88))**2).sum())#Compare with gait database.
                                    q_id = score.argmax()
                                    if True:
                                        id_rec = '%s' % self.name[q_id]
                                if True:
                                    cv2.putText(frame,id_rec,(x_max+20,y_max+20),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2,color=(0,0,255))
            else:
                self.gei_current = n.zeros((128,88), n.single)
                self.numInGEI = 0
                
            #Show results.
            self.currentFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            self.currentSeg=Image.fromarray(thresh).convert('RGB')
            self.currentSeg = ImageQt.ImageQt(self.currentSeg)
            height,width=self.currentFrame.shape[:2]
            img=QtGui.QImage(self.currentFrame,
                              width,
                              height,
                              QtGui.QImage.Format_RGB888)
            img=QtGui.QPixmap.fromImage(img)
            self.video_lable.setPixmap(img)
            seg=QtGui.QImage(self.currentSeg)
            seg=QtGui.QPixmap(seg)
            self.seg_label.setPixmap(seg)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = GaitDemo()
    window.show()
    sys.exit(app.exec_())