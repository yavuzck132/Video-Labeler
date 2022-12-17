# Detectron2 utilities and libraries
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.config import CfgNode
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg  # needed for getting configuration settings for the detectron2 model
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
from detectron2 import structures
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
# setup_logger()


# Pycocotools libraries
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Progress bar library
from tqdm.tk import trange, tqdm

# import some common libraries (AddedbyYK)
import numpy as np
import pandas as pd
import json, random
import sys
import matplotlib.pyplot as plt
import re  # - Regular expression libary
import time  # Time - date library
import copy
import torch, torchvision
from platform import python_version
from statistics import mean
import os
import io  # for saving images into memory and releasing data in memory
import datetime
from time import sleep
import collections
from IPython.display import clear_output  # for clearing Jupyter notebook outputs
from cv2 import CAP_PROP_POS_FRAMES

# previous libraries for tkinter and cv2
from tkinter import *
from tkinter import filedialog, ttk, messagebox
import PIL
from PIL import Image, ImageTk
import cv2
import shutil

# colors for the bboxes
COLORS = ['blue', 'pink', 'cyan', 'green', 'black', 'orange']


class App:

    def __init__(self, master):
        self.master = master

        # *** App States ***
        self.play = False  # Is the video currently playing?
        self.delay = 300  # Delay between frames - not sure what it should be, not accurate playback
        self.frame = 0  # Current frame
        self.frames = None  # Number of frames
        self.video_source = []  # List of video sources and their paths
        self.video_source_text = []  # List of name of the videos
        self.image_source = []
        self.image_source_text = []
        self.svSourcePath = os.getcwd()  # Get current working directory
        self.labelfilename = ''
        self.imgLabelFileName = ''
        self.predictionfilename = ''
        self.imagePredictionFileName = ''
        self.polygonFileName = ''
        self.zstackLabelFileName = ''
        self.zstackPredictionLabelFileName = ''
        self.selected_prediction_text = ""
        self.selected_image_prediction_text = ""
        self.selectedBboxIndex = -1
        self.deletedAll = False
        self.selectableBoxes = []
        self.selectedBBox = []
        self.editBoxes = []
        self.selectedEditBox = None
        self.moveLabel = False
        self.predictAllVids = True
        self.pop = None
        self.pb = None
        self.vidHeight = None
        self.vidWidth = None
        self.vidRatio = None
        self.imgWidth = None
        self.imgHeight = None
        self.selectedVideoIndex = 0
        self.selectedModelIndex = 0
        self.selectedImageIndex = 0
        self.selectedImageModelIndex = 0
        self.selectedTab = ""

        self.vid = None
        self.img = None
        self.RGB_img = None
        self.vid_source_text = ""
        self.img_source_text = ""
        self.photo = None
        self.next = "1"

        self.insertBbox = ""  # State of BBox insertion  (Changed by YCK)
        self.classcandidate_filename = 'class.txt'  # Get classes from this file
        self.cla_can_temp = []  # Put all the classes inside this array
        self.currentLabelclass = ''  # Current selected class
        self.loadPredictionsLabels = IntVar()  # Load saved predicted values if it is 1
        self.showPolyBbox = IntVar()
        self.showPolygons = IntVar()
        self.showLabels = IntVar()
        self.showZstackLabels = IntVar()
        self.showZstackPredictionLabels = IntVar()
        self.polygonPoints = []  # Added by YCK

        # *** App States No 2 ***
        self.imageDir = ''
        self.imageList = []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.tkimg = None
        self.Next_PR_plot_Button = False  # is the Next COCO Plot button pressed? By default it's not pressed = False
        self.Next_PR_plot_Button_zstack = False  # is the Next COCO Plot button pressed? By default it's not pressed = False

        # *** App models *** (AddedbyYK)
        self.models_path = os.getcwd() + "/videos/models"  # path to the models (not the model names)
        self.model_source_text = []  # save model names to list
        for file in os.listdir(self.models_path):
            if file.endswith('.pth'):  # if file is a .pth file (or a model, then read it)
                self.model_source_text.append(file[:-4])  # names of the models minus '.pth' (all models in folder)

        self.images_models_path = os.getcwd() + "/images/models"  # path to the models (not the model names)
        self.images_model_source_text = []  # save model names to list
        for file in os.listdir(self.images_models_path):
            if file.endswith('.pth'):  # if file is a .pth file (or a model, then read it)
                self.images_model_source_text.append(
                    file[:-4])  # names of the models minus '.pth' (all models in folder)

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0
        self.STATE['xp'], self.STATE['yp'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.lineList = []
        self.bboxClassId = None
        self.hl = None
        self.vl = None
        self.polygonId = None
        self.polygonBboxId = None
        self.zstackLabelBboxList = []
        self.zstackPredictionLabelBboxList = []

        # reference to predictions
        self.predIDList = []
        self.predID = None
        self.predList = []

        # *** Tabs Area ***
        self.tabControl = ttk.Notebook(root)
        self.tabControl.bind("<<NotebookTabChanged>>", self.selectTabMethod)

        self.tabs = dict()
        self.tabs['PAGE 1'] = ttk.Frame(self.tabControl)
        self.tabs['PAGE 2'] = ttk.Frame(self.tabControl)
        self.tabs['PAGE 3'] = ttk.Frame(self.tabControl)

        self.tabControl.add(self.tabs['PAGE 1'], text='Tab 1')
        self.tabControl.add(self.tabs['PAGE 2'], text='Tab 2')
        self.tabControl.add(self.tabs['PAGE 3'], text='Tab 3')
        self.tabControl.pack(expand=1, fill="both")

        self.selectedTab = self.tabControl.tab(self.tabControl.select(), "text")
        # This is needed to prevent user to enter any other value than integer
        vcmd = (master.register(self.validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        # *** Main Menu Bar ***
        menu = Menu(master)  # Create a menu and put it in main window
        root.config(menu=menu)  # We are configuring a menu for this project called menu

        fileMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=fileMenu)  # Add dropdown menu item
        fileMenu.add_command(label="New Project", command=self.doNothing)  # Add item to dropdown menu item
        fileMenu.add_command(label="Open Project", command=self.doNothing)
        fileMenu.add_separator()  # Creates a line that separates the menu items
        fileMenu.add_command(label="Exit", command=self.doNothing)

        # *** Toolbar ***
        self.toolbar = Frame(root, bg="lightgray")

        self.insertVideoButton = Button(self.toolbar, text="Insert Videos", command=self.importVideo)
        self.insertVideoButton.pack(side=LEFT, padx=2, pady=2)
        self.insertImageButton = Button(self.toolbar, text="Insert Image", command=self.importImage)
        self.insertImageButton.pack(side=LEFT, padx=2, pady=2)
        self.insertImageButton.pack_forget()

        self.toolbarInsertBbox = Button(self.toolbar, text="Insert Bbox",
                                        command=lambda: self.changeInsertBboxState("Rectangle"))  # Changed by YCK
        self.toolbarInsertBbox.pack(side=LEFT, padx=2, pady=2)
        self.toolbarInsertBbox["state"] = "disabled"

        # Added by YCK
        self.toolbarInsertPolyBbox = Button(self.toolbar, text="Insert polygon",
                                            command=lambda: self.changeInsertBboxState("polygon"))
        self.toolbarInsertPolyBbox.pack(side=LEFT, padx=2, pady=2)
        self.toolbarInsertPolyBbox["state"] = "disabled"

        self.toolbarInsertPolyBboxPoint = Button(self.toolbar, text="Insert polygon point",
                                                 command=lambda: self.changeInsertBboxState("addPolyPoint"))
        self.toolbarInsertPolyBboxPoint.pack(side=LEFT, padx=2, pady=2)
        self.toolbarInsertPolyBboxPoint["state"] = "disabled"

        self.toolbarDeletePolyBboxPoint = Button(self.toolbar, text="Delete polygon point",
                                                 command=lambda: self.changeInsertBboxState("deletePolyPoint"))
        self.toolbarDeletePolyBboxPoint.pack(side=LEFT, padx=2, pady=2)
        self.toolbarDeletePolyBboxPoint["state"] = "disabled"

        self.toolbar.pack(side=TOP, fill=X)

        # *** Main Frame ***
        self.tabFrame = Frame(root)
        self.tabFrame.pack(fill=BOTH, expand=1)

        # *** Right and Left Frame ***
        self.canvasArea = Frame(self.tabFrame)

        rightFrame = Frame(self.canvasArea, bg="lightgray", width=150)
        leftFrame = Frame(self.canvasArea, bg="lightgray", width=150)

        self.videoListboxFrame = Frame(rightFrame)
        self.imageListboxFrame = Frame(rightFrame)
        self.modelListboxFrame = Frame(leftFrame)  # (AddedbyYK)
        self.imagesModelListboxFrame = Frame(leftFrame)

        self.videoList = Listbox(self.videoListboxFrame)
        self.imageList = Listbox(self.imageListboxFrame)
        self.modelList = Listbox(self.modelListboxFrame)  # (AddedbyYK)
        self.imagesModelList = Listbox(self.imagesModelListboxFrame)

        # *** Add Scrollbar To List-boxes ***
        videoListboxScrollbar = Scrollbar(self.videoListboxFrame)
        videoListboxScrollbar.pack(side=RIGHT, fill=BOTH)
        self.videoList.config(yscrollcommand=videoListboxScrollbar.set)
        videoListboxScrollbar.config(command=self.videoList.yview)

        imageListboxScrollbar = Scrollbar(self.imageListboxFrame)
        imageListboxScrollbar.pack(side=RIGHT, fill=BOTH)
        self.imageList.config(yscrollcommand=imageListboxScrollbar.set)
        imageListboxScrollbar.config(command=self.imageList.yview)

        # (AddedbyYK)
        modelListboxScrollbar = Scrollbar(self.modelListboxFrame)
        modelListboxScrollbar.pack(side=RIGHT, fill=BOTH)
        self.modelList.config(yscrollcommand=modelListboxScrollbar.set)
        modelListboxScrollbar.config(command=self.modelList.yview)

        imagesModelListboxScrollbar = Scrollbar(self.imagesModelListboxFrame)
        imagesModelListboxScrollbar.pack(side=RIGHT, fill=BOTH)
        self.imagesModelList.config(yscrollcommand=imagesModelListboxScrollbar.set)
        imagesModelListboxScrollbar.config(command=self.imagesModelList.yview)

        self.deleteVideoButton = Button(rightFrame, text="Delete Video", command=self.deleteVideo)
        self.videoList.bind('<Double-Button>', lambda x: self.selectVideo(self.videoList.curselection()[0]))
        self.deleteAllVideosButton = Button(rightFrame, text="Delete All Videos", command=self.deleteAllVideos)

        self.deleteImageButton = Button(rightFrame, text="Delete Image", command=self.deleteImage)
        self.imageList.bind('<Double-Button>', lambda x: self.selectImage(self.imageList.curselection()[0]))
        self.deleteAllImagesButton = Button(rightFrame, text="Delete All Images", command=self.deleteAllImages)

        self.deleteAllBboxButton = Button(leftFrame, text="Delete All Labels", command=self.deleteAllConfirmed)

        self.savePredictionButton = Button(leftFrame, text="Convert Prediction", command=self.convertPrediction,
                                           state="disabled")
        self.savePredictionsButton = Button(leftFrame, text="Convert Predictions", comman=self.convertPredictions,
                                            state="disabled")
        self.predictionType = ttk.Combobox(leftFrame, state='readonly',
                                           values=["Current Frame", "This Video", "All Videos"])
        self.predictionType.current(0)

        # (AddedbyYK)
        DetectButton = Button(leftFrame, text="Detect Predictions", command=self.drawpredictions)
        self.modelList.bind('<Double-Button>', lambda x: self.loadmodel())
        self.imagesModelList.bind('<Double-Button>', lambda x: self.loadImageModel())

        # *** Class Combobox ***
        self.classname = StringVar()
        self.classcandidate = ttk.Combobox(leftFrame, state='readonly', textvariable=self.classname)
        if os.path.exists(self.classcandidate_filename):
            with open(self.classcandidate_filename) as cf:
                for line in cf.readlines():
                    self.cla_can_temp.append(line.strip('\n'))
        self.classcandidate['values'] = self.cla_can_temp
        self.classcandidate.current(0)
        self.currentLabelclass = self.classcandidate.get()
        self.btnclass = Button(leftFrame, text='Comfirm Class', command=self.setClass)

        loadPredictionCheckbox = Checkbutton(leftFrame, text='Load Predictions', variable=self.loadPredictionsLabels,
                                             command=self.editShowLabelPredState)
        showPolyBboxCheckbox = Checkbutton(leftFrame, text='Show Polygon Bbox', variable=self.showPolyBbox,
                                           command=self.editShowPolyBboxState)
        showPolygonsCheckbox = Checkbutton(leftFrame, text='Show Polygon', variable=self.showPolygons,
                                           command=self.editShowPolygonsState)
        showLabelsCheckbox = Checkbutton(leftFrame, text='Show Labels', variable=self.showLabels,
                                         command=self.editShowLabelsState)
        self.showZtackLabelCheckBox = Checkbutton(leftFrame, text='Show Zstack Labels', variable=self.showZstackLabels,
                                                  command=self.editShowZstackLabelState, state="disabled")
        self.showZstackPredictionLabelCheckbox = Checkbutton(leftFrame, text='Load Zstack Prediction',
                                                             variable=self.showZstackPredictionLabels,
                                                             command=self.editShowZstackLabelPredState,
                                                             state="disabled")

        # *** Fill in Right Frame ***
        self.videoList.pack()
        self.imageList.pack()
        self.videoListboxFrame.pack()
        self.deleteVideoButton.pack(padx=2, pady=2)
        self.deleteAllVideosButton.pack(padx=2, pady=2)
        rightFrame.pack(side=RIGHT, fill=Y)

        # *** Fill in Left Frame ***
        self.classcandidate.pack(padx=2, pady=2)
        self.btnclass.pack(padx=2, pady=2)
        self.deleteAllBboxButton.pack(padx=2, pady=2)
        self.modelList.pack()  # (AddedbyYK)
        self.modelListboxFrame.pack()  # (AddedbyYK)
        self.imagesModelList.pack()
        self.predictionType.pack(padx=2, pady=2)
        DetectButton.pack(padx=2, pady=2)  # (AddedbyYK)
        self.savePredictionButton.pack(padx=2, pady=2)
        self.savePredictionsButton.pack(padx=2, pady=2)
        leftFrame.pack(side=LEFT, fill=Y)
        self.showZstackPredictionLabelCheckbox.pack(side=BOTTOM)
        self.showZtackLabelCheckBox.pack(side=BOTTOM)
        showPolyBboxCheckbox.pack(side=BOTTOM)
        loadPredictionCheckbox.pack(side=BOTTOM)
        showPolygonsCheckbox.pack(side=BOTTOM)
        showLabelsCheckbox.pack(side=BOTTOM)

        # Buttons for the video
        self.buttonsBar = Frame(self.tabFrame, bg="lightgray")
        self.imageButtonsBar = Frame(self.tabFrame, bg="lightgray")

        self.btn_prevFrameButton = Button(self.buttonsBar, text="Prev Frame",
                                          command=lambda: self.setFrame(self.frame - 1))
        self.btn_prevFrameButton.pack(side=LEFT, padx=2, pady=2)

        self.btn_playButton = Button(self.buttonsBar, text="Reset Video", command=lambda: self.setFrame(0))
        self.btn_playButton.pack(side=LEFT, padx=2, pady=2)

        self.btn_playButton = Button(self.buttonsBar, text="Play/Stop", command=self.playButton)
        self.btn_playButton.pack(side=LEFT, padx=2, pady=2)

        self.btn_nextFrameButton = Button(self.buttonsBar, text="Next Frame",
                                          command=lambda: self.setFrame(self.frame + 1))
        self.btn_nextFrameButton.pack(side=LEFT, padx=2, pady=2)

        copyWithLabels = Label(self.buttonsBar, text="Copy Labels:")
        copyWithLabels.pack(side=LEFT, padx=2, pady=2)

        self.btn_prevFrameCopyButton = Button(self.buttonsBar, text="<",
                                              command=lambda: self.copyLabels(self.frame - 1))
        self.btn_prevFrameCopyButton.pack(side=LEFT, padx=2, pady=2)

        self.btn_nextFrameCopyButton = Button(self.buttonsBar, text=">",
                                              command=lambda: self.copyLabels(self.frame + 1))
        self.btn_nextFrameCopyButton.pack(side=LEFT, padx=2, pady=2)

        self.btn_setFrameEntry = Button(self.buttonsBar, text="Set Frame", command=lambda: self.getFrameNumber())
        self.btn_setFrameEntry.pack(side=RIGHT, padx=2, pady=2)

        self.setFrameEntry = Entry(self.buttonsBar, validate='key', validatecommand=vcmd)
        self.setFrameEntry.pack(side=RIGHT, pady=2)

        self.saveImageLabels = Button(self.imageButtonsBar, text="Save Labels",
                                      command=lambda: self.saveImageLabelsMethod())
        self.saveImageLabels.pack(side=LEFT, padx=2, pady=2)

        self.buttonsBar.pack(side=BOTTOM, fill=X)
        self.imageButtonsBar.pack(side=BOTTOM, fill=X)
        # *** StatusBar ***
        self.progressBar = Label(rightFrame, text="", bd=1, bg="lightgray", relief=SUNKEN, anchor=E)
        self.progressBar.pack(side=BOTTOM, fill=X)

        self.status = Label(self.tabFrame, text=(self.frame, "/", self.frames), bd=1, relief=SUNKEN,
                            anchor=E)  # bd and relief are border properties
        self.status.pack(side=BOTTOM, fill=X)

        # *** ModelListboxInsert *** (AddedbyYK)
        self.modelList.insert(END, *self.model_source_text)
        if self.model_source_text:
            self.modelList.select_set(0)
            self.modelList.itemconfig(self.selectedModelIndex, {'fg': 'red'})
            self.loadmodel()

        self.imagesModelList.insert(END, *self.images_model_source_text)
        if self.images_model_source_text:
            self.imagesModelList.select_set(0)
            self.imagesModelList.itemconfig(self.selectedImageModelIndex, {'fg': 'red'})
            self.loadImageModel()

        # *** Video Canvas ***
        self.canvas = Canvas(self.canvasArea, width=600, height=600)
        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<Button-3>", self.mouseRightClick)
        self.canvas.bind("<Motion>", self.mouseMove)
        self.canvas.bind("<ButtonRelease-1>", self.mouseRelease)
        self.master.bind("<Delete>", self.delBBox)
        self.master.bind("<Escape>", self.cancelDrawingKeypress)
        self.canvas.pack()

        self.canvasArea.pack(fill=BOTH, expand=Y)

        # *** tab2 Widgets ***

        # *** self.tabs['PAGE 3'] Widgets ***

        # Top toolbar
        toolbar_3 = Frame(self.tabs['PAGE 3'], bg="lightgray", height=30)
        toolbar_3.pack(side=TOP,
                       fill=X)  # fill = X means horizontal fill, fill = Y means vertical fill (no need for width)

        # Main frame (expanded)
        tabFrame_3 = Frame(self.tabs['PAGE 3'])  # tabframe takes everything below toolbar
        tabFrame_3.pack(fill=BOTH,
                        expand=1)  # fill = BOTH and expand = 1 fills the whole remaining screen (fills everything below toolbar frame)

        # Left and right frames
        canvasArea_3 = Frame(tabFrame_3)  # canvas area takes everything below the toolbar frame

        rightFrame_3 = Frame(canvasArea_3, bg="lightgray", width=150)  # rightframe is the frame within canvasarea
        leftFrame_3 = Frame(canvasArea_3, bg="lightgray", width=150)

        # Buttons before packing right and left frames

        rightFrame_3.pack(side=RIGHT, fill=Y)
        rightFrame_3.pack_propagate(0)
        leftFrame_3.pack(side=LEFT, fill=Y)
        leftFrame_3.pack_propagate(0)

        # Create JSON files Button
        ConvertJSONButton = Button(leftFrame_3, text="Create JSON Files", command=self.ConvertJSON)
        ConvertJSONButton.pack(padx=2, pady=2)

        # COCO stats button
        COCOStatsButton = Button(leftFrame_3, text="COCO Metrics", command=self.DisplayCOCOstats)
        COCOStatsButton.pack(padx=2, pady=2)

        # *** Right frame Buttons ***
        ShowsStatsButton = Button(rightFrame_3, text="Data Statistics", command=self.Datastats)
        ShowsStatsButton.pack(padx=2, pady=2)

        # Frames for COCO stats
        dispframe1 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe1.pack(fill=X)
        dispframe2 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe2.pack(fill=X)
        dispframe3 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe3.pack(fill=X)

        # Label and Display for COCO stats
        dispframe1label = Label(dispframe1, bg="papaya whip", text="AP50", font=('Arial', 10), width=8)
        dispframe1label.pack(side=LEFT, padx=2, pady=2)
        self.Display_COCO_stats = Entry(dispframe1, width=5, font=('Arial', 12))
        self.Display_COCO_stats.pack(side=LEFT, padx=2, pady=2)
        dispframe2label = Label(dispframe2, bg="papaya whip", text="AP75", font=('Arial', 10), width=8)
        dispframe2label.pack(side=LEFT, padx=2, pady=2)
        self.Display_COCO_stats2 = Entry(dispframe2, width=5, font=('Arial', 12))
        self.Display_COCO_stats2.pack(side=LEFT, padx=2, pady=2)
        dispframe3label = Label(dispframe3, bg="papaya whip", text="AP50:0.95", font=('Arial', 10), width=8)
        dispframe3label.pack(side=LEFT, padx=2, pady=2)
        self.Display_COCO_stats3 = Entry(dispframe3, width=5, font=('Arial', 12))
        self.Display_COCO_stats3.pack(side=LEFT, padx=2, pady=2)

        # Change COCO plots Button
        ChangeCOCOplotsButton = Button(leftFrame_3, text="COCO Plots", command=self.Next_PR_plot)  # Next_PR_plot
        ChangeCOCOplotsButton.pack(padx=2, pady=2)

        # CUSTOM stats Button
        CustomStatsButton = Button(leftFrame_3, text="CUSTOM Metrics", command=self.DisplayCustomstats)
        CustomStatsButton.pack(padx=2, pady=2)

        # Frames for CUSTOM stats
        dispframe4 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe4.pack(fill=X)
        dispframe5 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe5.pack(fill=X)

        # Label and Display for CUSTOM stats
        dispframe4label = Label(dispframe4, bg="papaya whip", text="AP50*", font=('Arial', 10), width=8)
        dispframe4label.pack(side=LEFT, padx=2, pady=2)
        self.Display_CUSTOM_stats = Entry(dispframe4, width=5, font=('Arial', 12))
        self.Display_CUSTOM_stats.pack(side=LEFT, padx=2, pady=2)
        Explain_star = Label(dispframe5, bg="lightgray", text="AP50*: Custom metric", font=('Arial', 8))
        Explain_star.pack(side=LEFT, padx=2, pady=2)

        # Change plot Button
        CustomplotButton = Button(leftFrame_3, text="Custom stats plot", command=self.Customstatsplot)  # Next_PR_plot
        CustomplotButton.pack(padx=2, pady=2)

        # *** Z-stack COCO_Eval

        # Create save z-stack buttons for 1) labels 2) predictions
        save_zstacklabelsButton = Button(leftFrame_3, text="Save z-stack labels", command=self.save_zstack_Labels)
        save_zstacklabelsButton.pack(padx=2, pady=2)

        save_zstackpredsButton = Button(leftFrame_3, text="Save z-stack predictions",
                                        command=self.save_zstack_predictions)
        save_zstackpredsButton.pack(padx=2, pady=2)

        # Create JSON files Button
        ConvertJSON_zstackButton = Button(leftFrame_3, text="Create z-stack JSON Files",
                                          command=self.ConvertJSON_zstack)
        ConvertJSON_zstackButton.pack(padx=2, pady=2)

        # COCO stats button
        COCOStats_zstackButton = Button(leftFrame_3, text="COCO z-stack Metrics", command=self.DisplayCOCOstats_zstack)
        COCOStats_zstackButton.pack(padx=2, pady=2)

        # Frames for COCO z-stack stats
        dispframe6 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe6.pack(fill=X)
        dispframe7 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe7.pack(fill=X)
        dispframe8 = Frame(leftFrame_3, bg="lightgray", height=30)
        dispframe8.pack(fill=X)

        # Label and Display for COCO stats
        dispframe6label = Label(dispframe6, bg="papaya whip", text="AP50", font=('Arial', 10), width=8)
        dispframe6label.pack(side=LEFT, padx=2, pady=2)
        self.Display_COCO_zstats = Entry(dispframe6, width=5, font=('Arial', 12))
        self.Display_COCO_zstats.pack(side=LEFT, padx=2, pady=2)
        dispframe7label = Label(dispframe7, bg="papaya whip", text="AP75", font=('Arial', 10), width=8)
        dispframe7label.pack(side=LEFT, padx=2, pady=2)
        self.Display_COCO_zstats2 = Entry(dispframe7, width=5, font=('Arial', 12))
        self.Display_COCO_zstats2.pack(side=LEFT, padx=2, pady=2)
        dispframe8label = Label(dispframe8, bg="papaya whip", text="AP50:0.95", font=('Arial', 10), width=8)
        dispframe8label.pack(side=LEFT, padx=2, pady=2)
        self.Display_COCO_zstats3 = Entry(dispframe8, width=5, font=('Arial', 12))
        self.Display_COCO_zstats3.pack(side=LEFT, padx=2, pady=2)

        # Change COCO plots Button
        ChangeCOCOplotsButton_zstack = Button(leftFrame_3, text="COCO z-stack Plots",
                                              command=self.Next_PR_plot_zstack)  # Next_PR_plot
        ChangeCOCOplotsButton_zstack.pack(padx=2, pady=2)

        # Bottom toolbar
        buttonsBar_3 = Frame(tabFrame_3, bg="lightgray", height=30)
        buttonsBar_3.pack(side=BOTTOM, fill=X)

        # Video canvas
        self.canvas_3 = Canvas(canvasArea_3, width=800, height=800)
        self.canvas_3.pack()
        canvasArea_3.pack(fill=BOTH)

        self.canvasAreaHeight = self.canvasArea.winfo_height()
        self.canvasAreaWidth = self.canvasArea.winfo_width()

        self.update()

    def doNothing(self):
        print("Doing Nothing")

    def selectTabMethod(self, event):
        id = self.tabControl.select()
        if self.tabControl.tab(id, "text") == "Tab 1":
            self.insertImageButton.pack_forget()
            self.imageListboxFrame.pack_forget()
            self.deleteImageButton.pack_forget()
            self.deleteAllImagesButton.pack_forget()
            self.imagesModelListboxFrame.pack_forget()
            self.imageButtonsBar.pack_forget()
            self.toolbar.pack(side=TOP, fill=X, in_=self.tabs['PAGE 1'])
            self.tabFrame.pack(fill=BOTH, expand=1, in_=self.tabs['PAGE 1'])
            self.insertVideoButton.pack(side=LEFT, padx=2, pady=2, before=self.toolbarInsertBbox)
            self.videoListboxFrame.pack()
            self.deleteVideoButton.pack(padx=2, pady=2)
            self.deleteAllVideosButton.pack(padx=2, pady=2)
            self.modelListboxFrame.pack(after=self.deleteAllBboxButton)
            self.predictionType.pack(padx=2, pady=2, after=self.modelListboxFrame)
            self.showZstackPredictionLabelCheckbox.pack(side=BOTTOM, after=self.savePredictionsButton)
            self.showZtackLabelCheckBox.pack(side=BOTTOM, after=self.showZstackPredictionLabelCheckbox)
            self.buttonsBar.pack(side=BOTTOM, fill=X, before=self.canvasArea)
            self.status.pack(side=BOTTOM, fill=X, after=self.buttonsBar)
            self.clearCanvas()
            if self.videoList.size() != 0:
                self.selectVideo(self.selectedVideoIndex)
        elif self.tabControl.tab(id, "text") == "Tab 2":
            self.insertVideoButton.pack_forget()
            self.videoListboxFrame.pack_forget()
            self.deleteVideoButton.pack_forget()
            self.deleteAllVideosButton.pack_forget()
            self.modelListboxFrame.pack_forget()
            self.predictionType.pack_forget()
            self.showZstackPredictionLabelCheckbox.pack_forget()
            self.showZtackLabelCheckBox.pack_forget()
            self.buttonsBar.pack_forget()
            self.status.pack_forget()
            self.toolbar.pack(side=TOP, fill=X, in_=self.tabs['PAGE 2'])
            self.tabFrame.pack(fill=BOTH, expand=1, in_=self.tabs['PAGE 2'])
            self.insertImageButton.pack(side=LEFT, padx=2, pady=2, before=self.toolbarInsertBbox)
            self.imageListboxFrame.pack()
            self.deleteImageButton.pack(padx=2, pady=2)
            self.deleteAllImagesButton.pack(padx=2, pady=2)
            self.imagesModelListboxFrame.pack(after=self.deleteAllBboxButton)
            self.imageButtonsBar.pack(side=BOTTOM, fill=X, before=self.canvasArea)
            self.clearCanvas()
            if self.imageList.size() != 0:
                self.selectImage(self.selectedImageIndex)

    # *** Functions for Tab 1 ***

    def editShowPolyBboxState(self):
        self.clearSelectedBbox()
        for index, bbox in enumerate(self.bboxList):
            if bbox[3] == "polygon":
                listBbox = list(bbox)
                if self.showPolyBbox.get() == 1 and self.showPolygons.get() == 1:
                    x1, y1, x2, y2 = self.getPolygonCoverPoints(bbox[0])
                    polygonBboxId = self.canvas.create_rectangle(x1, y1, x2, y2, width=2,
                                                                 outline=COLORS[
                                                                     int(self.currentLabelclass) % len(COLORS)],
                                                                 dash=(10, 10))
                    listBbox[2] = [bbox[2][0], polygonBboxId]
                else:
                    self.canvas.delete(bbox[2][1])
                    listBbox[2] = [bbox[2][0], None]
                bbox = tuple(listBbox)
                self.bboxList[index] = bbox

    def editShowPolygonsState(self):
        self.clearSelectedBbox()
        if self.showPolygons.get() == 1:
            self.toolbarInsertPolyBbox["state"] = "normal"
        else:
            self.toolbarInsertPolyBbox["state"] = "disabled"
            if self.insertBbox == "polygon":
                self.cancelDrawing()
        for index, bbox in enumerate(self.bboxList):
            if bbox[3] == "polygon":
                listBbox = list(bbox)
                if self.showPolygons.get() == 1:
                    polygonId = self.canvas.create_polygon(bbox[0], width=2,
                                                           outline=COLORS[int(self.currentLabelclass) % len(COLORS)],
                                                           fill='')
                    listBbox[2] = [polygonId, bbox[2][1]]
                    self.editPolygonPointBoxes(bbox[0])
                    listBbox[5] = self.editBoxes
                    self.editBoxes = []
                else:
                    self.canvas.delete(bbox[2][0])
                    listBbox[2] = [None, bbox[2][1]]
                    self.editBoxes = bbox[5]
                    listBbox[5] = []
                    self.clearEditBoxes()
                bbox = tuple(listBbox)
                self.bboxList[index] = bbox
        self.editShowPolyBboxState()

    def editShowLabelsState(self):
        self.clearSelectedBbox()
        if self.showLabels.get() == 1:
            self.toolbarInsertBbox["state"] = "normal"
            if self.loadPredictionsLabels.get() == 1:
                self.savePredictionsButton["state"] = "normal"
                self.savePredictionButton["state"] = "normal"
        else:
            self.toolbarInsertBbox["state"] = "disabled"
            self.savePredictionsButton["state"] = "disabled"
            self.savePredictionButton["state"] = "disabled"
        for index, bbox in enumerate(self.bboxList):
            if len(bbox) >= 7 and bbox[6] == "label":
                listBbox = list(bbox)
                if self.showLabels.get() == 1:
                    labelId = self.canvas.create_rectangle(bbox[0], bbox[1], \
                                                           bbox[2], bbox[3], \
                                                           width=2, outline=COLORS[int(bbox[4]) % len(COLORS)])
                    labelClassId = self.canvas.create_text(bbox[0], bbox[1], text=str(bbox[4]),
                                                           font="Calibri, 12", fill="purple", anchor="sw")
                    listBbox[5] = labelId
                    listBbox[7] = labelClassId
                else:
                    self.canvas.delete(bbox[5])
                    listBbox[5] = None
                    self.canvas.delete(bbox[7])
                    listBbox[7] = None
                bbox = tuple(listBbox)
                self.bboxList[index] = bbox

    def editShowLabelPredState(self):
        self.clearSelectedBbox()
        if self.loadPredictionsLabels.get() == 1 and self.showLabels.get() == 1:
            self.savePredictionsButton["state"] = "normal"
            self.savePredictionButton["state"] = "normal"
        else:
            self.savePredictionsButton["state"] = "disabled"
            self.savePredictionButton["state"] = "disabled"
        for index, predBox in enumerate(self.predList):
            listBbox = list(predBox)
            if self.loadPredictionsLabels.get() == 1:
                predId = self.canvas.create_rectangle(predBox[0], predBox[1], \
                                                      predBox[2], predBox[3], \
                                                      width=2, outline='red')
                predClassId = self.canvas.create_text(predBox[0], predBox[1], text=str(predBox[4]) + " %" + str(
                    "%.2f" % round((100 * predBox[5]), 2)),
                                                      font="Calibri, 12", fill="blue", anchor="sw")
                listBbox[6] = predId
                listBbox[7] = predClassId
            else:
                self.canvas.delete(predBox[6])
                listBbox[6] = None
                self.canvas.delete(predBox[7])
                listBbox[7] = None
            predBox = tuple(listBbox)
            self.predList[index] = predBox

    def editShowZstackLabelState(self):
        self.clearSelectedBbox()
        for index, bbox in enumerate(self.zstackLabelBboxList):
            if len(bbox) >= 7 and bbox[6] == "label":
                if self.showZstackLabels.get() == 1:
                    self.canvas.itemconfigure(bbox[5], state='normal')
                    self.canvas.itemconfigure(bbox[7], state='normal')
                    self.canvas.tag_raise(bbox[5])
                    self.canvas.tag_raise(bbox[7])
                else:
                    self.canvas.itemconfigure(bbox[5], state='hidden')
                    self.canvas.itemconfigure(bbox[7], state='hidden')

    def editShowZstackLabelPredState(self):
        self.clearSelectedBbox()
        for index, predBox in enumerate(self.zstackPredictionLabelBboxList):
            if self.showZstackPredictionLabels.get() == 1:
                self.canvas.itemconfigure(predBox[6], state='normal')
                self.canvas.itemconfigure(predBox[7], state='normal')
                self.canvas.tag_raise(predBox[6])
                self.canvas.tag_raise(predBox[7])
            else:
                self.canvas.itemconfigure(predBox[6], state='hidden')
                self.canvas.itemconfigure(predBox[7], state='hidden')

    def clearSelectedBbox(self):
        if self.selectedBBox:
            if len(self.selectedBBox) >= 7 and self.selectedBBox[0][6] != "label" or self.selectedBBox[0][
                3] != "polygon":
                self.canvas.itemconfig(self.selectedBBox[0][6], width=2)
            elif self.selectedBBox[0][3] == "polygon":
                self.canvas.itemconfig(self.selectedBBox[0][2][0], width=2)
                self.editBoxes = []
        self.clearEditBoxes()
        self.selectedBBox = []
        self.selectableBoxes = []

    def setClass(self):
        self.currentLabelclass = self.classcandidate.get()
        if self.selectedBBox and len(self.selectedBBox[0]) >= 7 and self.selectedBBox[0][6] == "label":
            self.canvas.itemconfig(self.selectedBBox[0][7], text=self.currentLabelclass)
            self.canvas.itemconfig(self.selectedBBox[0][5], outline=COLORS[int(self.currentLabelclass) % len(COLORS)])
            selectedBBox = list(self.selectedBBox)
            bboxValues = list(self.selectedBBox[0])
            bboxValues[4] = self.currentLabelclass
            selectedBBox[0] = tuple(bboxValues)
            self.selectedBBox = tuple(selectedBBox)
            self.bboxList[self.selectedBBox[1]] = self.selectedBBox[0]
        elif self.selectedBBox and self.selectedBBox[0][3] == "polygon" and self.showPolygons.get() == 1:
            self.canvas.itemconfig(self.selectedBBox[0][2][0],
                                   outline=COLORS[int(self.currentLabelclass) % len(COLORS)])
            if self.showPolyBbox.get() == 1:
                self.canvas.itemconfig(self.selectedBBox[0][2][1],
                                       outline=COLORS[int(self.currentLabelclass) % len(COLORS)])
            selectedBBox = list(self.selectedBBox)
            bboxValues = list(self.selectedBBox[0])
            bboxValues[1] = self.currentLabelclass
            selectedBBox[0] = tuple(bboxValues)
            self.selectedBBox = tuple(selectedBBox)
            self.bboxList[self.selectedBBox[1]] = self.selectedBBox[0]

    def loadmodel(self):  # (AddedbyYK)
        self.unselectLabel()
        self.modelList.itemconfig(self.selectedModelIndex, {'fg': 'black'})
        self.selectedModelIndex = self.modelList.curselection()[0]  # gets modelid
        self.current_model_path = os.path.join(self.models_path,
                                               self.model_source_text[
                                                   self.selectedModelIndex] + '.pth')  # gets full model path
        self.selected_prediction_text = self.model_source_text[self.selectedModelIndex]
        self.modelList.itemconfig(self.selectedModelIndex, {'fg': 'red'})
        # set configurations of the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = self.current_model_path  # path to the model that was just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom threshold for testing (any bounding box with a confidence score < 0.7 will not be displayed)
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
        self.predictor = DefaultPredictor(cfg)
        # self.predictor = predictor
        self.zstackPredictionLabelBboxList = []
        self.setFileNames()

    def loadImageModel(self):
        self.unselectLabel()
        self.imagesModelList.itemconfig(self.selectedImageModelIndex, {'fg': 'black'})
        self.selectedImageModelIndex = self.imagesModelList.curselection()[0]  # gets modelid
        self.current_model_path = os.path.join(self.models_path,
                                               self.model_source_text[
                                                   self.selectedModelIndex] + '.pth')  # gets full model path
        self.selected_prediction_text = self.model_source_text[self.selectedModelIndex]
        self.modelList.itemconfig(self.selectedModelIndex, {'fg': 'red'})
        # set configurations of the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = self.current_model_path  # path to the model that was just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom threshold for testing (any bounding box with a confidence score < 0.7 will not be displayed)
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
        self.imagePredictor = DefaultPredictor(cfg)
        # self.predictor = predictor
        self.setImageFileNames()

    def drawpredictions(self):  # (AddedbyYK)
        self.clearPredictions()
        self.unselectLabel()
        id = self.tabControl.select()
        if self.tabControl.tab(id, "text") == "Tab 1":
            self.checkPredictionType()
            ret, frame = self.vid.get_frame()
            if ret:
                self.outputs = self.predictor(frame)
                self.predbboxes = self.outputs['instances'].pred_boxes.tensor.tolist()  # all boxes for a single frame
                self.predbox_class = self.outputs['instances'].pred_classes.tolist()  # is a list of predicted classes
                self.predbox_scores = self.outputs['instances'].scores.tolist()  # scores of each predicted box
                for index, boxes in enumerate(self.predbboxes):
                    x1, y1, x2, y2 = boxes
                    predID = self.canvas.create_rectangle(int(x1 * self.vidRatio), int(y1 * self.vidRatio), \
                                                          int(x2 * self.vidRatio), int(y2 * self.vidRatio), \
                                                          width=2, outline='red')
                    tempClassId = self.canvas.create_text(int(x1 * self.vidRatio), int(y1 * self.vidRatio),
                                                          text=str(self.predbox_class[index]) + " %" + str(
                                                              "%.2f" % round((100 * self.predbox_scores[index]), 2)),
                                                          font="Calibri, 12", fill="purple", anchor="sw")
                    self.predList.append(
                        (int(x1 * self.vidRatio), int(y1 * self.vidRatio), int(x2 * self.vidRatio),
                         int(y2 * self.vidRatio),
                         self.predbox_class[index], self.predbox_scores[index],
                         predID, tempClassId))
                    self.predIDList.append(predID)
        elif self.tabControl.tab(id, "text") == "Tab 2":
            self.outputs = self.imagePredictor(self.img)
            self.predbboxes = self.outputs['instances'].pred_boxes.tensor.tolist()  # all boxes for a single frame
            self.predbox_class = self.outputs['instances'].pred_classes.tolist()  # is a list of predicted class
            for index, boxes in enumerate(self.predbboxes):
                x1, y1, x2, y2 = boxes
                predID = self.canvas.create_rectangle(int(x1 * self.vidRatio), int(y1 * self.vidRatio), \
                                                      int(x2 * self.vidRatio), int(y2 * self.vidRatio), \
                                                      width=2, outline='red')
                tempClassId = self.canvas.create_text(int(x1 * self.vidRatio), int(y1 * self.vidRatio),
                                                      text=str(self.predbox_class[index]) + " %" + str(
                                                          "%.2f" % round((100 * self.predbox_scores[index]), 2)),
                                                      font="Calibri, 12", fill="purple", anchor="sw")
                self.predList.append(
                    (int(x1 * self.vidRatio), int(y1 * self.vidRatio), int(x2 * self.vidRatio), int(y2 * self.vidRatio),
                     self.predbox_class[index], self.predbox_scores[index],
                     predID, tempClassId))
                self.predIDList.append(predID)
        self.savePredictions()

    def checkPredictionType(self):
        frameCount = 0
        predictedBoxes = []
        if self.predictionType.get() == "This Video":  # "All Videos"
            self.vid.goto_frame(0)
            while True:
                ret, frame = self.vid.get_frame()  # This moves to the next frame and returns it, so it does not return predictions for very first frame
                if ret:
                    self.outputs = self.predictor(frame)
                    self.predbboxes = self.outputs[
                        'instances'].pred_boxes.tensor.tolist()  # all boxes for a single frame
                    self.predbox_class = self.outputs[
                        'instances'].pred_classes.tolist()  # is a list of predicted classes
                    self.predbox_scores = self.outputs['instances'].scores.tolist()  # scores of each predicted box
                    for index, boxes in enumerate(self.predbboxes):
                        x1, y1, x2, y2 = boxes
                        predictedBoxes.append(
                            (int(x1) * self.vidRatio, int(y1) * self.vidRatio, int(x2) * self.vidRatio,
                             int(y2) * self.vidRatio, self.predbox_class[index], self.predbox_scores[index]))
                    if self.selected_prediction_text != "":
                        if not os.path.exists(
                                self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text):
                            os.makedirs(self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text)
                        if not os.path.exists(
                                self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text):
                            os.makedirs(
                                self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text)

                        frameCount = frameCount + 1  # Move this line to the bottom of the if after fixing the issue
                        self.predictionfilename = self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                            frameCount + 1) + ".txt"
                        if predictedBoxes != []:
                            with open(self.predictionfilename, 'w') as f:
                                for predbbox in predictedBoxes:
                                    f.write("{} ({}): {} {} {} {}\n".format((predbbox[4]), (predbbox[5]),
                                                                            int(float(predbbox[0]) / self.vidRatio),
                                                                            int(float(predbbox[1]) / self.vidRatio), (
                                                                                int((float(predbbox[2]) -
                                                                                     float(predbbox[
                                                                                               0])) / self.vidRatio)),
                                                                            (int(float(predbbox[3]) -
                                                                                 float(predbbox[1])) / self.vidRatio)))
                        elif os.path.exists(self.predictionfilename):
                            os.remove(self.predictionfilename)
                    predictedBoxes = []
                else:
                    break
            self.vid.goto_frame(self.frame)
        elif self.predictionType.get() == "All Videos":
            self.predictAllVideosMessage()
            self.predictVideos()

    # Predict on a single video or all videos
    def predictVideos(self):

        frameCount = 0
        predictedBoxes = []

        for vid_index, video in enumerate(tqdm(self.video_source, desc='Prediction Progress', leave=False)):
            self.progressBar["text"] = (vid_index + 1, "/", len(self.video_source))
            videoSource_text = self.video_source_text[vid_index]
            if self.predictAllVids or not self.predictAllVids and not os.path.exists(
                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + videoSource_text):
                vid = videoCapturer(video)
                vid.goto_frame(0)
                while True:
                    ret, frame = vid.get_frame()  # This moves to the next frame and returns it, so it does not return predictions for very first frame
                    if ret:
                        self.outputs = self.predictor(frame)
                        self.predbboxes = self.outputs[
                            'instances'].pred_boxes.tensor.tolist()  # all boxes for a single frame
                        self.predbox_class = self.outputs[
                            'instances'].pred_classes.tolist()  # is a list of predicted classes
                        self.predbox_scores = self.outputs['instances'].scores.tolist()  # scores of each predicted box
                        for index, boxes in enumerate(self.predbboxes):
                            x1, y1, x2, y2 = boxes
                            predictedBoxes.append(
                                (int(x1) * self.vidRatio, int(y1) * self.vidRatio, int(x2) * self.vidRatio,
                                 int(y2) * self.vidRatio, self.predbox_class[index],
                                 self.predbox_scores[index]))
                        if self.selected_prediction_text != "":
                            if not os.path.exists(
                                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text):
                                os.makedirs(self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text)
                            if not os.path.exists(
                                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + videoSource_text):
                                os.makedirs(
                                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + videoSource_text)

                            frameCount = frameCount + 1  # Move this line to the bottom of the if after fixing the issue
                            self.predictionfilename = self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + videoSource_text + "/" + videoSource_text + "_fr" + str(
                                frameCount + 1) + ".txt"
                            if predictedBoxes != []:
                                with open(self.predictionfilename, 'w') as f:
                                    for predbbox in predictedBoxes:
                                        f.write("{} ({}): {} {} {} {}\n".format((predbbox[4]), (predbbox[5]),
                                                                                int(float(predbbox[0]) / self.vidRatio),
                                                                                int(float(predbbox[1]) / self.vidRatio),
                                                                                (
                                                                                    int((float(predbbox[2]) -
                                                                                         float(predbbox[
                                                                                                   0])) / self.vidRatio)),
                                                                                (int(float(predbbox[3]) -
                                                                                     float(predbbox[
                                                                                               1])) / self.vidRatio)))
                            elif os.path.exists(self.predictionfilename):
                                os.remove(self.predictionfilename)
                        predictedBoxes = []
                    else:
                        frameCount = 0
                        # self.progress()
                        break
            root.update()

    def predictingVideosPopUp(self):
        self.pop = Toplevel(self.master)
        self.pop.geometry("500x400")
        self.pop.grab_set()
        self.pop.title("Predicting Video Labels")
        self.pb = ttk.Progressbar(
            self.pop,
            orient='horizontal',
            mode='determinate',
            length=450
        )
        self.pb.pack(pady=10)
        self.pb.after(self.delay, self.predictVideos)

    def progress(self):
        if self.pb['value'] < 100:
            self.pb['value'] += 50
            # value_label['text'] = update_progress_label()
        # else:
        #     showinfo(message='The progress completed!')

    def predictAllVideosMessage(self):
        msgBox = messagebox.askquestion("Predict Videos", "Do you want to predict videos that has prediction?")
        if msgBox == "yes":
            self.predictAllVids = True
            # self.predictingVideosPopUp()
        else:
            self.predictAllVids = False

    def clearBboxSelection(self):
        if self.selectedBboxIndex != -1:
            if len(self.bboxList) > self.selectedBboxIndex:
                self.canvas.itemconfig(self.bboxIdList[self.selectedBboxIndex], width=2)
            else:
                self.selectedBboxIndex = self.selectedBboxIndex - len(self.bboxList)
                self.canvas.itemconfig(self.predIDList[self.selectedBboxIndex], width=2)
            self.selectedBboxIndex = -1

    def changeInsertBboxState(self, insertState):
        if insertState == "label" or insertState == "polygon" or insertState == "":
            self.unselectLabel()
            self.selectedEditBox = None
            self.toolbarInsertPolyBboxPoint["state"] = "disabled"
            self.toolbarDeletePolyBboxPoint["state"] = "disabled"
        self.selectableBoxes = []
        self.insertBbox = insertState  # Changed by YCK
        if insertState != "deletePolyPoint":
            self.editBoxes = []

    def unselectLabel(self):
        if self.selectedBBox:
            if len(self.selectedBBox) >= 7 and self.selectedBBox[0][6] != "label" or self.selectedBBox[0][
                3] != "polygon":
                self.canvas.itemconfig(self.selectedBBox[0][6], width=2)
            elif self.selectedBBox and self.selectedBBox[0][3] == "polygon":
                self.canvas.itemconfig(self.selectedBBox[0][2][0], width=2)
            else:
                self.clearEditBoxes()
        self.selectedBBox = []
        self.bboxId = None
        self.insertBbox = ""

    def get_line(self, x1, y1, x2, y2):
        points = []
        issteep = abs(y2 - y1) > abs(x2 - x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2 - y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()
        return points

    def mouseClick(self, event):
        # Start drawing when mouse button-1 is clicked
        if self.insertBbox == "Rectangle" and self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
            self.STATE['click'] = 1
        elif self.insertBbox == "polygon" and self.STATE['click'] == 0:  # Added by YCK
            if not self.polygonPoints:
                self.polygonPoints.append(event.x)
                self.polygonPoints.append(event.y)
                self.drawEditBoxes(event.x, event.y)
                self.STATE['x'], self.STATE['y'] = event.x, event.y
            self.STATE['click'] = 1
        else:
            if self.STATE['click'] == 0:
                self.STATE['x'], self.STATE['y'] = event.x, event.y
                if self.selectedBBox:
                    if len(self.selectedBBox) >= 7 and self.selectedBBox[0][6] != "label" or self.selectedBBox[0][
                        3] != "polygon" and self.insertBbox != "addPolyPoint":
                        self.canvas.itemconfig(self.selectedBBox[0][6], width=2)
                    elif self.selectedBBox[0][3] == "polygon" and self.insertBbox != "addPolyPoint":
                        self.canvas.itemconfig(self.selectedBBox[0][2][0], width=2)
                if self.insertBbox == "addPolyPoint":
                    newPoints = self.selectedBBox[0][0]
                    points = None
                    selectedIndex = None
                    for index, point in enumerate(self.selectedBBox[0][0]):
                        if index == 0 or index % 2 == 0:
                            if index < len(self.selectedBBox[0][0]) - 3:
                                minx, maxx = min(self.selectedBBox[0][0][index],
                                                 self.selectedBBox[0][0][index + 2]), max(
                                    self.selectedBBox[0][0][index], self.selectedBBox[0][0][index + 2])
                                miny, maxy = min(self.selectedBBox[0][0][index + 1],
                                                 self.selectedBBox[0][0][index + 3]), max(
                                    self.selectedBBox[0][0][index + 1], self.selectedBBox[0][0][index + 3])
                                if minx - 4 <= event.x <= maxx + 4 and miny - 4 <= event.y <= maxy + 4:
                                    points = self.get_line(self.selectedBBox[0][0][index],
                                                           self.selectedBBox[0][0][index + 1],
                                                           self.selectedBBox[0][0][index + 2],
                                                           self.selectedBBox[0][0][index + 3])
                                    selectedIndex = index
                                    break
                            elif index == len(self.selectedBBox[0][0]) - 2:
                                minx, maxx = min(self.selectedBBox[0][0][0],
                                                 self.selectedBBox[0][0][index]), max(
                                    self.selectedBBox[0][0][0], self.selectedBBox[0][0][index])
                                miny, maxy = min(self.selectedBBox[0][0][1],
                                                 self.selectedBBox[0][0][index + 1]), max(
                                    self.selectedBBox[0][0][1], self.selectedBBox[0][0][index + 1])
                                if minx - 4 <= event.x <= maxx + 4 and miny - 4 <= event.y <= maxy + 4:
                                    points = self.get_line(self.selectedBBox[0][0][index],
                                                           self.selectedBBox[0][0][index + 1],
                                                           self.selectedBBox[0][0][0], self.selectedBBox[0][0][1])
                                    selectedIndex = index
                                    break
                    if points is not None:
                        for i, cordinates in enumerate(points):
                            if event.x - 50 <= cordinates[0] <= event.x + 50 and event.y - 50 <= cordinates[
                                1] <= event.y + 50:
                                newPoints.insert(selectedIndex + 2, event.x)
                                newPoints.insert(selectedIndex + 3, event.y)
                                break
                        self.redrawPolygon(newPoints, selectedIndex + 2)
                        self.STATE['click'] = 1
                    return
                if self.editBoxes:
                    for index, editbox in enumerate(self.editBoxes):
                        if (editbox[0] - 1) <= event.x <= (editbox[2] + 1) and (editbox[1] - 1) <= event.y <= (
                                editbox[3] + 1):
                            self.selectedEditBox = index
                            self.STATE['click'] = 1
                            break
                        else:
                            self.selectedEditBox = None
                    if self.insertBbox == "deletePolyPoint" and self.selectedEditBox is not None:
                        self.deletePolyPoint()
                if self.selectedEditBox is None:
                    if self.selectedBBox and self.selectedBBox[0][3] == "polygon":
                        self.editBoxes = []
                    else:
                        self.clearEditBoxes()
                    for index, bbox in enumerate(
                            self.bboxList):  # Add Label Bboxes that is inside the mouse clicked area
                        if len(bbox) >= 7 and bbox[6] == "label" and bbox[0] <= event.x <= bbox[2] and bbox[
                            1] <= event.y <= bbox[3] and self.showLabels.get() == 1:
                            self.selectableBoxes.append((bbox, index))
                        elif bbox[3] == "polygon" and self.showPolygons.get() == 1:
                            x1, y1, x2, y2 = self.getPolygonCoverPoints(bbox[0])
                            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                                point = Point(event.x, event.y)
                                iterator = iter(bbox[0])
                                zippedIterator = zip(iterator, iterator)
                                poly = Polygon(zippedIterator)
                                if poly.contains(point):
                                    self.selectableBoxes.append((bbox, index))
                    if self.loadPredictionsLabels.get() == 1:
                        for index, pred in enumerate(
                                self.predList):  # Add Predicted Bboxes that is inside the mouse clicked area
                            if pred[0] <= event.x <= pred[2] and pred[1] <= event.y <= pred[3]:
                                self.selectableBoxes.append((pred, index))
                    if not self.selectableBoxes:  # Clear bbox selection if selected an empty area
                        self.selectedBBox = []
                        self.toolbarInsertPolyBboxPoint["state"] = "disabled"
                        self.toolbarDeletePolyBboxPoint["state"] = "disabled"
                    else:  # If the boxes are inside each other, select the next possible box
                        if len(self.selectableBoxes) > 1 and self.selectedBBox and self.selectedBBox in self.selectableBoxes:
                            index = self.selectableBoxes.index(self.selectedBBox)
                            index = index + 1
                            if index == len(self.selectableBoxes):
                                index = 0
                            self.selectedBBox = self.selectableBoxes[index]
                        else:
                            self.selectedBBox = self.selectableBoxes[0]
                        self.moveLabel = True
                        self.selectableBoxes = []
                    if self.selectedBBox:
                        if len(self.selectedBBox[0]) >= 7 and self.selectedBBox[0][
                            6] == "label":  # Create Edit Boxes around Bbox
                            self.editBoxesPoints()
                            self.toolbarInsertPolyBboxPoint["state"] = "disabled"
                            self.toolbarDeletePolyBboxPoint["state"] = "disabled"
                        elif self.selectedBBox[0][3] == "polygon":
                            self.editBoxes = self.selectedBBox[0][5]
                            self.canvas.itemconfig(self.selectedBBox[0][2][0], width=4)
                            self.toolbarInsertPolyBboxPoint["state"] = "normal"
                            self.toolbarDeletePolyBboxPoint["state"] = "normal"
                        else:
                            self.canvas.itemconfig(self.selectedBBox[0][6], width=6)
                            self.toolbarInsertPolyBboxPoint["state"] = "disabled"
                            self.toolbarDeletePolyBboxPoint["state"] = "disabled"
                    self.STATE['click'] = 1

    def mouseRightClick(self, event):
        if self.insertBbox == "polygon":
            self.lineList.append((self.STATE['x'], self.STATE['y'], event.x, event.y, self.bboxId))
            for line in self.lineList:
                self.canvas.delete(line[4])
            self.polygonId = self.canvas.create_polygon(self.polygonPoints, width=2,
                                                        outline=COLORS[int(self.currentLabelclass) % len(COLORS)],
                                                        fill='')
            if self.showPolyBbox.get() == 1:
                x1, y1, x2, y2 = self.getPolygonCoverPoints(self.polygonPoints)
                self.polygonBboxId = self.canvas.create_rectangle(x1, y1, x2, y2, width=2,
                                                                  outline=COLORS[
                                                                      int(self.currentLabelclass) % len(COLORS)],
                                                                  dash=(10, 10))
            self.bboxList.append(
                (self.polygonPoints, self.currentLabelclass, [self.polygonId, self.polygonBboxId], "polygon",
                 self.bboxClassId, self.editBoxes))
            self.insertBbox = ""
            self.lineList = []
            self.polygonId = None
            self.polygonBboxId = None
            self.polygonPoints = []
            self.bboxId = None
            self.editBoxes = []

    def mouseRelease(self, event):
        if 1 == self.STATE['click'] and self.insertBbox == "Rectangle":  # Changed by YCK
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            if x2 - x1 > 6 and y2 - y1 > 6:
                self.bboxClassId = self.canvas.create_text(self.STATE['x'], self.STATE['y'],
                                                           text=self.currentLabelclass,
                                                           font="Calibri, 12", fill="purple", anchor="sw")
                self.bboxList.append((x1, y1, x2, y2, self.currentLabelclass, self.bboxId, "label", self.bboxClassId))
                self.bboxIdList.append(self.bboxId)
                self.bboxId = None
                self.insertBbox = ""  # Changed by YCK
                self.deletedAll = False
            else:
                self.canvas.delete(self.bboxId)
                self.bboxId = None
        elif 1 == self.STATE['click'] and self.selectedEditBox is not None and self.selectedBBox and \
                len(self.selectedBBox[0]) >= 7 and self.selectedBBox[0][6] == "label":
            self.bboxList = self.bboxList[:self.selectedBBox[1]] + [self.selectedBBox[0]] + self.bboxList[
                                                                                            self.selectedBBox[1] + 1:]
        elif 1 == self.STATE['click'] and self.moveLabel and self.selectedBBox and len(self.selectedBBox[0]) >= 7 and \
                self.selectedBBox[0][6] == "label":
            self.bboxList = self.bboxList[:self.selectedBBox[1]] + [self.selectedBBox[0]] + self.bboxList[
                                                                                            self.selectedBBox[1] + 1:]
            self.moveLabel = False
        elif 1 == self.STATE['click'] and self.insertBbox == "polygon":
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            if x2 - x1 > 4 or y2 - y1 > 4:
                if self.bboxId:
                    self.lineList.append((self.STATE['x'], self.STATE['y'], event.x, event.y, self.bboxId))
                self.drawEditBoxes(event.x, event.y)
                self.polygonPoints.append(event.x)
                self.polygonPoints.append(event.y)
                self.bboxId = None
                self.STATE['x'], self.STATE['y'] = event.x, event.y
        elif 1 == self.STATE['click'] and self.moveLabel and self.selectedBBox and self.selectedBBox[0][3] == "polygon":
            self.bboxList = self.bboxList[:self.selectedBBox[1]] + [self.selectedBBox[0]] + self.bboxList[
                                                                                            self.selectedBBox[1] + 1:]
            self.moveLabel = False
        elif 1 == self.STATE['click'] and self.selectedEditBox is not None and self.selectedBBox and \
                self.selectedBBox[0][3] == "polygon":
            self.bboxList = self.bboxList[:self.selectedBBox[1]] + [self.selectedBBox[0]] + self.bboxList[
                                                                                            self.selectedBBox[1] + 1:]
            self.moveLabel = False
        self.STATE['click'] = 0

    def mouseMove(self, event):
        # print('x: %d, y: %d' % (event.x, event.y))
        # Start drawing rectangle when the mouse is clicked
        self.STATE['xp'], self.STATE['yp'] = event.x, event.y
        if 1 == self.STATE['click'] and self.insertBbox == "Rectangle":  # Changed by YCK
            # Delete and recreate bbox every time you move your mouse
            if self.bboxId:
                self.canvas.delete(self.bboxId)
            # COLOR_INDEX = len(self.bboxIdList) % len(COLORS)
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=COLORS[int(self.currentLabelclass) % len(COLORS)])
        elif 1 == self.STATE['click'] and self.moveLabel and self.selectedBBox and len(self.selectedBBox[0]) >= 7 and \
                self.selectedBBox[0][6] == "label":
            xdifference = event.x - self.STATE['x']
            ydifference = event.y - self.STATE['y']
            self.STATE['x'] = event.x
            self.STATE['y'] = event.y
            newx1 = self.selectedBBox[0][0] + xdifference
            newy1 = self.selectedBBox[0][1] + ydifference
            newx2 = self.selectedBBox[0][2] + xdifference
            newy2 = self.selectedBBox[0][3] + ydifference
            selectedBboxClass = self.selectedBBox[0][4]
            selectedBboxIndex = self.selectedBBox[1]
            self.canvas.delete(self.selectedBBox[0][5])
            self.canvas.delete(self.selectedBBox[0][7])
            self.bboxId = self.canvas.create_rectangle(newx1, newy1,
                                                       newx2, newy2,
                                                       width=2,
                                                       outline=COLORS[int(self.selectedBBox[0][4]) % len(COLORS)])
            self.bboxClassId = self.canvas.create_text(newx1, newy1, text=self.selectedBBox[0][4],
                                                       font="Calibri, 12", fill="purple", anchor="sw")
            self.selectedBBox = []
            self.selectedBBox.append(
                (newx1, newy1, newx2, newy2, selectedBboxClass, self.bboxId, "label", self.bboxClassId))
            self.selectedBBox.append(selectedBboxIndex)
            self.clearEditBoxes()
            self.editBoxesPoints()
        elif 1 == self.STATE['click'] and self.selectedEditBox is not None and self.selectedBBox and \
                len(self.selectedBBox[0]) >= 7 and self.selectedBBox[0][6] == "label":
            self.canvas.delete(self.selectedBBox[0][5])
            self.canvas.delete(self.selectedBBox[0][7])
            if self.selectedEditBox == 0:
                self.redrawRectangle(event.x, event.y, self.selectedBBox[0][2], self.selectedBBox[0][3])
            elif self.selectedEditBox == 1:
                self.redrawRectangle(self.selectedBBox[0][0], event.y, self.selectedBBox[0][2], self.selectedBBox[0][3])
            elif self.selectedEditBox == 2:
                self.redrawRectangle(self.selectedBBox[0][0], event.y, event.x, self.selectedBBox[0][3])
            elif self.selectedEditBox == 3:
                self.redrawRectangle(self.selectedBBox[0][0], self.selectedBBox[0][1], event.x, self.selectedBBox[0][3])
            elif self.selectedEditBox == 4:
                self.redrawRectangle(self.selectedBBox[0][0], self.selectedBBox[0][1], event.x, event.y)
            elif self.selectedEditBox == 5:
                self.redrawRectangle(self.selectedBBox[0][0], self.selectedBBox[0][1], self.selectedBBox[0][2], event.y)
            elif self.selectedEditBox == 6:
                self.redrawRectangle(event.x, self.selectedBBox[0][1], self.selectedBBox[0][2], event.y)
            elif self.selectedEditBox == 7:
                self.redrawRectangle(event.x, self.selectedBBox[0][1], self.selectedBBox[0][2], self.selectedBBox[0][3])
        elif len(self.polygonPoints) > 1 and self.insertBbox == "polygon":
            if self.bboxId:
                self.canvas.delete(self.bboxId)
            # COLOR_INDEX = len(self.bboxIdList) % len(COLORS)
            self.bboxId = self.canvas.create_line(self.STATE['x'], self.STATE['y'],
                                                  event.x, event.y,
                                                  width=2, fill=COLORS[int(self.currentLabelclass) % len(COLORS)])
        elif 1 == self.STATE['click'] and self.moveLabel and self.selectedBBox and self.selectedBBox[0][3] == "polygon":
            newPoints = []
            xdifference = event.x - self.STATE['x']
            ydifference = event.y - self.STATE['y']
            self.STATE['x'] = event.x
            self.STATE['y'] = event.y
            for index, point in enumerate(self.selectedBBox[0][0]):
                if index % 2 == 0:
                    point = point + xdifference
                else:
                    point = point + ydifference
                newPoints.append(point)
            selectedBboxClass = self.selectedBBox[0][1]
            selectedBboxIndex = self.selectedBBox[1]
            self.canvas.delete(self.selectedBBox[0][2][0])
            self.canvas.delete(self.selectedBBox[0][2][1])
            self.polygonId = self.canvas.create_polygon(newPoints, width=4,
                                                        outline=COLORS[int(self.selectedBBox[0][1]) % len(COLORS)],
                                                        fill='')
            if self.showPolyBbox.get() == 1:
                x1, y1, x2, y2 = self.getPolygonCoverPoints(newPoints)
                self.polygonBboxId = self.canvas.create_rectangle(x1, y1, x2, y2, width=2,
                                                                  outline=COLORS[
                                                                      int(self.selectedBBox[0][1]) % len(COLORS)],
                                                                  dash=(10, 10))
            self.selectedBBox = []
            self.clearEditBoxes()
            self.editPolygonPointBoxes(newPoints)
            self.selectedBBox.append(
                (newPoints, selectedBboxClass, [self.polygonId, self.polygonBboxId], "polygon",
                 self.bboxClassId, self.editBoxes))
            self.selectedBBox.append(selectedBboxIndex)

        elif 1 == self.STATE['click'] and self.selectedEditBox is not None and self.selectedBBox and \
                self.selectedBBox[0][3] == "polygon":
            newPoints = self.selectedBBox[0][0]
            self.canvas.delete(self.selectedBBox[0][2][0])
            self.canvas.delete(self.selectedBBox[0][2][1])
            selectedBboxClass = self.selectedBBox[0][1]
            selectedBboxIndex = self.selectedBBox[1]
            for index, point in enumerate(self.selectedBBox[0][0]):
                if index % 2 == 0 and point == self.editBoxes[self.selectedEditBox][0] + 4 and self.selectedBBox[0][0][
                    index + 1] == self.editBoxes[self.selectedEditBox][1] + 4:
                    newPoints[index] = event.x
                    newPoints[index + 1] = event.y
                    break
            self.polygonId = self.canvas.create_polygon(newPoints, width=4,
                                                        outline=COLORS[int(self.selectedBBox[0][1]) % len(COLORS)],
                                                        fill='')
            if self.showPolyBbox.get() == 1:
                x1, y1, x2, y2 = self.getPolygonCoverPoints(newPoints)
                self.polygonBboxId = self.canvas.create_rectangle(x1, y1, x2, y2, width=2,
                                                                  outline=COLORS[
                                                                      int(self.selectedBBox[0][1]) % len(COLORS)],
                                                                  dash=(10, 10))
            self.selectedBBox = []
            self.clearEditBoxes()
            self.editPolygonPointBoxes(newPoints)
            self.selectedBBox.append(
                (newPoints, selectedBboxClass, [self.polygonId, self.polygonBboxId], "polygon",
                 self.bboxClassId, self.editBoxes))
            self.selectedBBox.append(selectedBboxIndex)

    def redrawRectangle(self, x1, y1, x2, y2):
        selectedBboxClass = self.selectedBBox[0][4]
        selectedBboxIndex = self.selectedBBox[1]
        self.bboxId = self.canvas.create_rectangle(x1, y1,
                                                   x2, y2,
                                                   width=2, outline=COLORS[int(self.selectedBBox[0][4]) % len(COLORS)])
        self.bboxClassId = self.canvas.create_text(x1, y1, text=self.selectedBBox[0][4],
                                                   font="Calibri, 12", fill="purple", anchor="sw")
        self.selectedBBox = []
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.selectedBBox.append((x1, y1, x2, y2, selectedBboxClass, self.bboxId, "label", self.bboxClassId))
        self.selectedBBox.append(selectedBboxIndex)
        self.clearEditBoxes()
        self.editBoxesPoints()

    def editBoxesPoints(self):
        self.drawEditBoxes(self.selectedBBox[0][0], self.selectedBBox[0][1])
        self.drawEditBoxes((self.selectedBBox[0][0] + self.selectedBBox[0][2]) / 2, self.selectedBBox[0][1])
        self.drawEditBoxes(self.selectedBBox[0][2], self.selectedBBox[0][1])
        self.drawEditBoxes(self.selectedBBox[0][2], (self.selectedBBox[0][1] + self.selectedBBox[0][3]) / 2)
        self.drawEditBoxes(self.selectedBBox[0][2], self.selectedBBox[0][3])
        self.drawEditBoxes((self.selectedBBox[0][0] + self.selectedBBox[0][2]) / 2, self.selectedBBox[0][3])
        self.drawEditBoxes(self.selectedBBox[0][0], self.selectedBBox[0][3])
        self.drawEditBoxes(self.selectedBBox[0][0], (self.selectedBBox[0][1] + self.selectedBBox[0][3]) / 2)

    def editPolygonPointBoxes(self, points):
        x, y = None, None
        for index, point in enumerate(points):
            if index % 2 == 0:
                x = point
            else:
                y = point
                self.drawEditBoxes(x, y)

    def drawEditBoxes(self, x, y):
        if self.selectedBBox and len(self.selectedBBox[0]) >= 7 and self.selectedBBox[0][6] == "label":
            tempId = self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4,
                                                  outline=COLORS[int(self.selectedBBox[0][4]) % len(COLORS)],
                                                  fill='blue')
        else:
            tempId = self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4,
                                                  outline=COLORS[int(self.currentLabelclass) % len(COLORS)],
                                                  fill='blue')
        self.editBoxes.append((x - 4, y - 4, x + 4, y + 4, tempId))

    def deletePolyPoint(self):
        newPoints = self.selectedBBox[0][0]
        self.canvas.delete(self.selectedBBox[0][2][0])
        self.canvas.delete(self.selectedBBox[0][2][1])
        selectedBboxClass = self.selectedBBox[0][1]
        selectedBboxIndex = self.selectedBBox[1]
        for index, point in enumerate(self.selectedBBox[0][0]):
            if index % 2 == 0 and point == self.editBoxes[self.selectedEditBox][0] + 4 and self.selectedBBox[0][0][
                index + 1] == self.editBoxes[self.selectedEditBox][1] + 4:
                newPoints.pop(index)
                newPoints.pop(index)
                break
        self.polygonId = self.canvas.create_polygon(newPoints, width=4,
                                                    outline=COLORS[int(self.selectedBBox[0][1]) % len(COLORS)],
                                                    fill='')
        if self.showPolyBbox.get() == 1:
            x1, y1, x2, y2 = self.getPolygonCoverPoints(newPoints)
            self.polygonBboxId = self.canvas.create_rectangle(x1, y1, x2, y2, width=2,
                                                              outline=COLORS[
                                                                  int(self.selectedBBox[0][1]) % len(COLORS)],
                                                              dash=(10, 10))
        self.selectedBBox = []
        self.clearEditBoxes()
        self.editPolygonPointBoxes(newPoints)
        self.selectedBBox.append(
            (newPoints, selectedBboxClass, [self.polygonId, self.polygonBboxId], "polygon",
             self.bboxClassId, self.editBoxes))
        self.selectedBBox.append(selectedBboxIndex)
        self.insertBbox = ""
        self.bboxList = self.bboxList[:self.selectedBBox[1]] + [self.selectedBBox[0]] + self.bboxList[
                                                                                        self.selectedBBox[1] + 1:]
        self.moveLabel = False
        self.STATE['click'] = 0

    def redrawPolygon(self, newPoints, selectedIndex):
        self.editBoxes = self.selectedBBox[0][5]
        self.clearEditBoxes()
        self.canvas.delete(self.selectedBBox[0][2][0])
        self.canvas.delete(self.selectedBBox[0][2][1])
        selectedBboxClass = self.selectedBBox[0][1]
        selectedBboxIndex = self.selectedBBox[1]
        self.polygonId = self.canvas.create_polygon(newPoints, width=4,
                                                    outline=COLORS[int(self.selectedBBox[0][1]) % len(COLORS)],
                                                    fill='')
        if self.showPolyBbox.get() == 1:
            x1, y1, x2, y2 = self.getPolygonCoverPoints(newPoints)
            self.polygonBboxId = self.canvas.create_rectangle(x1, y1, x2, y2, width=2,
                                                              outline=COLORS[
                                                                  int(self.selectedBBox[0][1]) % len(COLORS)],
                                                              dash=(10, 10))
        self.selectedBBox = []
        self.editPolygonPointBoxes(newPoints)
        self.selectedBBox.append(
            (newPoints, selectedBboxClass, [self.polygonId, self.polygonBboxId], "polygon",
             self.bboxClassId, self.editBoxes))
        self.selectedBBox.append(selectedBboxIndex)
        self.insertBbox = ""
        self.selectedEditBox = int(selectedIndex/2)

    def clearEditBoxes(self):
        for editBox in self.editBoxes:
            self.canvas.delete(editBox[4])
        self.editBoxes = []

    def clearPredictions(self):
        for predBoxes in self.predList:
            self.canvas.delete(predBoxes[6])
            self.canvas.delete(predBoxes[7])
        self.predList = []

    def delBBox(self, event):
        confirmDelete = False
        if self.selectedBBox and len(self.selectedBBox[0]) >= 7 and self.selectedBBox[0][6] == "label":
            self.canvas.delete(self.selectedBBox[0][5])
            self.canvas.delete(self.selectedBBox[0][7])
            confirmDelete = True
        elif self.selectedBBox and self.selectedBBox[0][3] == "polygon":
            self.canvas.delete(self.selectedBBox[0][2][0])
            self.canvas.delete(self.selectedBBox[0][2][1])
            confirmDelete = True
        elif len(self.lineList) > 0 and self.insertBbox == "polygon":
            self.STATE['x'], self.STATE['y'] = self.lineList[len(self.lineList) - 1][0], \
                                               self.lineList[len(self.lineList) - 1][1]
            self.canvas.delete(self.lineList[len(self.lineList) - 1][4])
            self.canvas.delete(self.bboxId)
            self.canvas.delete(self.editBoxes[len(self.editBoxes) - 1][4])
            self.bboxId = self.canvas.create_line(self.STATE['x'], self.STATE['y'],
                                                  self.STATE['xp'], self.STATE['yp'],
                                                  width=2, fill=COLORS[int(self.currentLabelclass) % len(COLORS)])
            self.lineList.pop()
            self.polygonPoints.pop()
            self.polygonPoints.pop()
            self.editBoxes.pop()
        if confirmDelete == True:
            self.selectedEditBox = None
            self.clearEditBoxes()
            print(self.selectedBBox[1])
            self.bboxList.pop(self.selectedBBox[1])
            self.unselectLabel()
            if len(self.bboxList) == 0:
                self.deletedAll = True

    def cancelDrawingKeypress(self, event):
        self.cancelDrawing()

    def cancelDrawing(self):
        if self.STATE['click'] == 0:
            if self.insertBbox == "polygon":
                for line in self.lineList:
                    self.canvas.delete(line[4])
                self.canvas.delete(self.bboxId)
                for editBox in self.editBoxes:
                    self.canvas.delete(editBox[4])
                self.editBoxes = []
                self.bboxId = None
                self.lineList = []
                self.polygonPoints = []
            self.unselectLabel()

    def clearBBox(self):
        self.unselectLabel()
        for idx in range(len(self.bboxList)):
            if len(self.bboxList[idx]) > 6:
                self.canvas.delete(self.bboxList[idx][5])
                self.canvas.delete(self.bboxList[idx][7])
            else:
                self.canvas.delete(self.bboxList[idx][2][0])
                self.canvas.delete(self.bboxList[idx][2][1])
                self.editBoxes = self.bboxList[idx][5]
                self.clearEditBoxes()
        self.bboxList = []
        self.deletedAll = True

    def deleteAllConfirmed(self):
        self.deletedAll = True
        self.clearBBox()

    def clearZstackBboxes(self):
        for bbox in self.zstackLabelBboxList:
            self.canvas.delete(bbox[5])
            self.canvas.delete(bbox[7])
        for predBox in self.zstackPredictionLabelBboxList:
            self.canvas.delete(predBox[6])
            self.canvas.delete(predBox[7])
        self.zstackLabelBboxList = []
        self.zstackPredictionLabelBboxList = []

    def resetBbox(self):
        self.deleteAllConfirmed()
        self.clearPredictions()
        self.labelfilename = ''
        self.imgLabelFileName = ''
        self.predictionfilename = ''
        self.zstackLabelFileName = ''
        self.zstackPredictionLabelFileName = ''

    def getPolygonCoverPoints(self, points):
        xValues = []
        yValues = []
        for index, point in enumerate(points):
            if index % 2 == 0:
                xValues.append(point)
            else:
                yValues.append(point)
        return min(xValues), min(yValues), max(xValues), max(yValues)

    def deleteVideo(self):
        checkDeletingCurrentVideo = False

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()

        selection = self.videoList.curselection()[0]
        if self.video_source_text[selection] == self.vid_source_text:
            checkDeletingCurrentVideo = True
        self.videoList.delete(ANCHOR)
        del self.video_source[selection]
        del self.video_source_text[selection]
        if len(self.video_source) == 0:
            self.clearCanvas()
            return
        if checkDeletingCurrentVideo:
            self.videoList.select_set(0)
            self.selectVideo(0)

    def deleteImage(self):
        checkDeletingCurrentImage = False

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()

        selection = self.imageList.curselection()[0]
        if self.image_source_text[selection] == self.img_source_text:
            checkDeletingCurrentImage = True
        self.imageList.delete(ANCHOR)
        del self.image_source[selection]
        del self.image_source_text[selection]
        if len(self.image_source) == 0:
            self.clearCanvas()
            return
        if checkDeletingCurrentImage:
            self.imageList.select_set(0)
            self.selectImage(0)

    def deleteAllVideos(self):

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()

        # self.unselectLabel()
        self.video_source.clear()
        self.video_source_text.clear()
        self.videoList.delete(0, END)
        self.clearCanvas()

    def deleteAllImages(self):

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()

        # self.unselectLabel()
        self.image_source.clear()
        self.image_source_text.clear()
        self.imageList.delete(0, END)
        self.clearCanvas()

    def clearCanvas(self):
        self.canvas.delete('all')
        self.cancelDrawing()
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.lineList = []
        self.bboxClassId = None
        self.hl = None
        self.vl = None
        self.polygonId = None
        self.polygonBboxId = None
        self.zstackLabelBboxList = []
        self.zstackPredictionLabelBboxList = []
        self.predIDList = []
        self.predID = None
        self.predList = []

    def setFileNames(self):
        self.labelfilename = self.svSourcePath + "/videos/label_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
            self.frame + 1) + ".txt"
        self.zstackLabelFileName = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_labels/" + self.vid_source_text + "/" + self.vid_source_text + "_fr1.txt"
        if os.path.exists(self.zstackLabelFileName):
            self.showZtackLabelCheckBox["state"] = "normal"
        else:
            self.showZtackLabelCheckBox["state"] = "disabled"
        if self.selected_prediction_text != "":
            self.predictionfilename = self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                self.frame + 1) + ".txt"
            self.zstackPredictionLabelFileName = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_predictions/" + self.selected_prediction_text + "/" + self.vid_source_text + "/" + self.vid_source_text + "_fr1.txt"
            if os.path.exists(self.zstackPredictionLabelFileName):
                self.showZstackPredictionLabelCheckbox["state"] = "normal"
            else:
                self.showZstackPredictionLabelCheckbox["state"] = "disabled"

    def setImageFileNames(self):
        self.imgLabelFileName = self.svSourcePath + "/images/label_files/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"
        if self.selected_image_prediction_text != "":
            self.imagePredictionFileName = self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text + "/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"

    def selectVideo(self, videoIndex):
        self.unselectLabel()
        self.setFileNames()
        self.saveLabels()
        self.savePolygons()
        self.videoListColorChanger()
        self.resetBbox()
        self.videoList.itemconfig(self.selectedVideoIndex, {'fg': 'black'})
        self.selectedVideoIndex = videoIndex
        self.videoList.itemconfig(videoIndex, {'fg': 'blue'})
        self.vid = videoCapturer(self.video_source[videoIndex])
        self.vid_source_text = self.video_source_text[videoIndex]
        self.frames = int(self.vid.frames)
        self.frame = 0
        self.setFileNames()
        self.loadLabels()
        self.loadPolygons()
        self.setFrame(0)
        self.clearZstackBboxes()
        self.loadZtackLabels()
        self.canvas.config(width=self.vid.width, height=self.vid.height)

    def selectImage(self, imageIndex):
        self.unselectLabel()
        self.setImageFileNames()
        self.saveImageLabelsMethod()
        self.imageListColorChanger()
        self.resetBbox()
        self.imageList.itemconfig(self.selectedImageIndex, {'fg': 'black'})
        self.selectedImageIndex = imageIndex
        self.imageList.itemconfig(imageIndex, {'fg': 'blue'})
        self.img = cv2.imread(self.image_source[imageIndex])
        self.RGB_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.imgHeight, self.imgWidth = self.img.shape[:2]
        self.img_source_text = self.image_source_text[0]
        self.getRatio(self.imgWidth, self.imgHeight)
        self.img_source_text = self.image_source_text[imageIndex]
        self.setImageFileNames()
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(self.RGB_img).resize(
                (int(self.imgWidth * self.vidRatio), int(self.imgHeight * self.vidRatio)),
                Image.NEAREST))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.loadImageLabels()
        self.loadImagePolygons()
        self.canvas.config(width=self.imgWidth, height=self.imgHeight)

    def importVideo(self):
        self.unselectLabel()
        """Browse for files when the Browse button is pressed"""
        # Open a file dialog and get the file path
        try:
            video_source = filedialog.askopenfilename(multiple=True,
                                                      initialdir=os.getcwd() + "/videos",
                                                      title="Select a File")
            messagebox.showinfo("Importing Videos", "For better visualization, keep the app fullscreen!")
            self.video_source = self.video_source + sorted(list(set(video_source) - set(self.video_source)))
            self.video_source_text.clear()
            self.videoList.delete(0, END)
            for x in self.video_source:
                base = os.path.basename(x)
                self.video_source_text.append(os.path.splitext(base)[0])
            for x in self.video_source_text:
                self.videoList.insert(END, x)
                if not os.path.exists(self.svSourcePath + "/videos/segmentation_files/" + x) and not os.path.exists(
                        self.svSourcePath + "/videos/label_files/" + x):
                    self.videoList.itemconfig(END, {'bg': 'Red'})
                elif not os.path.exists(self.svSourcePath + "/videos/label_files/" + x):
                    self.videoList.itemconfig(END, {'bg': 'Yellow'})
                elif not os.path.exists(self.svSourcePath + "/videos/segmentation_files/" + x):
                    self.videoList.itemconfig(END, {'bg': 'Orange'})
            self.videoList.itemconfig(0, {'fg': 'Blue'})
            self.selectedVideoIndex = 0
            self.vid = videoCapturer(self.video_source[0])
            self.vid_source_text = self.video_source_text[0]
            self.frames = int(self.vid.frames)
            self.getRatio(self.vid.width, self.vid.height)
            self.setFileNames()
            self.clearZstackBboxes()
            self.loadZtackLabels()
            self.setFrame(0)
            self.canvas.config(width=int(self.vid.width * self.vidRatio), height=int(self.vid.height * self.vidRatio))
            self.canvasAreaWidth = self.canvasArea.winfo_width()
            self.canvasAreaHeight = self.canvasArea.winfo_height()
        except AttributeError:
            print(AttributeError)

    def importImage(self):
        self.unselectLabel()
        """Browse for files when the Browse button is pressed"""
        # Open a file dialog and get the file path
        try:
            image_source = filedialog.askopenfilename(multiple=True,
                                                      initialdir=os.getcwd() + "/images",
                                                      title="Select a File")
            messagebox.showinfo("Importing Images", "For better visualization, keep the app fullscreen!")
            self.image_source = self.image_source + sorted(list(set(image_source) - set(self.image_source)))
            self.image_source_text.clear()
            self.imageList.delete(0, END)
            for x in self.image_source:
                base = os.path.basename(x)
                self.image_source_text.append(os.path.splitext(base)[0])
            for x in self.image_source_text:
                self.imageList.insert(END, x)
                if not os.path.exists(self.svSourcePath + "/images/segmentation_files/" + x) and not os.path.exists(
                        self.svSourcePath + "/images/label_files/" + x):
                    self.videoList.itemconfig(END, {'bg': 'Red'})
                elif not os.path.exists(self.svSourcePath + "/images/label_files/" + x):
                    self.videoList.itemconfig(END, {'bg': 'Yellow'})
                elif not os.path.exists(self.svSourcePath + "/images/segmentation_files/" + x):
                    self.videoList.itemconfig(END, {'bg': 'Orange'})
            self.imageList.itemconfig(0, {'fg': 'Red'})
            self.selectedImageIndex = 0
            self.img = cv2.imread(self.image_source[0])
            self.RGB_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.imgHeight, self.imgWidth = self.img.shape[:2]
            self.img_source_text = self.image_source_text[0]
            self.getRatio(self.imgWidth, self.imgHeight)
            self.setImageFileNames()
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(self.RGB_img).resize(
                    (int(self.imgWidth * self.vidRatio), int(self.imgHeight * self.vidRatio)),
                    Image.NEAREST))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.config(width=int(self.imgWidth * self.vidRatio), height=int(self.imgHeight * self.vidRatio))
            self.canvasAreaHeight = self.canvasArea.winfo_height()
            self.canvasAreaWidth = self.canvasArea.winfo_width()
            self.loadImageLabels()
            self.loadImagePolygons()
        except AttributeError:
            print(AttributeError)

    def getRatio(self, itemWidth, itemHeight):
        if self.canvasArea.winfo_width() - 300 < itemWidth:
            self.vidWidth = self.canvasArea.winfo_width()
        else:
            self.vidWidth = itemWidth
        if self.canvasArea.winfo_height() < itemHeight:
            self.vidHeight = self.canvasArea.winfo_height()
        else:
            self.vidHeight = itemHeight
        if itemHeight - self.vidHeight > itemWidth - self.vidWidth:
            self.vidRatio = self.vidHeight / itemHeight
        else:
            self.vidRatio = self.vidWidth / itemWidth

    def playButton(self):

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()
        if self.vid:
            if self.play:
                self.play = False
            else:
                self.play = True
                self.clearBboxSelection()

    def getFrameNumber(self):
        if self.vid:
            frame_no = int(self.setFrameEntry.get()) - 1

            self.setFrame(frame_no)

    # This method is to prevent user from typing any other value than integer to Entry
    def validate(self, action, index, value_if_allowed,
                 prior_value, text, validation_type, trigger_type, widget_name):
        if value_if_allowed:
            try:
                int(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def setFrame(self, frame_no):
        # Unselect selected bboxes
        self.saveLabels()
        self.savePolygons()
        self.videoListColorChanger()
        self.selectedEditBox = None
        if 0 <= frame_no < self.frames:
            if self.vid:
                if not self.play:
                    # Get a frame from the video source only if the video is supposed to play
                    self.clearPredictions()
                    ret, frame = self.vid.goto_frame(frame_no)
                    self.frame = frame_no
                    if ret:
                        self.photo = PIL.ImageTk.PhotoImage(
                            image=PIL.Image.fromarray(frame).resize(
                                (int(self.vid.width * self.vidRatio), int(self.vid.height * self.vidRatio)),
                                Image.NEAREST))
                        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                        self.status["text"] = (self.frame + 1, "/", self.frames)
                        self.loadLabels()
                        self.loadPolygons()
                        for bbox in self.zstackLabelBboxList:
                            if self.showZstackLabels.get() == 1:
                                self.canvas.tag_raise(bbox[5])
                                self.canvas.tag_raise(bbox[7])
                        for predBox in self.zstackPredictionLabelBboxList:
                            if self.showZstackPredictionLabels.get() == 1:
                                self.canvas.tag_raise(predBox[6])
                                self.canvas.tag_raise(predBox[7])

    def copyLabels(self, frame_no):

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()
        if self.showLabels.get() == 1:
            if 0 <= frame_no < self.frames:
                if not os.path.exists(
                        self.svSourcePath + "/videos/label_files/" + self.vid_source_text):
                    os.makedirs(self.svSourcePath + "/videos/label_files/" + self.vid_source_text)
                if frame_no > self.frame:
                    labelfilename = self.svSourcePath + "/videos/label_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                        self.frame + 2) + ".txt"
                else:
                    labelfilename = self.svSourcePath + "/videos/label_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                        self.frame) + ".txt"
                if labelfilename == '':
                    return
                if self.bboxList != []:
                    with open(labelfilename, 'w') as f:
                        for bbox in self.bboxList:
                            if len(bbox) > 6:
                                f.write("{}: {} {} {} {}\n".format(bbox[4], float(float(bbox[0]) / self.vidRatio),
                                                                   float(float(bbox[1]) / self.vidRatio),
                                                                   (float(
                                                                       (float(bbox[2]) - float(
                                                                           bbox[0])) / self.vidRatio)),
                                                                   (float(
                                                                       (float(bbox[3]) - float(
                                                                           bbox[1])) / self.vidRatio))))
                if os.path.exists(self.svSourcePath + "/videos/label_files/" + self.vid_source_text) and len(
                        os.listdir(self.svSourcePath + "/videos/label_files/" + self.vid_source_text)) == 0:
                    os.rmdir(self.svSourcePath + "/videos/label_files/" + self.vid_source_text)
        if self.showPolygons.get() == 1:
            if 0 <= frame_no < self.frames:
                exists_polygon = False
                if not os.path.exists(
                        self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text):
                    os.makedirs(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text)
                if frame_no > self.frame:
                    labelfilename = self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                        self.frame + 2) + ".txt"
                else:
                    labelfilename = self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                        self.frame) + ".txt"
                if labelfilename == '':
                    return
                if self.bboxList != []:
                    for bbox in self.bboxList:
                        if bbox[3] == "polygon":
                            exists_polygon = True
                            break
                    if exists_polygon:
                        with open(labelfilename, 'w') as f:
                            for bbox in self.bboxList:
                                if bbox[3] == "polygon":
                                    x1, y1, x2, y2 = self.getPolygonCoverPoints(bbox[0])
                                    points = [element / self.vidRatio for element in bbox[0]]
                                    f.write("{}: {} {} {} {} {}\n".format(bbox[1], float(float(x1) / self.vidRatio),
                                                                          float(float(y1) / self.vidRatio),
                                                                          float(float(x2) / self.vidRatio),
                                                                          float(float(y2) / self.vidRatio),
                                                                          points))
                if os.path.exists(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text) and len(
                        os.listdir(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text)) == 0:
                    os.rmdir(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text)
            self.setFrame(frame_no)

    def convertPrediction(self):
        if self.selectedBBox and self.selectedBBox[0][6] != "label":
            x1, y1, x2, y2, bboxclass, bboxscore, predId, predClassId = self.predList[self.selectedBBox[1]]
            if self.showLabels.get() == 1:
                bboxID = self.canvas.create_rectangle(float(x1), float(y1), \
                                                      float(x2), float(y2), \
                                                      width=2, outline=COLORS[int(bboxclass) % len(COLORS)])
                tempClassId = self.canvas.create_text(float(x1), float(y1), text=self.currentLabelclass,
                                                      font="Calibri, 12", fill="purple", anchor="sw")
            self.bboxList.append((float(x1), float(y1), float(x2), float(y2), bboxclass, bboxID, "label", tempClassId))
            self.bboxIdList.append(bboxID)
            self.deletedAll = False
        self.unselectLabel()

    def convertPredictions(self):

        # Unselect selected bboxes
        self.unselectLabel()
        self.selectedEditBox = None
        self.clearEditBoxes()

        for index, boxes in enumerate(self.predList):
            x1, y1, x2, y2, bboxclass, bboxscore, predId, predClassId = boxes
            bboxID = self.canvas.create_rectangle(float(x1), float(y1), \
                                                  float(x2), float(y2), \
                                                  width=2, outline=COLORS[int(bboxclass) % len(COLORS)])
            tempClassId = self.canvas.create_text(float(x1), float(y1), text=self.currentLabelclass,
                                                  font="Calibri, 12", fill="purple", anchor="sw")
            self.bboxList.append((float(x1), float(y1), float(x2), float(y2), bboxclass, bboxID, "label", tempClassId))
            self.bboxIdList.append(bboxID)
        self.unselectLabel()
        self.deletedAll = False

    def saveImageLabelsMethod(self):
        exists_label = False
        if self.imgLabelFileName == '':
            return
        if not os.path.exists(
                self.svSourcePath + "/images/label_files/" + self.img_source_text):
            os.makedirs(self.svSourcePath + "/images/label_files/" + self.img_source_text)
        if self.deletedAll and os.path.exists(self.imgLabelFileName):
            os.remove(self.imgLabelFileName)
        if self.bboxList != []:
            for bbox in self.bboxList:
                if len(bbox) >= 7 and bbox[6] == "label":
                    exists_label = True
                    break
            if exists_label:
                with open(self.imgLabelFileName, 'w') as f:
                    for bbox in self.bboxList:
                        if len(bbox) >= 7 and bbox[6] == "label":
                            f.write("{}: {} {} {} {}\n".format(bbox[4], float(float(bbox[0]) / self.vidRatio),
                                                               float(float(bbox[1]) / self.vidRatio),
                                                               (float(
                                                                   (float(bbox[2]) - float(bbox[0])) / self.vidRatio)),
                                                               (float(
                                                                   (float(bbox[3]) - float(bbox[1])) / self.vidRatio))))
        if os.path.exists(self.svSourcePath + "/images/label_files/" + self.img_source_text) and len(
                os.listdir(self.svSourcePath + "/images/label_files/" + self.img_source_text)) == 0:
            os.rmdir(self.svSourcePath + "/images/label_files/" + self.img_source_text)
        self.saveImagePolygonsMethod()

    def saveLabels(self):
        exists_label = False
        if self.labelfilename == '':
            return
        if not os.path.exists(
                self.svSourcePath + "/videos/label_files/" + self.vid_source_text):
            os.makedirs(self.svSourcePath + "/videos/label_files/" + self.vid_source_text)
        if self.deletedAll and os.path.exists(self.labelfilename):
            os.remove(self.labelfilename)
        if self.bboxList != []:
            for bbox in self.bboxList:
                if len(bbox) >= 7 and bbox[6] == "label":
                    exists_label = True
                    break
            if exists_label:
                with open(self.labelfilename, 'w') as f:
                    for bbox in self.bboxList:
                        if len(bbox) >= 7 and bbox[6] == "label":
                            f.write("{}: {} {} {} {}\n".format(bbox[4], float(float(bbox[0]) / self.vidRatio),
                                                               float(float(bbox[1]) / self.vidRatio),
                                                               (float(
                                                                   (float(bbox[2]) - float(bbox[0])) / self.vidRatio)),
                                                               (float(
                                                                   (float(bbox[3]) - float(bbox[1])) / self.vidRatio))))
        if os.path.exists(self.svSourcePath + "/videos/label_files/" + self.vid_source_text) and len(
                os.listdir(self.svSourcePath + "/videos/label_files/" + self.vid_source_text)) == 0:
            os.rmdir(self.svSourcePath + "/videos/label_files/" + self.vid_source_text)

    def savePolygons(self):
        exists_polygon = False
        if not os.path.exists(
                self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text):
            os.makedirs(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text)
        labelfileName = self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
            self.frame + 1) + ".txt"
        if self.deletedAll and os.path.exists(labelfileName):
            os.remove(labelfileName)
        if self.bboxList != []:
            for bbox in self.bboxList:
                if bbox[3] == "polygon":
                    exists_polygon = True
                    break
            if exists_polygon:
                with open(labelfileName, 'w') as f:
                    for bbox in self.bboxList:
                        if bbox[3] == "polygon":
                            x1, y1, x2, y2 = self.getPolygonCoverPoints(bbox[0])
                            points = [element / self.vidRatio for element in bbox[0]]
                            f.write("{}: {} {} {} {} {}\n".format(bbox[1], float(float(x1) / self.vidRatio),
                                                                  float(float(y1) / self.vidRatio),
                                                                  float(float(x2) / self.vidRatio),
                                                                  float(float(y2) / self.vidRatio),
                                                                  points))
        if os.path.exists(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text) and len(
                os.listdir(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text)) == 0:
            os.rmdir(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text)
        self.clearBBox()

    def videoListColorChanger(self):
        if os.path.exists(self.svSourcePath + "/videos/label_files/" + self.vid_source_text) and os.path.exists(
                self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text):
            self.videoList.itemconfig(self.selectedVideoIndex, {'bg': 'White'})
        elif os.path.exists(self.svSourcePath + "/videos/label_files/" + self.vid_source_text):
            self.videoList.itemconfig(self.selectedVideoIndex, {'bg': 'Orange'})
        elif os.path.exists(self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text):
            self.videoList.itemconfig(self.selectedVideoIndex, {'bg': 'Yellow'})
        else:
            self.videoList.itemconfig(self.selectedVideoIndex, {'bg': 'Red'})

    def imageListColorChanger(self):
        if os.path.exists(self.svSourcePath + "/images/label_files/" + self.img_source_text) and os.path.exists(
                self.svSourcePath + "/images/segmentation_files/" + self.img_source_text):
            self.imageList.itemconfig(self.selectedImageIndex, {'bg': 'White'})
        elif os.path.exists(self.svSourcePath + "/images/label_files/" + self.img_source_text):
            self.imageList.itemconfig(self.selectedImageIndex, {'bg': 'Orange'})
        elif os.path.exists(self.svSourcePath + "/images/segmentation_files/" + self.img_source_text):
            self.imageList.itemconfig(self.selectedImageIndex, {'bg': 'Yellow'})
        else:
            self.imageList.itemconfig(self.selectedImageIndex, {'bg': 'Red'})

    def saveImagePolygonsMethod(self):
        exists_polygon = False
        if not os.path.exists(
                self.svSourcePath + "/images/segmentation_files/" + self.img_source_text):
            os.makedirs(self.svSourcePath + "/images/segmentation_files/" + self.img_source_text)
        labelfileName = self.svSourcePath + "/images/segmentation_files/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"
        if self.deletedAll and os.path.exists(labelfileName):
            os.remove(labelfileName)
        if self.bboxList != []:
            for bbox in self.bboxList:
                if bbox[3] == "polygon":
                    exists_polygon = True
                    break
            if exists_polygon:
                with open(labelfileName, 'w') as f:
                    for bbox in self.bboxList:
                        if bbox[3] == "polygon":
                            x1, y1, x2, y2 = self.getPolygonCoverPoints(bbox[0])
                            points = [element / self.vidRatio for element in bbox[0]]
                            f.write("{}: {} {} {} {} {}\n".format(bbox[1], float(float(x1) / self.vidRatio),
                                                                  float(float(y1) / self.vidRatio),
                                                                  float(float(x2) / self.vidRatio),
                                                                  float(float(y2) / self.vidRatio),
                                                                  points))
        if os.path.exists(self.svSourcePath + "/images/segmentation_files/" + self.img_source_text) and len(
                os.listdir(self.svSourcePath + "/images/segmentation_files/" + self.img_source_text)) == 0:
            os.rmdir(self.svSourcePath + "/images/segmentation_files/" + self.img_source_text)

    def savePredictions(self):
        if self.selected_prediction_text != "":
            if not os.path.exists(
                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text):
                os.makedirs(self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text)
            if not os.path.exists(
                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text):
                os.makedirs(
                    self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text)

            self.predictionfilename = self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                self.frame + 1) + ".txt"
            if self.predList != []:
                with open(self.predictionfilename, 'w') as f:
                    for predbbox in self.predList:
                        f.write("{} ({}): {} {} {} {}\n".format((predbbox[4]), (predbbox[5]),
                                                                float(float(predbbox[0]) / self.vidRatio),
                                                                float(float(predbbox[1]) / self.vidRatio),
                                                                (float(float(predbbox[2]) - float(
                                                                    predbbox[0])) / self.vidRatio),
                                                                (float(float(predbbox[3]) - float(
                                                                    predbbox[1])) / self.vidRatio)))
            elif os.path.exists(self.predictionfilename):
                os.remove(self.predictionfilename)

    def saveImagePredictions(self):
        if self.selected_image_prediction_text != "":
            if not os.path.exists(
                    self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text):
                os.makedirs(self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text)
            if not os.path.exists(
                    self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text + "/" + self.img_source_text):
                os.makedirs(
                    self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text + "/" + self.img_source_text)

            self.imagePredictionFileName = self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text + "/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"
            if self.predList != []:
                with open(self.imagePredictionFileName, 'w') as f:
                    for predbbox in self.predList:
                        f.write("{} ({}): {} {} {} {}\n".format((predbbox[4]), (predbbox[5]),
                                                                float(float(predbbox[0]) / self.vidRatio),
                                                                float(float(predbbox[1]) / self.vidRatio),
                                                                (float(float(predbbox[2]) - float(
                                                                    predbbox[0])) / self.vidRatio),
                                                                (float(float(predbbox[3]) - float(
                                                                    predbbox[1])) / self.vidRatio)))
            elif os.path.exists(self.imagePredictionFileName):
                os.remove(self.imagePredictionFileName)

    def loadLabels(self):
        tmpId = None
        tmpClassId = None
        self.labelfilename = self.svSourcePath + "/videos/label_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
            self.frame + 1) + ".txt"
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    tmp = line.split()
                    tmp[0] = tmp[0].replace(":", "")
                    tmp[3] = float((float(tmp[3]) + float(tmp[1])) * self.vidRatio)
                    tmp[4] = float((float(tmp[4]) + float(tmp[2])) * self.vidRatio)
                    tmp[1] = float(float(tmp[1]) * self.vidRatio)
                    tmp[2] = float(float(tmp[2]) * self.vidRatio)
                    # color_index = (len(self.bboxList) - 1) % len(COLORS)
                    if self.showLabels.get() == 1:
                        tmpId = self.canvas.create_rectangle(tmp[1], tmp[2], \
                                                             tmp[3], tmp[4], \
                                                             width=2, outline=COLORS[int(tmp[0]) % len(COLORS)])
                        tmpClassId = self.canvas.create_text(tmp[1], tmp[2], text=str(tmp[0]),
                                                             font="Calibri, 12", fill="purple", anchor="sw")
                    self.bboxList.append((tmp[1], tmp[2], tmp[3], tmp[4], tmp[0], tmpId, "label", tmpClassId))
                    self.bboxIdList.append(tmpId)
        self.loadPredictionsFunction()

    def loadImageLabels(self):
        tmpId = None
        tmpClassId = None
        self.imgLabelFileName = self.svSourcePath + "/images/label_files/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"
        if os.path.exists(self.imgLabelFileName):
            with open(self.imgLabelFileName) as f:
                for (i, line) in enumerate(f):
                    tmp = line.split()
                    tmp[0] = tmp[0].replace(":", "")
                    tmp[3] = float((float(tmp[3]) + float(tmp[1])) * self.vidRatio)
                    tmp[4] = float((float(tmp[4]) + float(tmp[2])) * self.vidRatio)
                    tmp[1] = float(float(tmp[1]) * self.vidRatio)
                    tmp[2] = float(float(tmp[2]) * self.vidRatio)
                    if self.showLabels.get() == 1:
                        tmpId = self.canvas.create_rectangle(tmp[1], tmp[2], \
                                                             tmp[3], tmp[4], \
                                                             width=2, outline=COLORS[int(tmp[0]) % len(COLORS)])
                        tmpClassId = self.canvas.create_text(tmp[1], tmp[2], text=str(tmp[0]),
                                                             font="Calibri, 12", fill="purple", anchor="sw")
                    self.bboxList.append((tmp[1], tmp[2], tmp[3], tmp[4], tmp[0], tmpId, "label", tmpClassId))
                    self.bboxIdList.append(tmpId)
        self.loadImagePredictionsFunction()

    def loadPolygons(self):
        polygonId = None
        polygonBboxId = None
        labelfileName = self.svSourcePath + "/videos/segmentation_files/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
            self.frame + 1) + ".txt"
        if os.path.exists(labelfileName):
            with open(labelfileName) as f:
                for (i, line) in enumerate(f):
                    tmp = line.split()
                    tmp[0] = tmp[0].replace(":", "")
                    tmp[1] = float(float(tmp[1]) * self.vidRatio)
                    tmp[2] = float(float(tmp[2]) * self.vidRatio)
                    tmp[3] = float(float(tmp[3]) * self.vidRatio)
                    tmp[4] = float(float(tmp[4]) * self.vidRatio)
                    polygonPoints = [float(float(tmp[5].replace("[", "").replace(",", "")) * self.vidRatio)]
                    for element in tmp[6:len(tmp) - 1]:
                        polygonPoints.append(float(float(element.replace(",", "")) * self.vidRatio))
                    polygonPoints.append(
                        float(float(tmp[len(tmp) - 1].replace("]", "").replace(",", "")) * self.vidRatio))
                    if self.showPolygons.get() == 1:
                        polygonId = self.canvas.create_polygon(polygonPoints, width=2,
                                                               outline=COLORS[
                                                                   int(tmp[0]) % len(COLORS)], fill='')
                    if self.showPolyBbox.get() == 1:
                        polygonBboxId = self.canvas.create_rectangle(tmp[1], tmp[2], tmp[3], tmp[4], width=2,
                                                                     outline=COLORS[
                                                                         int(tmp[0]) % len(
                                                                             COLORS)],
                                                                     dash=(10, 10))
                    if self.showPolygons.get() == 1:
                        for j, k in zip(polygonPoints[0::2], polygonPoints[1::2]):
                            self.drawEditBoxes(j, k)
                    self.bboxList.append(
                        (polygonPoints, tmp[0], [polygonId, polygonBboxId], "polygon",
                         self.bboxClassId, self.editBoxes))
                    self.editBoxes = []

    def loadImagePolygons(self):
        polygonId = None
        polygonBboxId = None
        labelfileName = self.svSourcePath + "/images/segmentation_files/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"
        if os.path.exists(labelfileName):
            with open(labelfileName) as f:
                for (i, line) in enumerate(f):
                    tmp = line.split()
                    tmp[0] = tmp[0].replace(":", "")
                    tmp[1] = float(float(tmp[1]) * self.vidRatio)
                    tmp[2] = float(float(tmp[2]) * self.vidRatio)
                    tmp[3] = float(float(tmp[3]) * self.vidRatio)
                    tmp[4] = float(float(tmp[4]) * self.vidRatio)
                    polygonPoints = [float(float(tmp[5].replace("[", "").replace(",", "")) * self.vidRatio)]
                    for element in tmp[6:len(tmp) - 1]:
                        polygonPoints.append(float(float(element.replace(",", "")) * self.vidRatio))
                    polygonPoints.append(
                        float(float(tmp[len(tmp) - 1].replace("]", "").replace(",", "")) * self.vidRatio))
                    if self.showPolygons.get() == 1:
                        polygonId = self.canvas.create_polygon(polygonPoints, width=2,
                                                               outline=COLORS[
                                                                   int(tmp[0]) % len(COLORS)], fill='')
                    if self.showPolyBbox.get() == 1:
                        polygonBboxId = self.canvas.create_rectangle(tmp[1], tmp[2], tmp[3], tmp[4], width=2,
                                                                     outline=COLORS[
                                                                         int(tmp[0]) % len(
                                                                             COLORS)],
                                                                     dash=(10, 10))
                    if self.showPolygons.get() == 1:
                        for j, k in zip(polygonPoints[0::2], polygonPoints[1::2]):
                            self.drawEditBoxes(j, k)
                    self.bboxList.append(
                        (polygonPoints, tmp[0], [polygonId, polygonBboxId], "polygon",
                         self.bboxClassId, self.editBoxes))
                    self.editBoxes = []

    def loadPredictionsFunction(self):
        tmpId = None
        tempClassId = None
        if self.selected_prediction_text != "":
            self.predictionfilename = self.svSourcePath + "/videos/predictions/" + self.selected_prediction_text + "/" + self.vid_source_text + "/" + self.vid_source_text + "_fr" + str(
                self.frame + 1) + ".txt"
            if os.path.exists(self.predictionfilename):
                with open(self.predictionfilename) as f:
                    for (i, line) in enumerate(f):
                        tmp = line.split()
                        tmp[1] = tmp[1].replace(":", "")
                        tmp[1] = tmp[1].replace("(", "")
                        tmp[1] = tmp[1].replace(")", "")
                        tmp[1] = float(tmp[1])
                        tmp[4] = float((float(tmp[4]) + float(tmp[2])) * self.vidRatio)
                        tmp[5] = float((float(tmp[5]) + float(tmp[3])) * self.vidRatio)
                        tmp[2] = float(float(tmp[2]) * self.vidRatio)
                        tmp[3] = float(float(tmp[3]) * self.vidRatio)
                        # color_index = (len(self.bboxList) - 1) % len(COLORS)
                        if self.loadPredictionsLabels.get() == 1:
                            tmpId = self.canvas.create_rectangle(tmp[2], tmp[3], \
                                                                 tmp[4], tmp[5], \
                                                                 width=2, outline='red')
                            tempClassId = self.canvas.create_text(tmp[2], tmp[3], text=str(tmp[0]) + " %" + str(
                                "%.2f" % round((100 * tmp[1]), 2)),
                                                                  font="Calibri, 12", fill="blue", anchor="sw")
                        self.predList.append((tmp[2], tmp[3], tmp[4], tmp[5], tmp[0], tmp[1], tmpId, tempClassId))
                        self.predIDList.append(tmpId)

    def loadImagePredictionsFunction(self):
        tmpId = None
        tempClassId = None
        if self.selected_image_prediction_text != "":
            self.imagePredictionFileName = self.svSourcePath + "/images/predictions/" + self.selected_image_prediction_text + "/" + self.img_source_text + "/" + self.img_source_text + "_fr1.txt"
            if os.path.exists(self.imagePredictionFileName):
                with open(self.imagePredictionFileName) as f:
                    for (i, line) in enumerate(f):
                        tmp = line.split()
                        tmp[1] = tmp[1].replace(":", "")
                        tmp[1] = tmp[1].replace("(", "")
                        tmp[1] = tmp[1].replace(")", "")
                        tmp[1] = float(tmp[1])
                        tmp[4] = float((float(tmp[4]) + float(tmp[2])) * self.vidRatio)
                        tmp[5] = float((float(tmp[5]) + float(tmp[3])) * self.vidRatio)
                        tmp[2] = float(float(tmp[2]) * self.vidRatio)
                        tmp[3] = float(float(tmp[3]) * self.vidRatio)
                        # color_index = (len(self.bboxList) - 1) % len(COLORS)
                        if self.loadPredictionsLabels.get() == 1:
                            tmpId = self.canvas.create_rectangle(tmp[2], tmp[3], \
                                                                 tmp[4], tmp[5], \
                                                                 width=2, outline='red')
                            tempClassId = self.canvas.create_text(tmp[2], tmp[3], text=str(tmp[0]) + " %" + str(
                                "%.2f" % round((100 * tmp[1]), 2)),
                                                                  font="Calibri, 12", fill="blue", anchor="sw")
                        self.predList.append((tmp[2], tmp[3], tmp[4], tmp[5], tmp[0], tmp[1], tmpId, tempClassId))
                        self.predIDList.append(tmpId)

    def loadZtackLabels(self):
        tmpId = None
        tmpClassId = None
        if os.path.exists(self.zstackLabelFileName):
            with open(self.zstackLabelFileName) as f:
                for (i, line) in enumerate(f):
                    tmp = line.split()
                    tmp[0] = tmp[0].replace(":", "")
                    tmp[3] = float(float(tmp[3]) * self.vidRatio)
                    tmp[4] = float(float(tmp[4]) * self.vidRatio)
                    tmp[1] = float(float(tmp[1]) * self.vidRatio)
                    tmp[2] = float(float(tmp[2]) * self.vidRatio)
                    # color_index = (len(self.bboxList) - 1) % len(COLORS)
                    tmpId = self.canvas.create_rectangle(tmp[1], tmp[2], \
                                                         tmp[3], tmp[4], \
                                                         width=3, outline=COLORS[(int(tmp[0]) + 1) % len(COLORS)],
                                                         dash=(10, 10), state='hidden')
                    tmpClassId = self.canvas.create_text(tmp[1], tmp[2], text=str(tmp[0]),
                                                         font="Calibri, 12", fill="purple", anchor="sw", state='hidden')
                    if self.showZstackLabels.get() == 1:
                        self.canvas.itemconfigure(tmpId, state='normal')
                        self.canvas.itemconfigure(tmpClassId, state='normal')
                        self.canvas.tag_raise(tmpId)
                        self.canvas.tag_raise(tmpClassId)
                    self.zstackLabelBboxList.append(
                        (tmp[1], tmp[2], tmp[3], tmp[4], tmp[0], tmpId, "label", tmpClassId))
        self.loadZstackPredictionLabel()

    def loadZstackPredictionLabel(self):
        tmpId = None
        tempClassId = None
        if self.selected_prediction_text != "":
            if os.path.exists(self.zstackPredictionLabelFileName):
                with open(self.zstackPredictionLabelFileName) as f:
                    for (i, line) in enumerate(f):
                        tmp = line.split()
                        tmp[1] = tmp[1].replace(":", "")
                        tmp[1] = tmp[1].replace("(", "")
                        tmp[1] = tmp[1].replace(")", "")
                        tmp[1] = float(tmp[1])
                        tmp[4] = float(float(tmp[4]) * self.vidRatio)
                        tmp[5] = float(float(tmp[5]) * self.vidRatio)
                        tmp[2] = float(float(tmp[2]) * self.vidRatio)
                        tmp[3] = float(float(tmp[3]) * self.vidRatio)
                        # color_index = (len(self.bboxList) - 1) % len(COLORS)
                        tmpId = self.canvas.create_rectangle(tmp[2], tmp[3], \
                                                             tmp[4], tmp[5], \
                                                             width=3, outline='red', dash=(10, 10), state='hidden')
                        tempClassId = self.canvas.create_text(tmp[2], tmp[3], text=str(tmp[0]) + " %" + str(
                            "%.2f" % round((100 * tmp[1]), 2)),
                                                              font="Calibri, 12", fill="blue", anchor="sw",
                                                              state='hidden')
                        if self.showZstackPredictionLabels.get() == 1:
                            self.canvas.itemconfigure(tmpId, state='normal')
                            self.canvas.itemconfigure(tempClassId, state='normal')
                            self.canvas.tag_raise(tmpId)
                            self.canvas.tag_raise(tempClassId)
                        self.zstackPredictionLabelBboxList.append(
                            (tmp[2], tmp[3], tmp[4], tmp[5], tmp[0], tmp[1], tmpId, tempClassId))

    def update(self):
        id = self.tabControl.select()
        if self.tabControl.tab(id, "text") == "Tab 1":
            if self.play:
                # Get a frame from the video source only if the video is supposed to play
                ret, frame = self.vid.get_frame()
                if ret:
                    self.photo = PIL.ImageTk.PhotoImage(
                        image=PIL.Image.fromarray(frame).resize(
                            (int(self.vid.width * self.vidRatio), int(self.vid.height * self.vidRatio)), Image.NEAREST))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                    self.frame = self.frame + 1
                    self.status["text"] = (self.frame + 1, "/", self.frames)
                    self.clearBBox()
                    self.clearPredictions()
                    self.loadLabels()
                    self.loadPolygons()
                    for bbox in self.zstackLabelBboxList:
                        if self.showZstackLabels.get() == 1:
                            self.canvas.tag_raise(bbox[5])
                            self.canvas.tag_raise(bbox[7])
                    for predBox in self.zstackPredictionLabelBboxList:
                        if self.showZstackPredictionLabels.get() == 1:
                            self.canvas.tag_raise(predBox[6])
                            self.canvas.tag_raise(predBox[7])
                else:
                    self.play = False
            if self.vid:
                if self.canvasAreaHeight + 4 != self.canvasArea.winfo_height() or self.canvasAreaWidth + 4 != self.canvasArea.winfo_width():
                    self.getRatio(self.vid.width, self.vid.height)
                    self.canvas.config(width=int(self.vid.width * self.vidRatio),
                                       height=int(self.vid.height * self.vidRatio))
                    self.canvasAreaWidth = self.canvasArea.winfo_width()
                    self.canvasAreaHeight = self.canvasArea.winfo_height()
        elif self.tabControl.tab(id, "text") == "Tab 2":
            if self.img is not None:
                if self.canvasAreaHeight + 4 != self.canvasArea.winfo_height() or self.canvasAreaWidth + 4 != self.canvasArea.winfo_width():
                    self.getRatio(self.imgWidth, self.imgHeight)
                    self.canvas.config(width=int(self.imgWidth * self.vidRatio),
                                       height=int(self.imgHeight * self.vidRatio))
                    self.canvasAreaWidth = self.canvasArea.winfo_width()
                    self.canvasAreaHeight = self.canvasArea.winfo_height()
        self.master.after(self.delay, self.update)

    # *** Functions for Tab 2 ***

    # Calculates the centroid (euclidean) distances of each ground truth bbox vs prediction bboxes
    # Output is of shape MXN (M = number of gt bboxes, N = number of pred bboxes)
    # centroid distances = np.linalg.norm(gt_centers-pred_centers))
    def centroid_distances(self, bbox_gt, bbox_pred):

        # calculate centroids (X,Y) of ground truth (gt) bboxes & predicted bboxes (pred)
        bboxes_gt_centers = structures.Boxes(torch.Tensor(bbox_gt)).get_centers()
        bboxes_pred_centers = structures.Boxes(torch.Tensor(bbox_pred)).get_centers()

        # empty array to save distances between cetroids of gt bbox vs pred bbox
        center_dist = np.empty((bboxes_gt_centers.shape[0], 0), float)

        # Calculating the euclidean distance and storing into the 2D empty array
        for pred_centers in bboxes_pred_centers:
            l2_dist = np.array([])
            for gt_centers in bboxes_gt_centers:
                euc_dist = round(float(np.linalg.norm(gt_centers - pred_centers)), 3)
                l2_dist = np.append(l2_dist, euc_dist)
            l2_dist = np.expand_dims(l2_dist, axis=1)
            center_dist = np.append(center_dist, l2_dist, axis=1)

        return center_dist

    # Calculates the comparative areas between ground truth bboxes vs prediction bboxes
    # Output is of shape MXN (M = number of gt bboxes, N = number of pred bboxes)
    # Comparative areas equation = (abs(GT_bbox - Pred_bbox)/max(GT_bbox,Pred_bbox))
    # The results of comparative areas range between [0-1]:
    # comparative areas = 0 --> the GT_bbox_area = Pred_bbox_area
    # comparative areas > 0 and < 0.5 --> either the GT_bbox_area or Pred_bbox_area is up to 2 times larger than the other bbox (value of 0.5 means 2 times larger)
    # comparative areas > 0.5 --> either one of the bbox is more than 2 times larger than the other bbox
    # To know how large without visualizing, use the formula given: scale = 1/(1-comparative_area) --> gives the scale of how large one bbox to the other comparatively
    # Example: 1/(1-0.5) = 2 --> One bbox is 2 times larger than the other , 1/(1-0.75) = 4 --> One bbox is 4 times larger than the other
    def comparative_areas(self, bbox_gt, bbox_pred):

        # calculate areas of ground truth (gt) bboxes & predicted bboxes (pred)
        bboxes_gt_areas = structures.Boxes(torch.Tensor(bbox_gt)).area()
        bboxes_pred_areas = structures.Boxes(torch.Tensor(bbox_pred)).area()

        # empty array to save distances between cetroids of gt bbox vs pred bbox
        comparative_areas = np.empty((bboxes_gt_areas.shape[0], 0), float)

        # Calculating the comparative areas and storing into the 2D empty array
        for pred_areas in bboxes_pred_areas:

            areas = np.array([])
            for gt_areas in bboxes_gt_areas:
                pairwise_areas = round(float(abs(gt_areas - pred_areas) / max(gt_areas, pred_areas)), 3)
                areas = np.append(areas, pairwise_areas)

            areas = np.expand_dims(areas, axis=1)
            comparative_areas = np.append(comparative_areas, areas, axis=1)

        return comparative_areas

    def category_check_box(self, Annot_categories, Pred_categories):

        # Get prediction and annotation categories for building the checkbox
        # Pred_bboxes, Pred_categories, Pred_scores = get_pred_bbox_props(outputs) # outputs instead of pred_bboxes
        # Annot_bboxes, Annot_categories = get_annot_bbox_props(dictionary)

        # empty array to save distances between cetroids of gt bbox vs pred bbox
        check_box = np.empty((Annot_categories.shape[0], 0), int)

        # Calculating the comparative areas and storing into the 2D empty array
        for categories_pb in Pred_categories:

            check = np.array([])
            for categories_gt in Annot_categories:
                if categories_pb == categories_gt:
                    pairwise_check = 1
                else:
                    pairwise_check = 0
                check = np.append(check, pairwise_check)

            check = np.expand_dims(check, axis=1)
            check_box = np.append(check_box, check, axis=1)

        return check_box

    def track_TP_FP_FN(self, Pred_bboxes, Pred_categories, Annot_bboxes, Annot_categories, custom_method=True):
        # only use in this condition --> if bboxes_gt.tensor.size > 0 & bboxes_pred.size > 0:

        ## Helps find TP & FN (checks rows --> annotations) ##
        # GT in rows and PB in columns
        if (Annot_bboxes.shape[0] == 0) and (Pred_bboxes.shape[0] == 0):
            TP_indices, FN_indices, Pred_TP_indices, FP_indices, FP_to_TP = np.asarray([]), np.asarray([]), np.asarray(
                []), np.asarray([]), np.asarray([])
        elif (Annot_bboxes.shape[0] != 0) and (Pred_bboxes.shape[0] == 0):  # Only Annot exists
            TP_indices, FN_indices, Pred_TP_indices, FP_indices, FP_to_TP = np.asarray([]), np.arange(0,
                                                                                                      Annot_bboxes.shape[
                                                                                                          0]), np.asarray(
                []), np.asarray([]), np.asarray([])
        elif (Annot_bboxes.shape[0] == 0) and (Pred_bboxes.shape[0] != 0):  # Only Pred exists
            TP_indices, FN_indices, Pred_TP_indices, FP_indices, FP_to_TP = np.asarray([]), np.asarray([]), np.asarray(
                []), np.arange(0, Pred_bboxes.shape[0]), np.asarray([])
        elif (Annot_bboxes.shape[0] != 0) and (Pred_bboxes.shape[0] != 0):  # Both Pred and Annot exists
            # Calculate IOUs
            bboxes_gt = structures.Boxes(torch.Tensor(Annot_bboxes))
            bboxes_pred = structures.Boxes(torch.Tensor(Pred_bboxes))
            IOUs = structures.pairwise_iou(bboxes_gt, bboxes_pred)

            # Calculate category check box
            categorical_check_box = self.category_check_box(Annot_categories, Pred_categories)
            conditional_indices = np.nonzero((categorical_check_box == 1) & (IOUs.numpy() >= 0.5))

            # Unique Annot TP_indices - # Initial TP indices:
            TP_indices = np.unique(conditional_indices[0])

            # Maximum column for each row of TP_indices (Pred_TP_indices)
            Pred_TP_indices = np.asarray([])
            for eachrow in TP_indices:
                Pred_TP_indices = np.append(Pred_TP_indices, np.argmax(IOUs.numpy()[eachrow, :], axis=0))

            # Initial Pred TP indices:
            Pred_TP_indices = np.asarray(Pred_TP_indices, dtype=int)

            # total pred bboxes
            total_cols = np.arange(0, Pred_bboxes.shape[0])
            # Initial Pred FP indices:
            FP_indices = np.delete(total_cols, Pred_TP_indices.tolist())

            # total gt bboxes
            total_rows = np.arange(0, Annot_bboxes.shape[0])
            # Initial FN indices:
            FN_indices = np.delete(total_rows, TP_indices)

            # if custom method is not used returns an empty FP_to_TP
            FP_to_TP = np.asarray([])

            # If at least one GT bbox is FN and at least one Pred bbox is FP =>
            # check for 4 new conditions --> (IOAs >= 0.5), (cent_dist < 30), (comp_area > 0 & <= 0.6), (categorical match = True) --> if conditions are satisfied change FN & FP to TP
            if (custom_method == True) & (FP_indices.size > 0) & (FN_indices.size > 0):
                # print('Using the custom method!') - uncomment to indicate custom method is being utilized

                # original GT bboxes that are FN
                check_GT_bbox = np.asarray(bboxes_gt.tensor)[FN_indices.tolist()]  # gt bboxes
                bboxes_gt_new = structures.Boxes(torch.Tensor(check_GT_bbox))
                check_GT_categories = Annot_categories[FN_indices.tolist()]  # gt categories
                # original PB bboxes that are FP
                check_PB_bbox = np.asarray(bboxes_pred.tensor)[FP_indices.tolist()]  # pb bboxes
                bboxes_pred_new = structures.Boxes(torch.Tensor(check_PB_bbox))
                check_PB_categories = Pred_categories[FP_indices.tolist()]  # pb categories

                # Calculating new IOAs (not IOUs) & other conditions based on FN & FP indices
                IOAs_new = structures.pairwise_ioa(bboxes_gt_new, bboxes_pred_new)
                cent_dist = self.centroid_distances(check_GT_bbox, check_PB_bbox)
                comp_area = self.comparative_areas(check_GT_bbox, check_PB_bbox)
                categorical_check_box = self.category_check_box(check_GT_categories,
                                                                check_PB_categories)  # check categorical match (class match)

                # The columns and rows which meet the new 3 conditions --> (IOAs >= 0.5), (cent_dist < 30), (comp_area > 0 & <= 0.6)
                myindices = np.nonzero((categorical_check_box == 1) & (IOAs_new.numpy() >= 0.5) & (cent_dist < 30) & (
                        (comp_area > 0) & (comp_area <= 0.8)))
                # myindices --> myindices[:,0] are for annotations --> FN, myindices[:,1] are for predictions --> FP

                # FP predictions that turned TP
                FP_to_TP = FP_indices[myindices[1].tolist()]

                # Remaining FP indices after fixing for TP:
                FP_indices = np.delete(FP_indices, myindices[1].tolist())

                # FN annotations that turned TP
                FN_to_TP = FN_indices[myindices[0].tolist()]

                # Remaining FN indices after fixing for TP:
                FN_indices = np.delete(FN_indices, myindices[0].tolist())

                # New TP annotations
                TP_indices = np.sort(np.append(TP_indices, FN_to_TP))

                # New TP predictions
                Pred_TP_indices = np.sort(np.append(Pred_TP_indices, FP_to_TP))
        else:
            print('\nEncountered an unexpected behavior!\n')

        return TP_indices, FN_indices, Pred_TP_indices, FP_indices, FP_to_TP

    def DisplayCustomstats(self):

        # Get Json paths for both COCO annotations and COCO predictions
        Json_annot_path = os.path.join(self.svSourcePath, "Videos", "label_files", 'COCO_annotations.json')
        Json_pred_path = os.path.join(self.svSourcePath, "Videos", "predictions", self.selected_prediction_text,
                                      'COCO_predictions.json')

        # Load COCO annotations
        f = open(Json_annot_path)
        annot_data = json.load(f)
        f.close()

        # Load COCO predictions
        f = open(Json_pred_path)
        pred_data = json.load(f)
        f.close()

        # Test
        annot_images = pd.DataFrame(annot_data['images'])
        image_path = annot_images['filename']  # Not used - Can use if needed to get frame names of image_id (annotated)

        # Convert COCO annotations to a dataframe
        annots = annot_data['annotations']
        annots_df = pd.DataFrame(annots)

        # Find unique categories to loop through and find all AP50_custom values
        self.unique_category_ids = annots_df.category_id.unique()

        # Use images dataframe from COCO annotations and extract full image_ids
        image_id = list(range(1, len(annot_data['images']) + 1))

        # Convert COCO predictions to a dataframe
        preds_df = pd.DataFrame(pred_data)
        preds_df['TP'] = 0  # create new column of TP with zeros inside
        preds_df['FP'] = 0  # create new column of FP with zeros inside

        # Initialize an empty list to save all AP50_custom values per category
        AP_per_category = []
        self.PR_curves_per_category = []

        # Process each unique category separately
        for category in self.unique_category_ids:
            new_annots_df = annots_df.loc[annots_df['category_id'] == category]
            new_preds_df = preds_df.loc[preds_df['category_id'] == category]
            total_annotations = len(new_annots_df)

            # If annotations or predictions exist
            if (len(new_annots_df) > 0):

                for index in image_id:  # image_id: index = image_id

                    if index in new_annots_df['image_id'].values:
                        current_annots = new_annots_df.loc[new_annots_df['image_id'] == index]
                        current_annots_bboxes = np.array([np.array(i) for i in current_annots['bbox'].values])
                        current_annots_category = current_annots['category_id'].values

                    elif index not in new_annots_df['image_id'].values:
                        current_annots_bboxes = np.array([np.array([]), np.array([]), np.array([]), np.array([])]).T
                        current_annots_category = np.array([np.array([]), np.array([]), np.array([]), np.array([])]).T

                    if index in new_preds_df['image_id'].values:
                        current_preds = new_preds_df[new_preds_df['image_id'] == index]
                        current_preds_bboxes = np.array([np.array(i) for i in current_preds['bbox'].values])
                        current_preds_scores = current_preds['score'].values
                        current_preds_category = current_preds['category_id'].values

                    elif index not in new_preds_df['image_id'].values:
                        current_preds_bboxes = np.array([np.array([]), np.array([]), np.array([]), np.array([])]).T
                        current_preds_scores = np.array([np.array([]), np.array([]), np.array([]), np.array([])]).T
                        current_preds_category = np.array([np.array([]), np.array([]), np.array([]), np.array([])]).T

                    # Track TP, FN, Pred_TP, FP, FP_to_TP
                    TP_indices, FN_indices, Pred_TP_indices, FP_indices, FP_to_TP = self.track_TP_FP_FN(
                        current_preds_bboxes, current_preds_category, current_annots_bboxes, current_annots_category,
                        custom_method=True)

                    # If there are predictions in given image_id append to df
                    if index in new_preds_df['image_id'].values:
                        TP_bboxes = current_preds_bboxes[Pred_TP_indices.tolist()]  # find TP bboxes
                        TP_bboxes.tolist()
                        FP_bboxes = current_preds_bboxes[FP_indices.tolist()]  # find FP bboxes
                        FP_bboxes.tolist()

                        if len(TP_bboxes) > 0:  # if TP bboxes is not empty
                            for i in TP_bboxes:
                                matching_index = new_preds_df.index[
                                    new_preds_df['bbox'].apply(lambda x: x == i.tolist())].tolist()
                                new_preds_df['TP'].iloc[matching_index] = 1

                        if len(FP_bboxes) > 0:
                            for i in FP_bboxes:
                                matching_index = new_preds_df.index[
                                    new_preds_df['bbox'].apply(lambda x: x == i.tolist())].tolist()
                                new_preds_df['FP'].iloc[matching_index] = 1

                # Score sorted dataframe
                preds_df_copy = new_preds_df.copy()
                preds_df_sorted = preds_df_copy.sort_values(by='score', ascending=False)

                # Create empty columns to avoid warning
                preds_df_sorted['CUMSUM_TP'] = " "
                preds_df_sorted['CUMSUM_FP'] = " "
                preds_df_sorted['Precision'] = " "
                preds_df_sorted['Recall'] = " "

                # Cumulative Sum of TP and FP
                preds_df_sorted['CUMSUM_TP'] = preds_df_sorted['TP'].cumsum()
                preds_df_sorted['CUMSUM_FP'] = preds_df_sorted['FP'].cumsum()
                preds_df_sorted['Precision'] = preds_df_sorted['CUMSUM_TP'] / (
                        preds_df_sorted['CUMSUM_TP'] + preds_df_sorted['CUMSUM_FP'])
                preds_df_sorted['Recall'] = preds_df_sorted['CUMSUM_TP'] / len(new_annots_df)

                # Append 2 extra rows to dataframe for recall = 0 (starting point) and precision = 0 (end point) to calculate interpolated-precision
                start_row = []
                start_row.insert(0, {'image_id': 'Start', 'category_id': preds_df_sorted['category_id'].iloc[0],
                                     'bbox': [float(0), float(0), float(0), float(0)],
                                     'score': preds_df_sorted['score'].iloc[0], 'TP': preds_df_sorted['TP'].iloc[0],
                                     'FP': preds_df_sorted['FP'].iloc[0],
                                     'CUMSUM_TP': preds_df_sorted['CUMSUM_TP'].iloc[0],
                                     'CUMSUM_FP': preds_df_sorted['CUMSUM_FP'].iloc[0], 'Precision': float(0),
                                     'Recall': float(0)})
                preds_df_sorted = pd.concat([pd.DataFrame(start_row), preds_df_sorted], ignore_index=True)

                df_end = {'image_id': 'End', 'category_id': preds_df_sorted['category_id'].iloc[-1],
                          'bbox': [float(0), float(0), float(0), float(0)], 'score': preds_df_sorted['score'].iloc[-1],
                          'TP': preds_df_sorted['TP'].iloc[-1], 'FP': preds_df_sorted['FP'].iloc[-1],
                          'CUMSUM_TP': preds_df_sorted['CUMSUM_TP'].iloc[-1],
                          'CUMSUM_FP': preds_df_sorted['CUMSUM_FP'].iloc[-1], 'Precision': float(0),
                          'Recall': preds_df_sorted['Recall'].iloc[-1]}
                preds_df_sorted = preds_df_sorted.append(df_end, ignore_index=True)

                # Create Interpolated Precision column
                preds_df_sorted['Interp_precision'] = preds_df_sorted['Precision']

                for j in range(preds_df_sorted['Interp_precision'].size - 1, 0, -1):
                    preds_df_sorted['Interp_precision'].iloc[j - 1] = np.maximum(
                        preds_df_sorted['Interp_precision'].iloc[j - 1], preds_df_sorted['Interp_precision'].iloc[j])

                Interp_precision = np.asarray(preds_df_sorted['Interp_precision'].tolist())
                Recall = np.asarray(preds_df_sorted['Recall'].tolist())

                k = np.where(Recall[1:] != Recall[:-1])[
                    0]  # Finds all points that has change (steps down - individual rectangles)

                # Calculate area of each rectangle and add them all
                AP50_custom = np.sum((Recall[k + 1] - Recall[k]) * Interp_precision[k + 1])

                # Append each AP50_custom per category
                AP_per_category.append(AP50_custom)
                # Append each PR_curve per category
                self.PR_curves_per_category.append(preds_df_sorted)

            else:  # if there are no predicitions for the given annotated category
                # When there are no predictions on that category - creates a AP50_Custom = 0 so that the specific category that was not predicted represents the mAP well
                AP50_custom = 0
                # Create PR_curves to display 0 precision in the graph
                column_names = ['image_id', 'category_id', 'bbox', 'score', 'TP', 'FP', 'CUMSUM_TP', 'CUMSUM_FP',
                                'Precision', 'Recall', 'Interp_precision']
                preds_df_sorted = pd.DataFrame(columns=column_names)
                preds_df_sorted['image_id'] = np.zeros(11)
                preds_df_sorted['category_id'] = np.zeros(11)
                preds_df_sorted['bbox'] = np.zeros(11)
                preds_df_sorted['score'] = np.zeros(11)
                preds_df_sorted['TP'] = np.zeros(11)
                preds_df_sorted['FP'] = np.zeros(11)
                preds_df_sorted['CUMSUM_TP'] = np.zeros(11)
                preds_df_sorted['CUMSUM_FP'] = np.zeros(11)
                preds_df_sorted['Precision'] = np.zeros(11)
                preds_df_sorted['Recall'] = np.arange(0.0, 1.1, 0.1)
                preds_df_sorted['Interp_precision'] = np.zeros(11)

                # Append each AP50_custom per category
                AP_per_category.append(AP50_custom)
                # Append each PR_curve per category
                self.PR_curves_per_category.append(preds_df_sorted)

        # Calculate the custom mean AP for all categories (mAP_custom)
        mAP_custom = mean(AP_per_category)
        # print(mAP_custom)

        # Take the mAP_custom value and display it at AP50_custom
        self.AP50cumstomstats = f'{mAP_custom:.3}'

        # Get the entry values on app screen for custom stats display
        AP50customvalues = self.Display_CUSTOM_stats.get()
        # If the entries are empty, display stats
        if not AP50customvalues:
            self.Display_CUSTOM_stats.insert(0, self.AP50cumstomstats)
        # elif the entries are not empty, first clear entries and then display stats
        elif AP50customvalues:
            self.Display_CUSTOM_stats.delete(0, END)
            self.Display_CUSTOM_stats.insert(0, self.AP50cumstomstats)

    def Customstatsplot(self):

        # Plot parameters
        plt.rcParams["figure.figsize"] = [8, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 13
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1.25)
        plt.ylim(0, 1.25)
        plt.title('Precision-Recall Curve-AP50_custom')

        for index in range(0, len(self.unique_category_ids)):  # get unique category ids later within app
            PR_plot = self.PR_curves_per_category[index]  # the 0 needs to change automatically
            ax = plt.gca()
            PR_plot.plot(kind='line', x='Recall', y='Interp_precision', linewidth=2, ax=ax, )

        plt.legend(list(self.unique_category_ids));

        # To display plot on canvas in self.tabs['PAGE 3']
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        img = Image.open(img_buf)

        # # Convert to a displayable image
        self.plotimage_2 = PIL.ImageTk.PhotoImage(image=img)
        self.canvas_3.create_image(self.canvas_3.winfo_width() / 2, self.canvas_3.winfo_height() / 2,
                                   image=self.plotimage_2, anchor=CENTER)

        # # Close figure until next time to save memory otherwise, all created figures are saved in memory
        img_buf.close()
        plt.close()

    def Datastats(self):

        # Read all the categories from label_files
        Path_to_labels = os.path.join(self.svSourcePath, "Videos", "label_files")
        current_labels = os.listdir(Path_to_labels)
        video_list = self.video_source_text
        common_labels = list(set(video_list) & set(current_labels))
        total_labels = []  # for total labels

        # Find all common labels in .txt
        if common_labels:
            for i in range(0, len(common_labels)):
                Full_path_to_label = os.path.join(Path_to_labels, common_labels[i])  # path to labels

                # Check if both labels and preds for the same video exist and if true, list files and folders within the label
                if os.path.exists(Full_path_to_label):
                    current_labels = list(filter(lambda x: (x.endswith('.txt') & x.startswith(common_labels[i])),
                                                 os.listdir(
                                                     Full_path_to_label)))  # filter based on starting name and extension (.txt)
                total_labels.extend(current_labels)

        # common videos for labels
        videos = list(re.sub('_fr\d+.txt', '', elements) for elements in total_labels)

        # find categories in common labels
        self.label_category_stats = []  # empty to append categories later

        # find total number of categories
        for (index, frames) in enumerate(total_labels):
            current_label_path = os.path.join(Path_to_labels, videos[index], total_labels[index])
            if os.path.exists(current_label_path):  ## TODO: Add if .txt is not empty entirely
                with open(current_label_path) as f:
                    for (j, line) in enumerate(f):
                        if line:  # if line is not empty
                            tmp = line.split()
                            tmp[0] = tmp[0].replace(":", "")
                            self.label_category_stats.append(int(tmp[0]))

        # Start plotting category frequency for common labels (labels & videos)
        category_frequency = collections.Counter(self.label_category_stats)  # creates dictionary {id: count}
        max_category_id = max(self.label_category_stats)
        # min_category_id = min(self.categories)
        category_frequency = sorted(category_frequency.items())
        class_freq_xaxis = [i[0] for i in category_frequency]  # category_id
        class_freq_yaxis = [i[1] for i in category_frequency]  # frequency

        plt.rcParams["figure.figsize"] = [8, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 13
        plt.figure()

        plt.xticks(range(-1, max_category_id + 1))
        xlocs, xlabs = plt.xticks()

        for i, v in enumerate(class_freq_yaxis):
            plt.text(xlocs[i] + 1, v, str(v))

        plt.title('Categories-Bar plot based on {} videos'.format(len(self.video_source_text)))
        plt.xlabel('Categories')
        plt.ylabel('Frequency')
        plt.xlim(-1, max_category_id + 1)
        plt.bar(class_freq_xaxis, class_freq_yaxis, align='center')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)

        self.plotimage_2 = PIL.ImageTk.PhotoImage(image=img)
        self.canvas_3.create_image(self.canvas_3.winfo_width() / 2, self.canvas_3.winfo_height() / 2,
                                   image=self.plotimage_2, anchor=CENTER)

        img_buf.close()
        plt.close()

    def Get_AP_stats(self, GT_path=None, DT_path=None, annType=None):

        # If one of these input parameters are not entered, ask user to enter all of the parameters
        if (GT_path == None) | (DT_path == None) | (annType == None):
            print('Please enter all of the required inputs: GT path, DT Path, annType')
        # When all 3 parameters are given, run the program
        else:

            annType = ['segm', 'bbox', 'keypoints']
            annType = annType[
                1]  # specify type here (you might need to specify as segm or bbox depending on your annotations and predictions)
            print('Running demo for *%s* results.' % (annType))

            # Gt = labels, Dt = Detections
            cocoGt = COCO(GT_path)  # replace with yours (labels in coco format)
            cocoDt = cocoGt.loadRes(DT_path)  # replace with yours (instances or predictions in coco format)

            imgIds = sorted(cocoGt.getImgIds())  # Number of images (image IDs)

            # running COCO evaluation
            cocoEval = COCOeval(cocoGt, cocoDt, annType)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            os.system('cls')
            clear_output(wait=False)
            AP_statistics = cocoEval.stats  # same IOU statistics that are displayed (normally printed if not print is not cleared) returned in a numpy array

            ### Start drawing the plots (Five plots in total = 4 plots for each category and final 5th plot is mAP --> average over everything)
            categories = cocoEval.params.catIds  # category IDs
            categories_legend = [str(value) for value in categories]  # number of classes for graph legend
            recall = cocoEval.params.recThrs  # 101 Recall Threshold values (corresponds to precision values)
            AP50_95 = np.zeros([101, len(categories)])  # empty numpy array to save the results for the averaging
            category_length = [*range(0,
                                      len(categories))]  # category_length = Required to go over the categories while plotting P-R curves

            # Mean over IoU Thresholds [50:5:95] but, not classes
            for i in category_length:
                # Get 101 Precision values (There are a total of 101 values)
                precision = cocoEval.eval['precision'][:, :, i, 0,
                            2]  # cocoEval.eval['precision'][0,:,i,0,2] =  [AP50:0.05:0.95 (10 items), 101 Precision values, category ID = (number of classes), area_range = [0, 10000000000.0], maxdets = [100]
                new_mean = np.mean(precision, axis=0)
                new_mean = np.reshape(new_mean, [101, -1])
                AP50_95[:, [i]] = new_mean

            # Mean over both IoU Thresholds and classes (average over everything = mean average precision or mAP)
            mAP = np.reshape(np.mean(AP50_95, axis=1), [101, -1])

            myprecision = cocoEval.eval['precision'][:, :, :, 0, 2]
            AP50_95 = np.reshape(AP50_95, [1, 101, len(categories)])
            All_coco_PR_curves = np.append(myprecision, AP50_95, axis=0)

            return AP_statistics, category_length, All_coco_PR_curves, recall, categories_legend  # Returns all of the AP in a numpy array

    def track_plot(self):  # To track plot index and keep it within
        if not self.Next_PR_plot_Button:
            self.next_plot = 0  # initial plot = AP50
        elif (self.Next_PR_plot_Button) and (self.next_plot < 2):
            self.next_plot = self.next_plot + 1
        elif (self.Next_PR_plot_Button) and (self.next_plot == 2):
            self.next_plot = 0
        return self.next_plot

    def Next_PR_plot(self):

        self.next_plot = self.track_plot()

        required_list = [0, 5,
                         10]  # indices that already exist in cocoEval.eval['precision'] with shape = [10,101,number of classes,4,3]
        title = ['AP50', 'AP75', 'AP50:5:95']  # Titles of the 5 P-R graphs
        # Plot per category
        plt.rcParams["figure.figsize"] = [8, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 13
        # plt.figure(figsize=(10, 10),dpi=100) # creates a 1000X1000 figure 10X100 = 1000

        # current_plot_index
        current_index = required_list[self.next_plot]  # current plot index

        # Plot a new figure with given specs below for each IoU Threshold
        plt.figure()
        # plt.figure(figsize=(10, 10),dpi=100)
        plt.xlabel('Recall')
        plt.xlim([0, 1.25])
        plt.ylabel('Precision')
        plt.ylim([0, 1.25])
        plt.title('Precision-Recall Curve' + '-' + title[self.next_plot])

        # To plot AP50,AP75,AP50:95

        for j in self.category_length:
            precision_new = self.all_coco_PR_curves[current_index, :, j]
            plt.plot(self.recall.T, precision_new)
        plt.legend(self.categories_legend)
        # plt.show()
        # self.canvas_3 = Canvas for self.tabs['PAGE 3']

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        img = Image.open(img_buf)

        self.plotimage_2 = PIL.ImageTk.PhotoImage(image=img)
        self.canvas_3.create_image(self.canvas_3.winfo_width() / 2, self.canvas_3.winfo_height() / 2,
                                   image=self.plotimage_2, anchor=CENTER)

        # Already pressed the button once so now it's TRUE
        self.Next_PR_plot_Button = True  # Changed from False to True and stays TRUE for the rest of the time

        # Close figure until next time to save memory otherwise, all created figures are saved in memory
        img_buf.close()
        plt.close()

    def Create_COCO_JSON(self, Path_to_labels=None, Path_to_preds=None, video_list=None):

        # List of videos currently present within the app
        # video_list = ['0729N1-1R1', '0729N1-1R2','071321M1-8R3'] # Example list of videos ## Change later within app to self.video_source_text

        # Use these paths both for reading annotations & predictions and outputs for 2 COCO json files
        # Path_to_labels = r'C:\Users\ykocoglu\Videos\label_files' # Need this path ## Change later within app to self.svSourcePath + "/videos/label_files/
        # Path_to_preds = r'C:\Users\ykocoglu\Videos\predictions\model_final' # Need this path ## Change later within app to self.svSourcePath + /videos/predictions/ + (find model path)
        coco_annotation_output_file = os.path.join(Path_to_labels, 'COCO_annotations.json')
        coco_prediction_output_file = os.path.join(Path_to_preds, 'COCO_predictions.json')

        # If the file already exists, remove it
        if os.path.exists(coco_annotation_output_file):
            os.remove(coco_annotation_output_file)
        elif os.path.exists(coco_prediction_output_file):
            os.remove(coco_prediction_output_file)

        total_images = []  # empty list for saving all images

        # Find common videos & labels & predictions
        label_list = os.listdir(Path_to_labels)
        pred_list = os.listdir(Path_to_preds)
        common_list = list(set(label_list) & set(video_list) & set(pred_list))
        print(common_list)
        print(len(common_list))

        # if common list is not empty
        if common_list:
            # Get the total frames both in labels and predictions for the given video list and create total_images for image_id tracking
            for i in range(0, len(common_list)):

                Full_path_to_label = os.path.join(Path_to_labels, common_list[i])  # path to labels
                Full_path_to_preds = os.path.join(Path_to_preds, common_list[i])  # path to predictions

                # Check if both labels and preds for the same video exist and if true, list files and folders within the label
                if (os.path.exists(Full_path_to_label)) & (os.path.exists(Full_path_to_preds)):
                    current_labels = list(filter(lambda x: (x.endswith('.txt') & x.startswith(common_list[i])),
                                                 os.listdir(
                                                     Full_path_to_label)))  # filter based on starting name and extension (.txt)
                    current_preds = list(filter(lambda x: (x.endswith('.txt') & x.startswith(common_list[i])),
                                                os.listdir(
                                                    Full_path_to_preds)))  # filter based on starting name and extension (.txt)
                    current_labels_preds = sorted(set(current_labels + current_preds))
                elif (not os.path.exists(Full_path_to_label)) & (os.path.exists(Full_path_to_preds)):
                    current_labels = []
                    current_preds = list(filter(lambda x: (x.endswith('.txt') & x.startswith(common_list[i])),
                                                os.listdir(
                                                    Full_path_to_preds)))  # filter based on starting name and extension (.txt)
                    current_labels_preds = sorted(set(current_labels + current_preds))
                elif (os.path.exists(Full_path_to_label)) & (not os.path.exists(Full_path_to_preds)):
                    current_labels = list(filter(lambda x: (x.endswith('.txt') & x.startswith(common_list[i])),
                                                 os.listdir(
                                                     Full_path_to_label)))  # filter based on starting name and extension (.txt)
                    current_preds = []
                    current_labels_preds = sorted(set(current_labels + current_preds))
                elif (not os.path.exists(Full_path_to_label)) & (not os.path.exists(Full_path_to_preds)):
                    current_labels = []
                    current_preds = []
                    current_labels_preds = sorted(set(current_labels + current_preds))

                total_images.extend(current_labels_preds)  # total images in both labels and preds

            # Total Existing videos (if video does not exist, it's ignored and not listed)
            videos = list(re.sub('_fr\d+.txt', '', elements) for elements in
                          total_images)  # use for creating path of videos later

            width, height = [1002,
                             1002]  # change this if necessary later (not necessary as it's meant to be metadata only for now)
            coco_images = []  # coco_images needs to have all the image_ids
            coco_annotations = []
            categories = []
            count = 0
            coco_predictions = []  # format =  [{image_id: , category_id: , bbox: [] , score: }]

            for (index, frames) in enumerate(total_images):

                current_label_path = os.path.join(Path_to_labels, videos[index], total_images[index])
                current_coco_images_dict = {"id": index + 1, "width": width, "height": height,
                                            "filename": current_label_path}
                coco_images.append(current_coco_images_dict)
                current_pred_path = os.path.join(Path_to_preds, videos[index], total_images[index])
                # If the frame is labeled and it exists
                # Create coco_annotations and track image_ids and bbox_ids
                if os.path.exists(current_label_path):  ## TODO: Add if .txt is not empty entirely
                    with open(current_label_path) as f:
                        for (j, line) in enumerate(f):
                            if line:  # if line is not empty
                                tmp = line.split()
                                tmp[0] = tmp[0].replace(":", "")
                                tmp[3] = float(tmp[3]) + float(tmp[1])
                                tmp[4] = float(tmp[4]) + float(tmp[2])
                                tmp[1] = float(tmp[1])
                                tmp[2] = float(tmp[2])
                                current_bbox = torch.Tensor(
                                    np.reshape(np.asarray([float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])]),
                                               [1, 4]))
                                current_bbox_area = structures.Boxes(current_bbox).area()
                                current_bbox, *other = current_bbox.tolist()
                                current_bbox_area, *other = current_bbox_area.tolist()
                                count = count + 1
                                categories.append(int(tmp[0]))
                                current_annotations = {"id": count, "image_id": index + 1, "bbox": current_bbox,
                                                       "area": current_bbox_area, "iscrowd": 0, "category_id": int(
                                        tmp[0])}  # {"id":, "image_id":, "bbox":, "area":, "iscrowd":,"category_id":}
                                coco_annotations.append(current_annotations)
                # Create coco_predictions and track image_ids
                if os.path.exists(current_pred_path):
                    with open(current_pred_path) as folder:
                        for (k, pred_line) in enumerate(folder):
                            if pred_line:  # if line is not empty
                                tmp_pred = pred_line.split()
                                tmp_pred[0] = int(tmp_pred[0])  # category_id
                                tmp_pred[1] = float(
                                    tmp_pred[1].translate({ord(':'): None, ord('('): None, ord(')'): None, }))  # score
                                tmp_pred[4] = float(tmp_pred[4]) + float(tmp_pred[2])
                                tmp_pred[5] = float(tmp_pred[5]) + float(tmp_pred[3])
                                tmp_pred[2] = float(tmp_pred[2])
                                tmp_pred[3] = float(tmp_pred[3])
                                current_bbox_pred = [float(tmp_pred[2]), float(tmp_pred[3]), float(tmp_pred[4]),
                                                     float(tmp_pred[5])]  # bbox
                                current_predictions = {"image_id": index + 1, "category_id": tmp_pred[0],
                                                       "bbox": current_bbox_pred, "score": tmp_pred[
                                        1]}  # {"id":, "image_id":, "bbox":, "area":, "iscrowd":,"category_id":}
                                coco_predictions.append(current_predictions)

            categories_list = list(set(categories))
            self.categories = categories

            categories_unique = []
            for i in categories_list:
                current_category = {"id": categories_list[i]}
                categories_unique.append(current_category)

            info = {"date_created": str(datetime.datetime.now()),
                    "description": "COCO json format for COCO evaluation within the annotation app.",
                    }

            coco_dict = {"info": info,
                         "images": coco_images,
                         "categories": categories_unique,
                         "licenses": None,
                         "annotations": coco_annotations,

                         }

            with open(coco_annotation_output_file, 'w+') as outfile:
                json.dump(coco_dict, outfile)
            with open(coco_prediction_output_file, 'w+') as outfile:
                json.dump(coco_predictions, outfile)

            return self.categories

    # Converts labels and predictions to 2 JSON files (COCO_annotations and COCO_predictions)
    def ConvertJSON(self):

        # Required paths to labels, predictions (with currently selected model), and current video list
        Path_to_labels = os.path.join(self.svSourcePath, "Videos", "label_files")
        Path_to_preds = os.path.join(self.svSourcePath, "Videos", "predictions", self.selected_prediction_text)
        video_list = self.video_source_text

        self.Create_COCO_JSON(Path_to_labels=Path_to_labels, Path_to_preds=Path_to_preds, video_list=video_list)

    def DisplayCOCOstats(self):  # Displays COCO statistics

        GT_path = os.path.join(self.svSourcePath, "Videos", "label_files", "COCO_annotations.json")
        DT_path = os.path.join(self.svSourcePath, "Videos", "predictions", self.selected_prediction_text,
                               "COCO_predictions.json")
        annType = 1  # 1 for box, 0 for segmentation

        # self.cocostat = "None"
        self.COCOstats, self.category_length, self.all_coco_PR_curves, self.recall, self.categories_legend = self.Get_AP_stats(
            GT_path=GT_path, DT_path=DT_path, annType=annType)
        self.AP50stats = f'{self.COCOstats[1]:.3}'
        self.AP75stats = f'{self.COCOstats[2]:.3}'
        self.AP50_95stats = f'{self.COCOstats[0]:.3}'

        AP50 = self.Display_COCO_stats.get()
        AP75 = self.Display_COCO_stats2.get()
        AP50_95 = self.Display_COCO_stats3.get()
        # If the entries are empty, display stats
        if (not AP50) or (not AP75) or (not AP50_95):
            self.Display_COCO_stats.insert(0, self.AP50stats)
            self.Display_COCO_stats2.insert(0, self.AP75stats)
            self.Display_COCO_stats3.insert(0, self.AP50_95stats)
        # elif the entries are not empty, first clear entries and then display stats
        elif (AP50) or (AP75) or (AP50_95):
            self.Display_COCO_stats.delete(0, END)
            self.Display_COCO_stats2.delete(0, END)
            self.Display_COCO_stats3.delete(0, END)
            self.Display_COCO_stats.insert(0, self.AP50stats)
            self.Display_COCO_stats2.insert(0, self.AP75stats)
            self.Display_COCO_stats3.insert(0, self.AP50_95stats)

    def count_track_instances(self, Path=None, video_name=None):

        Prediction_video_path = os.path.join(Path, video_name)

        # Can be either predictions or labels
        if os.path.exists(Prediction_video_path):

            predictions = os.listdir(Prediction_video_path)

            if predictions:
                first, second = map(list, zip(*map(lambda x: x.split('_'), predictions)))
                numbers = [int(i) for i in re.findall(r'[0-9]+', ''.join(second))]  # finds frame numbers only

                # sorts predictions according to ascending frame numbers
                zipped_lists = zip(numbers, predictions)
                sorted_pairs = sorted(zipped_lists)
                tuples = zip(*sorted_pairs)
                list1, predictions = [list(tuple) for tuple in tuples]

            current_video = []
            if 'predictions' in Path:
                for i, frame in enumerate(predictions):
                    current_pred_path = os.path.join(Prediction_video_path, frame)

                    if os.path.exists(current_pred_path):
                        with open(current_pred_path) as folder:
                            for (k, pred_line) in enumerate(folder):
                                if pred_line:  # if line is not empty
                                    tmp_pred = pred_line.split()
                                    tmp_pred[0] = int(tmp_pred[0])  # category_id
                                    tmp_pred[1] = float(
                                        tmp_pred[1].translate(
                                            {ord(':'): None, ord('('): None, ord(')'): None, }))  # score
                                    tmp_pred[4] = float(tmp_pred[4]) + float(tmp_pred[2])
                                    tmp_pred[5] = float(tmp_pred[5]) + float(tmp_pred[3])
                                    tmp_pred[2] = float(tmp_pred[2])
                                    tmp_pred[3] = float(tmp_pred[3])
                                    current_bbox_pred = [float(tmp_pred[2]), float(tmp_pred[3]), float(tmp_pred[4]),
                                                         float(tmp_pred[5])]  # bbox
                                    current_predictions = {"frame_no": list1[i], "category": tmp_pred[0],
                                                           "score": tmp_pred[1], "bbox": current_bbox_pred}
                                    current_video.append(current_predictions)
            elif 'label_files' in Path:
                for i, frame in enumerate(predictions):
                    current_pred_path = os.path.join(Prediction_video_path, frame)

                    if os.path.exists(current_pred_path):
                        with open(current_pred_path) as folder:
                            for (k, pred_line) in enumerate(folder):
                                if pred_line:  # if line is not empty
                                    tmp_pred = pred_line.split()
                                    tmp_pred[0] = tmp_pred[0].replace(":", "")
                                    tmp_pred[0] = int(tmp_pred[0])  # category_id
                                    tmp_pred[3] = float(tmp_pred[3]) + float(tmp_pred[1])
                                    tmp_pred[4] = float(tmp_pred[4]) + float(tmp_pred[2])
                                    tmp_pred[1] = float(tmp_pred[1])
                                    tmp_pred[2] = float(tmp_pred[2])
                                    current_bbox_pred = [float(tmp_pred[1]), float(tmp_pred[2]), float(tmp_pred[3]),
                                                         float(tmp_pred[4])]  # bbox
                                    current_predictions = {"frame_no": list1[i], "category": tmp_pred[0],
                                                           "bbox": current_bbox_pred}
                                    current_video.append(current_predictions)

            nex_frame = []
            init_frame = list1[0]
            initial_frame_bool = [frames['frame_no'] == init_frame for frames in current_video]
            initial_frame = [current_video[x] for x in [i for i, x in enumerate(initial_frame_bool) if x]]
            initial_frame_boxes = torch.Tensor([frames['bbox'] for frames in initial_frame])

            for index in range(1, len(list1)):  # range(1,len(list1)):

                # Get next frames
                nex_frame = list1[index]
                next_frame_bool = [frames['frame_no'] == nex_frame for frames in current_video]
                next_frame = [current_video[x] for x in [i for i, x in enumerate(next_frame_bool) if
                                                         x]]  # find matching bboxes in given previous frame
                next_frame_bboxes = [frames['bbox'] for frames in next_frame]
                next_frame_boxes = torch.Tensor(next_frame_bboxes)

                # Calculate IOU between previous and next frame boxes
                IOU = structures.pairwise_iou(structures.Boxes(initial_frame_boxes), structures.Boxes(next_frame_boxes))

                # Check which indices need to be accumulated to from prev frame to the next frame
                conditional_indices = np.nonzero(IOU.numpy() < 0.5)

                # Unique Annot TP_indices - # Initial TP indices:
                indices = np.unique(conditional_indices[1])

                # Maximum column for each row of TP_indices (Pred_TP_indices)
                new_indices = np.asarray([])
                for eachcol in indices:
                    if np.max(IOU.numpy()[:, eachcol], axis=0) < 0.50:
                        old_indices = eachcol
                        new_indices = np.append(new_indices, old_indices)

                new_indices = new_indices.astype('int64')
                new_indices = torch.tensor(new_indices)
                selected_next_frame_boxes = torch.index_select(next_frame_boxes, 0, new_indices)

                initial_frame_boxes = torch.cat((initial_frame_boxes, selected_next_frame_boxes), 0)

        else:
            initial_frame_boxes = torch.tensor([0, 0, 0, 0])

        return initial_frame_boxes

    def save_zstack_Labels(self):

        # Get z-stack labels in loop
        Z_stack_label_path = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_labels"
        Label_path = self.svSourcePath + "/videos/label_files"
        video_list = [name for name in os.listdir(Label_path) if os.path.isdir(os.path.join(Label_path, name))]

        fake_class = int(0)  # fake class - for the time being since only a single class is present

        # If label path exists and not empty inside the folder
        for vid_index, video_name in enumerate(tqdm(video_list, desc='Saving Z-stack Labels', leave=False)):
            self.progressBar["text"] = (vid_index + 1, "/", len(video_list))
            label_list = os.listdir(os.path.join(Label_path, video_name))
            substrings = ['_fr', '.txt']  # check for substrings of files in given label folder (video_name)

            # If all the files inside label folder have '_fr' as substring and end with '.txt'
            if all(x in y for y in label_list for x in substrings) and (label_list != []):

                zstack_labels = self.count_track_instances(Path=Label_path,
                                                           video_name=video_name)  # get static bboxes (count)

                # Create a label folder and save text file inside it (for creating coco.json files)
                if not os.path.exists(os.path.join(Z_stack_label_path, video_name)):
                    os.makedirs(os.path.join(Z_stack_label_path, video_name))
                elif os.path.exists(os.path.join(Z_stack_label_path, video_name)):
                    shutil.rmtree(os.path.join(Z_stack_label_path, video_name))
                    os.makedirs(os.path.join(Z_stack_label_path, video_name))

                if zstack_labels != []:
                    zstacktextfile = os.path.join(Z_stack_label_path, video_name,
                                                  video_name + "_fr1.txt")  # z stack text file to write with fake frame number (_fr1) for coco_evaluation
                    with open(zstacktextfile, 'w') as f:
                        for bbox in zstack_labels:
                            if len(bbox) >= 4:
                                f.write("{}: {} {} {} {}\n".format(fake_class,
                                                                   float(bbox.tolist()[0]),
                                                                   float(bbox.tolist()[1]),
                                                                   float(bbox.tolist()[2]),
                                                                   float(bbox.tolist()[3])))
            root.update()

    def save_zstack_predictions(self):

        modelname = self.selected_prediction_text  # currently selected model
        Z_stack_pred_path = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_predictions/" + modelname

        score_reducer = 1e-12  # value used to order predictions with fake confidence scores
        fake_score = 1  # initial fake score
        fake_class = int(0)  # fake class - for the time being since only a single class is present

        Pred_path = self.svSourcePath + "/videos/predictions/" + modelname  # change to selected model from listbox (self.model_text or similar)

        if not os.path.exists(Pred_path):
            messagebox.showwarning(title="Predictions not found!",
                                   message="Predictions do not exist for selected model...Please detect required predictions for the selected model first & try again!")
        else:

            # if prediction (model) folder does not exist create it
            if not os.path.exists(Z_stack_pred_path):
                os.makedirs(Z_stack_pred_path)

            # Acquire video list to loop over
            video_list = [name for name in os.listdir(Pred_path) if os.path.isdir(os.path.join(Pred_path, name))]
            # tqdm(self.video_source, desc='Prediction Progress', leave=False)
            # If label path exists and not empty inside the folder
            for vid_index, video_name in enumerate(tqdm(video_list, desc='Saving Z-stack Predictions', leave=False)):
                self.progressBar["text"] = (vid_index + 1, "/", len(video_list))
                pred_list = os.listdir(os.path.join(Pred_path, video_name))
                substrings = ['_fr', '.txt']  # check for substrings of files in given label folder (video_name)

                # If all the files inside label folder have '_fr' as substring and end with '.txt' and pred_list is not empty (folder not empty)
                if (all(x in y for y in pred_list for x in substrings)) and (pred_list != []):

                    zstack_preds = self.count_track_instances(Path=Pred_path,
                                                              video_name=video_name)  # get static bboxes (count)

                    # Create a prediction folder and save text file inside it (for creating coco.json files)
                    if not os.path.exists(os.path.join(Z_stack_pred_path, video_name)):
                        os.makedirs(os.path.join(Z_stack_pred_path, video_name))
                    elif os.path.exists(os.path.join(Z_stack_pred_path, video_name)):
                        shutil.rmtree(os.path.join(Z_stack_pred_path, video_name))
                        os.makedirs(os.path.join(Z_stack_pred_path, video_name))

                    if zstack_preds != []:
                        zstacktextfile = os.path.join(Z_stack_pred_path, video_name,
                                                      video_name + "_fr1.txt")  # z stack text file to write with fake frame number (_fr1) for coco_evaluation
                        with open(zstacktextfile, 'w') as f:
                            for bbox in zstack_preds:
                                fake_score = float(fake_score - score_reducer)
                                if len(bbox) >= 4:
                                    f.write("{} ({}): {} {} {} {}\n".format(fake_class,
                                                                            fake_score,
                                                                            float(bbox.tolist()[0]),
                                                                            float(bbox.tolist()[1]),
                                                                            float(bbox.tolist()[2]),
                                                                            float(bbox.tolist()[3])))
                root.update()

    # Converts labels and predictions to 2 JSON files (COCO_annotations and COCO_predictions)
    def ConvertJSON_zstack(self):

        # Label and prediction paths for z-stack evaluation
        Z_stack_label_path = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_labels"
        modelname = self.selected_prediction_text  # currently selected model
        Z_stack_pred_path = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_predictions/" + modelname

        # Required paths to labels, predictions (with currently selected model), and current video list
        Path_to_labels = Z_stack_label_path
        Path_to_preds = Z_stack_pred_path
        video_list = self.video_source_text

        self.Create_COCO_JSON(Path_to_labels=Path_to_labels, Path_to_preds=Path_to_preds, video_list=video_list)

    def DisplayCOCOstats_zstack(self):  # Displays COCO statistics

        # Label and prediction paths for z-stack evaluation
        Z_stack_label_path = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_labels"
        modelname = self.selected_prediction_text  # currently selected model
        Z_stack_pred_path = self.svSourcePath + "/videos/Z_stack_eval/Z_stack_predictions/" + modelname

        GT_path = os.path.join(Z_stack_label_path, "COCO_annotations.json")
        DT_path = os.path.join(Z_stack_pred_path, "COCO_predictions.json")
        annType = 1  # 1 for box, 0 for segmentation

        # self.cocostat = "None"
        self.COCOstats_zstack, self.category_length_zstack, self.all_coco_PR_curves_zstack, self.recall_zstack, self.categories_legend_zstack = self.Get_AP_stats(
            GT_path=GT_path, DT_path=DT_path, annType=annType)
        self.AP50stats_zstack = f'{self.COCOstats_zstack[1]:.3}'
        self.AP75stats_zstack = f'{self.COCOstats_zstack[2]:.3}'
        self.AP50_95stats_zstack = f'{self.COCOstats_zstack[0]:.3}'

        AP50 = self.Display_COCO_zstats.get()
        AP75 = self.Display_COCO_zstats2.get()
        AP50_95 = self.Display_COCO_zstats3.get()
        # If the entries are empty, display stats
        if (not AP50) or (not AP75) or (not AP50_95):
            self.Display_COCO_zstats.insert(0, self.AP50stats_zstack)
            self.Display_COCO_zstats2.insert(0, self.AP75stats_zstack)
            self.Display_COCO_zstats3.insert(0, self.AP50_95stats_zstack)
        # elif the entries are not empty, first clear entries and then display stats
        elif (AP50) or (AP75) or (AP50_95):
            self.Display_COCO_zstats.delete(0, END)
            self.Display_COCO_zstats2.delete(0, END)
            self.Display_COCO_zstats3.delete(0, END)
            self.Display_COCO_zstats.insert(0, self.AP50stats_zstack)
            self.Display_COCO_zstats2.insert(0, self.AP75stats_zstack)
            self.Display_COCO_zstats3.insert(0, self.AP50_95stats_zstack)

    def track_plot_zstack(self):  # To track plot index and keep it within
        if not self.Next_PR_plot_Button_zstack:
            self.next_plot_zstack = 0  # initial plot = AP50
        elif (self.Next_PR_plot_Button_zstack) and (self.next_plot_zstack < 2):
            self.next_plot_zstack = self.next_plot_zstack + 1
        elif (self.Next_PR_plot_Button_zstack) and (self.next_plot_zstack == 2):
            self.next_plot_zstack = 0
        return self.next_plot_zstack

    def Next_PR_plot_zstack(self):

        self.next_plot_zstack = self.track_plot_zstack()

        required_list = [0, 5,
                         10]  # indices that already exist in cocoEval.eval['precision'] with shape = [10,101,number of classes,4,3]
        title = ['AP50', 'AP75', 'AP50:5:95']  # Titles of the 5 P-R graphs
        # Plot per category
        plt.rcParams["figure.figsize"] = [8, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 13
        # plt.figure(figsize=(10, 10),dpi=100) # creates a 1000X1000 figure 10X100 = 1000

        # current_plot_index
        current_index = required_list[self.next_plot_zstack]  # current plot index

        # Plot a new figure with given specs below for each IoU Threshold
        plt.figure()
        # plt.figure(figsize=(10, 10),dpi=100)
        plt.xlabel('Recall')
        plt.xlim([0, 1.25])
        plt.ylabel('Precision')
        plt.ylim([0, 1.25])
        plt.title('Precision-Recall Curve' + '-' + title[self.next_plot_zstack])

        # To plot AP50,AP75,AP50:95

        for j in self.category_length_zstack:
            precision_new = self.all_coco_PR_curves_zstack[current_index, :, j]
            plt.plot(self.recall_zstack.T, precision_new)
        plt.legend(self.categories_legend_zstack)
        # plt.show()
        # self.canvas_3 = Canvas for self.tabs['PAGE 3']

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        img = Image.open(img_buf)

        self.plotimage_2 = PIL.ImageTk.PhotoImage(image=img)
        self.canvas_3.create_image(self.canvas_3.winfo_width() / 2, self.canvas_3.winfo_height() / 2,
                                   image=self.plotimage_2, anchor=CENTER)

        # Already pressed the button once so now it's TRUE
        self.Next_PR_plot_Button_zstack = True  # Changed from False to True and stays TRUE for the rest of the time

        # Close figure until next time to save memory otherwise, all created figures are saved in memory
        img_buf.close()
        plt.close()


# *** Class for video frame visualization ***
class videoCapturer:
    def __init__(self, video_source):
        # Open the video source
        self.capturedVideo = cv2.VideoCapture(video_source)
        if not self.capturedVideo.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source values
        self.width = self.capturedVideo.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capturedVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.capturedVideo.get(cv2.CAP_PROP_FRAME_COUNT)

    def goto_frame(self, frame_no):
        """
        Go to specific frame
        """
        if self.capturedVideo.isOpened():
            self.capturedVideo.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # Set current frame
            ret, frame = self.capturedVideo.read()  # Retrieve frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)

    def get_frame(self):
        if self.capturedVideo.isOpened():
            ret, frame = self.capturedVideo.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)


# *** Main app loop ***

root = Tk()  # Creates a blank window
root.state("zoomed")
App(root)
root.update()
root.mainloop()  # Keep the GUI open until close button clicked

# checkPredictions method does not return predictions for very first frame
# https://github.com/xiaqunfeng/BBox-Label-Tool
# Code tries to load labels for lastframe_no+1. Last frame = 40, when clicking next frame at frame 40, it tries to load labels for
#    frame no 41.
# https://beapython.dev/2020/05/14/is-recursion-bad-in-python/

# Add Progress Bar To Predictions For All Videos!!!# Detectron2 utilities and libraries
