# Video-Labeler
This app allows user to label a video by frame or an image, so image processing can be run for predictions in the future.

Installation: It explains in details inside instructions.txt file.

There are 3 tabs. Tab 1 is used for to work with videos, Tab 2 is used for to work with images. Tab 3 is done by other person for to show statistics. But I am not sure about it myself.

You can insert multiple videos or images. Right listbox will show the list of videos or images. You delete selected or all of the items in the listbox. You select item by double clicking it, and it will be showed on canvas.
Videos or images will have different colors and background colors depending on if they are selected, if they haved bbox, polygons or predictions.
  Red background color: If item has no bbox or polygon saved.
  Yellow background color: It has only polygon saved.
  Orange background color: It has only bbox saved.
  White background color: It has bot bbox and polygon saved.
  Blue font color: Selected item.

Insert Bbox: You can activate it by clicking show labels checkbox. Show labels checkbox allows you to load bboxes that are saved and predicted. Insert Bbox allows you add bbox. You must hold left mouse button, and when you release, it finishes drawing it. You can move, modify the bbox.
Insert Polygon: You can activate it by clicking show polygons checkbox. Show labels checkbox allows you to load polygon that are saved and predicted. Insert Polygon allows you add Polygon. When you right mouse button, it finishes drawing polygon and connects the first and last inserted points. Delete button will delete the last point that was added. Esc button will cancel drawing polygon. You can move, modify the polygon.
Insert Polygon Point: When you select a polygon, it gets activated. It allows you to add extra point to selected polygon line.
Delete Polygon Point: When you select a polygon, it gets activated. It allows you to delete polygon point.

Left Side: 
  The number 0 is for class. You get the class from a text file "class.txt". The colors are switching in between ['blue', 'pink', 'cyan', 'green', 'black', 'orange'] colors.
  When you click confirm class, your next bbox or polygon will be will be in that class. If you selected an existing polygon or bbox, it will change the class of that bbox or polygon.
  List box will show the list of models you have. It exists in videos/models and images/models folder. By double clicking the model, you activate it. 
  Drop box allows you select between, selected frame, selected video and all the inserted videos options. These options will run prediction when you click detect predictions. Detected predictions will be in red color and can't be modified. They will be saved to [images or videos]/predictions/[bbox or polygon]. Predictions can be converted to normal bbox or polygons. You convert it by selection or all at once.
  Load prediction checkbox will show prediction if it exists on current frame or image.
  Show polygon bbox, will draw a bbox around polygons.
  Zstack labels and predictions are labels that exists and same for one video from beggining frame to end frame.
  By clicking prev, next, play/stop, reset video buttons, you save the bboxes and polygons you drew.

Bbox save:[videos or images]/label_files/[name of the video or image]. Format: "class: x1, y1, x2-x1, y2-y1"
Predictions save:[videos or images]/predictions/[name of the video or image]. Format: "class: polygon cover points (x1, y1, x2, y2), points"

![image](https://user-images.githubusercontent.com/33734353/229110744-3c81ad43-5030-4547-8103-004001259b60.png)

Example of bounding boxes detected using the Faste R-CNN model. In the case below all the boxes were True positives (TP). The confidence score is given on the top left side of the box. 
![image](https://github.com/yavuzck132/Video-Labeler/blob/master/1691392097452.jpg)
