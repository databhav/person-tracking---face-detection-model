# Person tracking and detection using OpenCV, YOLO and face_recognition



**Deployment link will be added soon**



This model works on OpenCV and face_recognition library. The model works on the following:

1. Person Detection

2. Face Detection

3. Feature Extraction

4. Re-identification (**IN PROGRESS**)

5. Model Evaluation using visualisation



## 1. Person Detection:

<img src="file:///home/anubhav/pyprojects/us/readme_images/person_detection1.png" title="" alt="" width="510">

The model is trained to only identify and highlight person in a video feed. It also tracks their movement. The model also captures the image of the person when confidence is more than 90%.



<img src="file:///home/anubhav/pyprojects/us/readme_images/person_detection2.png" title="" alt="" width="571">



## 2. Face Detection

There are two ways in which model can extract faces

- Using YOLOv8n-Face library

- Using Face_recognition library

The script for the use of YOLOv8n-Face library is given in the <mark>alternate.py</mark> file. The script only identifies face when there's a scope of face extraction. To explain in detail when confidence that the object in the frame is a person is 90%, at that moment the face detection starts and when face detection confidence is more than 70%, then the face is extracted and saved for feature extraction.

<img title="" src="file:///home/anubhav/pyprojects/us/readme_images/person_detection3.png" alt="" width="522">



On the other hand face_recognition library which is used primarily for face detection is used when the images with more than 90% confidence have been saved. Afterwards, the faces are detected from the images.

<img src="file:///home/anubhav/pyprojects/us/readme_images/face_detection2.png" title="" alt="" width="528">



## 3. Feature Extraction

using the python script <mark>local.py</mark> the features are extracted from every single face from every single image which was extracted. The two features that are extracted are '128-D embedding' and 'color histograms'



![](/home/anubhav/pyprojects/us/readme_images/feature_extraction1.png)

afterwards using cosine similarity, duplicates and similar images are removed to keep only the required files for model training for re-identification using extracted features.



## 4. Re-identification (IN PROGRESS)

## 5. Model evaluation

The model was made to work on different 15 second segments of clip and of the first 15 seconds of the clip in <mark>output_clips</mark> folder it can be seen that only two people's data was extracted and after the running of script <mark>main.py</mark> and <mark>local.py</mark> these are the following results:![](/home/anubhav/pyprojects/us/readme_images/modl_eval1.png)

.jpg.txt files contain the 128-D embeddings and the .jpg-h.txt contains color histogram values which can be used for re-identification. Also the video quality is 360p.



#### NOTE:

**Readme will be updated soon with better model evaluation after Re-identification**
