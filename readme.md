# Person tracking and detection using OpenCV, YOLO and face_recognition



**Deployment link will be added soon**



This model works on OpenCV and face_recognition library. The model works on the following:

1. Person Detection

2. Face Detection

3. Feature Extraction

4. Re-identification (**IN PROGRESS**)

5. Model Evaluation using visualisation



## 1. Person Detection:

![person_detection1](https://github.com/databhav/person-tracking---face-detection-model/assets/124768416/bde314b5-4301-458c-b331-0044c1e6dab6)

The model is trained to only identify and highlight person in a video feed. It also tracks their movement. The model also captures the image of the person when confidence is more than 90%.

![person_detection2](https://github.com/databhav/person-tracking---face-detection-model/assets/124768416/47e4c147-ddc5-4869-9749-49d5c9fc4674)



## 2. Face Detection

There are two ways in which model can extract faces

- Using YOLOv8n-Face library

- Using Face_recognition library

The script for the use of YOLOv8n-Face library is given in the <mark>alternate.py</mark> file. The script only identifies face when there's a scope of face extraction. To explain in detail when confidence that the object in the frame is a person is 90%, at that moment the face detection starts and when face detection confidence is more than 70%, then the face is extracted and saved for feature extraction.

![person_detection3](https://github.com/databhav/person-tracking---face-detection-model/assets/124768416/c67ea223-fac1-4130-8610-66b8fdbcaee4)




On the other hand face_recognition library which is used primarily for face detection is used when the images with more than 90% confidence have been saved. Afterwards, the faces are detected from the images.


![face_detection2](https://github.com/databhav/person-tracking---face-detection-model/assets/124768416/6227d41a-1674-4bbb-8e61-cb811c43effe)


## 3. Feature Extraction

using the python script <mark>local.py</mark> the features are extracted from every single face from every single image which was extracted. The two features that are extracted are '128-D embedding' and 'color histograms'



![feature_extraction1](https://github.com/databhav/person-tracking---face-detection-model/assets/124768416/ad4725d6-5aac-4c03-b104-bbc2bacc6ac2)

afterwards using cosine similarity, duplicates and similar images are removed to keep only the required files for model training for re-identification using extracted features.



## 4. Re-identification (IN PROGRESS)

## 5. Model evaluation

The model was made to work on different 15 second segments of clip and of the first 15 seconds of the clip in <mark>output_clips</mark> folder it can be seen that only two people's data was extracted and after the running of script <mark>main.py</mark> and <mark>local.py</mark> these are the following results:
![modl_eval1](https://github.com/databhav/person-tracking---face-detection-model/assets/124768416/ff6002cb-ea94-42b7-88cc-4c54922fce8e)

.jpg.txt files contain the 128-D embeddings and the .jpg-h.txt contains color histogram values which can be used for re-identification. Also the video quality is 360p.



#### NOTE:

**Readme will be updated soon with better model evaluation after Re-identification**
