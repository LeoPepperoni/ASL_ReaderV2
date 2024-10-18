# ASL Reader V2

ASL Reader v2 is the second iteration of our modifications to an open-source machine learning model aimed at recognizing and classifying American Sign Language (ASL) gestures. The model uses video input data to classify various ASL signs based on saved training data, making it a valuable tool for real-time ASL interpretation.

Our machine learning model is built using a K-Nearest Neighbor (KNN) algorithm, which classifies input video frames by comparing them to known ASL gesture examples. This version introduces several improvements in accuracy, performance, and user experience over the original implementation.

[[Web Demo](https://www.asl-live.com/)]

# How it works
 
by combining the results of a [Holistic](https://google.github.io/mediapipe/solutions/holistic.html) model over multiple frames. We can create a reasonable set of requirements for interpreting sign language, which include body language, facial expression, and hand gestures.

The next step is to predict the sign features vector using a classifier model. Lastly, output the class prediction using K-Nearest Neighbor classification.


# Installation

- Install python 3.9
- Install dependencies
  ```
  pip3 install -r requirements.txt 
  ```

# Run demo

```
python3 webcam_demo.py
```

- Use record mode to add more sign.  
  ![record_mode](assets/record_mode.gif)

- Play mode.  
  ![play_mode](assets/play_mode.gif)

# Run Flask app (built-in endpoints)

```
python3 AWS.py
```
- Running this file will launch an instance of the model locally that offers an endpoint (/upload_video) that can be sent a video, and returns predictions upon recieveing sufficent data.
- In order for this endpoint to work, the model must already be trained on at least one word (but you will obviously want more depending on your prediction needs)

# Train classifier

You can add a custom sign by using Record mode in the full demo program.  
But if you want to train the classifier from scratch you can check out the process [`here`](/notebooks/train_translator.ipynb)
