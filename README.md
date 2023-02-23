# README

## data
All images used for this image classification project is saved under the <code>data/\<classname\></code> folder.

## train the classifier
Run <code>python train_furniture_classifier.py</code>, the trained model is going to be saved as a file named <code>trained_model</code>. 

Alternatively, one can play with <code>train_furniture_classifier.ipynb</code>, which is equivalent to <code>train_furniture_classifier.py</code>.

## deploy locally (after cloning this repository)
Inside the <code>inference_api</code> directory, run:
1. <code>docker build -t furniture-classifier-app .</code>
2. <code>docker run -p 80:80 furniture-classifier-app</code>
3. Open http://0.0.0.0:80/docs.

<b>Note</b>: If new model is trained and want to use it for the api, replace <code>inference_api/app/model/trained_model</code> with the new model file.

## pull docker image from remote repository
<b>Docker Image CI</b> workflow pushes docker image to <code>siqingh/img_cl</code> repository on Docker Hub.

To run the docker image, do the following:
1. Run <code>docker pull siqingh/img_cl:main</code> to pull the docker image.
2. Run <code>docker run --name furniture-classifier-app -p 80:80 -d siqingh/img_cl:main</code> to deploy.
3. Open http://0.0.0.0:80/docs. 

## use the api from the webpage
1. Once http://0.0.0.0:80/docs is opened, click on the drop down arrow for /api/classify.
2. Click on <b>Try it Out</b> button.
3. Click on <b>Choose File</b> in the payload section to upload an image
4. Click on <b>Execute</b>.
5. Predicted class will show in the <b>Server response section</b>.
