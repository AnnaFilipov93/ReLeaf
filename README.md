Note - reuploaded organized version for the project

ReLeaf - web application

The app identifies flowers and leaves diseases.

In the process, a deep CNN network model is trained, resNet50,  to recognize 7 flowers (daisy, sunflower, dandelion, rose, tulip, iris, and water lily) and 4 leaf diseases (healthy, dryness, magnesium lack, and bug holes) from datasets that I’ve collected.

YOLO v3 is used to detect the objects to separate between flowers and the leaves before sending each of the images to the corresponding model.

A digital poster in Hebrew: here

To run the code just run the file app.py.

Before you download the code make sure that you have the Git LFS extension for Git that allows you to work with large files.  Click here to download the extension.