File Structure

venv/
archive/ -- this is the downloaded file from kaggle
.gitignore -- don't upload images to github. add them to your ignore file.
                github wont let you do it anyway
data/ - folder holding all images in there class folder
    0/ - folder conatains all images for that class
        img1.jpg
        img2.jpg - images 
        ...
    1/
    2/
    3/
    4/
    5/
    6/
    7/
    8/
    9/
Dataloader.py
dataset.py
model.py
train.py
trainning.py
visionPreprocess.py