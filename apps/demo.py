import streamlit as st
import sys
import pandas as pd
import zipfile
import os

sys.path.insert(1, '../machine_learning/predict.py')
from machine_learning.predict import pred


def missingSteps(predrawOutput):
    completedStep = list()
    for i in predrawOutput:
        completedStep.append(i.split(':')[1])
    completedStep = set(completedStep)
    completedStep = list(completedStep)

    classKeys = ['Step 1', 'Step 2 Left', 'Step 2 Right', 'Step 3', 'Step 4 Left', 'Step 4 Right', 'Step 5 Left',
                 'Step 5 Right', 'Step 6 Left', 'Step 6 Right', 'Step 7 Left', 'Step 7 Right']

    missingClasses = list(set(classKeys) - set(completedStep))
    missingClasses = sorted(missingClasses)
    filteredImages, caption = list(), list()

    for i in range(0, len(missingClasses)):
        if missingClasses[i] == 'Step 1':
            filteredImages.append('./images/steps/step1.png')
            caption.append('Step 1')
        if missingClasses[i] == 'Step 2 Left':
            filteredImages.append('./images/steps/step2l.png')
            caption.append('Step 2 Left')
        if missingClasses[i] == 'Step 2 Right':
            filteredImages.append('./images/steps/step2r.png')
            caption.append('Step 2 Right')
        if missingClasses[i] == 'Step 3':
            filteredImages.append('./images/steps/step3.png')
            caption.append('Step 3')
        if missingClasses[i] == 'Step 4 Left':
            filteredImages.append('./images/steps/step4l.png')
            caption.append('Step 4 Left')
        if missingClasses[i] == 'Step 4 Right':
            filteredImages.append('./images/steps/step4r.png')
            caption.append('Step 4 Right')
        if missingClasses[i] == 'Step 5 Left':
            filteredImages.append('./images/steps/step5l.png')
            caption.append('Step 5 Left')
        if missingClasses[i] == 'Step 5 Right':
            filteredImages.append('./images/steps/step5r.png')
            caption.append('Step 5 Right')
        if missingClasses[i] == 'Step 6 Left':
            filteredImages.append('./images/steps/step6l.png')
            caption.append('Step 6 Left')
        if missingClasses[i] == 'Step 6 Right':
            filteredImages.append('./images/steps/step6r.png')
            caption.append('Step 6 Right')
        if missingClasses[i] == 'Step 7 Left':
            filteredImages.append('./images/steps/step7l.png')
            caption.append('Step 7 Left')
        if missingClasses[i] == 'Step 7 Right':
            filteredImages.append('./images/steps/step7r.png')
            caption.append('Step 7 Right')

    idx = 0
    for _ in range(len(filteredImages) - 1):
        cols = st.beta_columns(4)

        if idx < len(filteredImages):
            cols[0].image(filteredImages[idx], width=150, caption=caption[idx])
        idx += 1

        if idx < len(filteredImages):
            cols[1].image(filteredImages[idx], width=150, caption=caption[idx])
        idx += 1

        if idx < len(filteredImages):
            cols[2].image(filteredImages[idx], width=150, caption=caption[idx])
        idx += 1
        if idx < len(filteredImages):
            cols[3].image(filteredImages[idx], width=150, caption=caption[idx])
            idx = idx + 1
        else:
            break


def predOutput(predrawOutput):
    actualFileNames = list()
    for i in predrawOutput:
        actualFileNames.append(i.split(":")[0])
    predStep = list()
    for i in predrawOutput:
        predStep.append(i.split(":")[1])
    data = list()
    for i in range(0, len(actualFileNames)):
        data.append([actualFileNames[i], predStep[i]])
    df = pd.DataFrame(data, columns=['File Name', 'Prediction'])
    st.dataframe(df)


def unzipFiles(zip_file):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall("./videos/")


def app():
    st.title('Demo')

    st.write("""
    Record a video of you washing your hands like this
    """)

    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/demo.PNG')

    st.write("""
    Here are a few demo videos:
    """)

    selectedViewVideo = st.selectbox("View Demo Video", ["Step 1", "Step 2 Left", "Step 3", "Step 4 Left"])

    if (selectedViewVideo == "Step 1"):
        st.video('./videos/Step 1(DEMO).mp4')
    elif (selectedViewVideo == 'Step 2 Left'):
        st.video('./videos/Step 2 Left(DEMO).mov')
    elif (selectedViewVideo == 'Step 3'):
        st.video('./videos/Step 3(DEMO).mp4')
    elif (selectedViewVideo == 'Step 4 Left'):
        st.video('./videos/Step 4 Left(DEMO).mov')

    st.write("""
            # Uploading Custom Files
            
            Please compile all the custom video(s) into a single .zip file.   
            There are pre-uploaded demo files in `Running the Model`.
            """)

    zip_file = st.file_uploader("Upload zipped Videos", type=['zip'])
    if zip_file is not None:
        # file_details = {"filename": zip_file, "filetype": zip_file.type, "filesize": zip_file.size}
        # st.write(file_details)
        st.success("File Uploaded!")
        unzipFiles(zip_file)

    st.write("""
        # Running the Model
        """)

    allFilesInVideo = []
    for (dirpath, dirnames, filenames) in os.walk('./videos'):
        for i in filenames:
            eachFile = dirpath + '/' + i
            allFilesInVideo.append(eachFile)
        break

    allFileNames = []

    for i in allFilesInVideo:
        singleName = i.split('/')[-1]
        allFileNames.append(singleName)

    allFileNames = tuple(allFileNames)

    selectedVideo = st.multiselect("Uploaded Videos (Multi-Select)",
                                   allFileNames)

    actualSelectedFiles = list()

    for i in selectedVideo:
        singleFileName = './videos/' + i
        actualSelectedFiles.append(singleFileName)

    # predOutput = 5
    # st.success(f"{predOutput}")
    if st.button("Predict!"):
        with st.spinner("Loading..."):
            if selectedVideo:
                # Launch Prediction Module
                predrawOutput = pred(actualSelectedFiles)
                st.write("""
                        # Predicted Output:
                        """)
                predOutput(predrawOutput)
                st.write("""
                        # Missing Steps:
                        """)
                missingSteps(predrawOutput)
            else:
                st.error("Select A File")
