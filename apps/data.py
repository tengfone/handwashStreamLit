import streamlit as st

def app():
    st.title('Data')

    st.write("The `Data` obtained consist of a sample [Kaggle Dataset](https://www.kaggle.com/realtimear/hand-wash-dataset) and our customized Dataset")

    st.write("""
    ### Understanding the Dataset
    The Hand Wash Dataset consists of 25 individual videos for each of the 12 hand wash actions. This gives a total of 300 videos for the entire dataset.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig1.PNG')
    st.write("""
    Given the small sample size of the Hand Wash dataset, we recorded more hand washing videos
    to increase the dataset size. To increase the robustness of the model, we took videos in widely
    different settings and environments to provide as much variance as possible. The varied
    parameters include camera position, actor, background and illumination.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig2.PNG')
    st.write("""
    After combining our self-collected and the online dataset, we did a stratified train-validation-test
    split, which gives rise to the distribution as shown in Figure 2. We ensured that the final
    distribution is balanced, which prevents the model from being overly biased toward any majority
    class during training and inference.

    ### Prepocessing videos into numpy arrays
    Processing of each video, including open, sampling and closing, takes approximately 1 second.
    Without first converting the videos into 3D numpy arrays, the model would have to redo this
    processing operation for every iteration, resulting in extremely long training times. Therefore, it
    was imperative to perform a preprocessing step once at the start so that subsequent iterations
    are able to access the already processed videos to reduce unnecessary computational cost.
    Specifically, every fourth frame of each video was sampled and resized to 128 x 128 until a total
    of 32 frames was reached. For the purposes of more efficient model training, we would be
    working with this processed numpy dataset.

    ### Prepocessing of numpy arrays
    The numpy arrays had to be further processed before being fed into the neural network. First, a
    series of consecutive frames was sampled from the numpy dataset using the time cropping
    function. Here, the number of selected frames refers to the number of timesteps for the RNN
    layer. Second, normalization was done by dividing each pixel by 255. This allows faster
    convergence when training our model, ensuring computational efficiency as larger values can
    slow down the learning process significantly. Third, standard data augmentation was applied to
    the train dataset to reduce overfitting. In particular, we explored translation in the horizontal
    direction and contrast as augmentation techniques.""")
    
