import streamlit as st

def app():
    st.write("""
    ## Our Approach
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
    direction and contrast as augmentation techniques.

    ### Evaluation Metrics
    Since this is a classification problem, we will be using accuracy as the main metric to evaluate
    the model performance. The accuracy is calculated as follows:
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/eval.PNG')
    st.write("""
    In addition, we constructed the confusion matrix in order to better understand which actions that
    tend to be misclassified leading to the poor model performance. A confusion matrix effectively
    reports the number of predicted and true values for each class.

    ## Experiments

    ### Different Architectures
    In this work, we experimented with two different approaches: ConvLSTM and CNN-LSTM. We
    then shortlisted the best model based on their validation accuracy. Further experiments using
    hyperparameter tuning and data augmentation are described in the subsequent subsections.

    #### ConvLSTM
    ConvLSTM is a type of RNN with gates similar to LSTM, except that the internal matrix
    multiplications are exchanged with convolution operations. This implementation allows the
    layers to extract spatio-temporal features from the video frames.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig3.PNG')
    st.write("""
    We modified the ConvLSTM model [6] and came up with the final model architecture consisting
    of one ConvLSTM layer with ReLu activation and one Dropout layer, followed by a fully
    connected layer with softmax for classification (Figure 3). Initial experiments with more layers
    result in model overfitting, which is likely due to our small dataset size.

    #### CNN-LSTM
    CNN-LSTM model is a typical architecture for many video action recognition tasks (Figure 4).
    The CNN model can first extract high-level spatial features from the frames of the input video.
    Initially, we used a simple CNN architecture for the CNN backbone, with only one 2D
    convolutional layer, one fully-connected layer with ReLU activation and one Dropout layer. We
    also experimented with transfer learning using pre-trained backbones such as AlexNet and
    ResNet-50.
    The extracted spatial features are then fed into the LSTM layer to extract temporal correlation
    between the frames by remembering past ones. Finally, the last output from the LSTM layer can
    be put into a final fully connected layer with softmax for classification.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig4.PNG')
    st.write("""
    #### Comparing performances of architectures
    We trained for 50 epochs for each of the four proposed architectures and recorded the
    validation accuracies in Table 1 to determine which is the best architecture.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab1.PNG')
    st.write("""
    The CNN-LSTM with AlexNet backbone was able to achieve the highest validation accuracy of
    58.4%. Comparing the pre-trained CNN models, ResNet50 and AlexNet, ResNet-50 performs
    worse than AlexNet. This could be because the ResNet-50 has much more parameters than
    AlexNet and given our small dataset size, a model with too many parameters will result in
    overfitting.
    Nonetheless, using pre-trained CNN models can help to boost the accuracy and improve the
    model performance, as compared to training a custom model from scratch as shown in Table 1,
    where both the ConvLSTM model and CNN-LSTM model with custom CNN layers yielded
    poorer validation accuracies. In particular, the performance of the CNN-LSTM model with
    custom CNN layers was almost as poor as a random model, which would have an expected
    validation accuracy of 8.3%. This is because the pretrained models have already learnt general
    features that can be reused, making it easier for the models to learn and converge.
    Therefore, in subsequent experiments and training of the final model, we will use the
    CNN-LSTM architecture with pretrained AlexNet weights in the CNN layers.

    ### Hyperparameter Tuning
    We conducted a grid search to find the best combination of hyperparameters, including batch
    size and learning rate. The search space for batch size was [8, 16, 32] and the search space for
    learning rate was [0.01, 0.001, 0.0001]. Each combination was run for 20 epochs. The
    experimental results are recorded in Table 2.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab2.PNG')
    st.write("""
    We find that a batch size of 32 and learning rate of 0.001 achieves the highest validation
    accuracy of 51.9%. Hence, we will use these hyperparameters for training our final model.
    We also varied the spatial dimensions of the input frames from the video, and the validation
    accuracy is recorded in Table 3.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab3.PNG')
    st.write("""
    From Table 3, it is evident that as spatial dimension increases, the validation accuracy also
    increases. This is because the larger the spatial dimension, the higher the resolution is of each
    input frame. The CNN layers would then be able to extract the spatial features more effectively
    since the finer details of the hands are present. This could help the model better distinguish
    between similar handwashing actions.   

    However, the time taken to train the model for the same number of epochs increases with the
    spatial dimension, since the kernels have to convolve over a larger pixel area. There is a
    tradeoff between the model performance and time taken during training, but comparing the
    spatial dimensions of 84 x 84 and 128 x 128, the increase in validation accuracy is much more
    significant (about 26% accuracy) than the increase in time taken (about 2:50 min). It is much
    better to train for longer periods of time to achieve better model performance than to train
    quickly but have poor performance.

    ### Data Augmentation
    Using the AlexNet as the pretrained CNN model with a batch size of 32, learning rate of 0.001
    and frame size of 128 by 128, we applied data augmentation on the original dataset, using
    translation and contrast with the goal of reducing overfitting.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab4.PNG')
    st.write("""
    From Table 4, data augmentation did not improve the validation accuracy even though in
    principle, data augmentation should increase the total number of effective training samples and
    thereby reduce overfitting. This may be because we froze most of the layers of the AlexNet
    backbone and are only updating the top few layers and the LSTM layer, thus the bottom layers
    are unable to generalize well on the augmented data. Introducing the new augmented data
    during the fine-tuning process could disturb the predictive power of the model, resulting in a
    decrease in overall model performance. Data augmentation may thus work better when we train
    the model from scratch.

    ### Results & Discussion

    #### Final Evaluation on Test Set
    Overall, using our best model, the CNN-LSTM model with pretrained AlexNet backbone, batch
    size of 32, learning rate of 0.001 and frame size of 128 by 128, we are able to achieve a test
    accuracy of 88.5%.

    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig5.PNG')
    st.write("""
    From Figure 5, we can see that the validation loss is much higher than the train loss, and that
    the validation accuracy is much lower than the train accuracy. Moreover, we can see that the
    validation loss is quite volatile, which means that our model is overfitting. This could be due to
    our dataset being very small in size, resulting in the model not seeing enough samples to be
    able to generalize well. Nonetheless, we have reduced overfitting as much as possible by
    adding regularization such as Dropout layers and saving the model based on best validation
    accuracy
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig6.PNG')
    st.write("""
    Overall, our proposed solution is able to predict the correct classes for most of the test samples
    (Figure 6). However, some classes are harder to predict, such as Step 6 Right. There are some
    “weird” cases of model misclassification in Step 6 Right where the actual and wrongly predicted
    classes do not look similar. This can be seen in Figure 7a where Step 6 Right has been wrongly
    classified as Step 5 Right, when the actual hand washing action for Step 5 Right is as shown in
    Figure 7b. Another instance of a “weird” misclassification is when Step 6 Right is wrongly
    predicted as Step 7 Right as shown in Figure 8a and Figure 8b.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig7_8.PNG')
    st.write("""
    From the confusion matrix on the test set in Figure 6, we can see that there is one case where
    Step 6 Left is wrongly predicted as Step 6 Right, and one case where Step 7 Left is wrongly
    predicted as Step 7 Right. This indicates that there is some difficulty in differentiating between
    the left and right actions, which is one of the possible reasons for misclassification in our model.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig9.PNG')
    st.write("""
    Additionally, if the sink and the hands are of similar colours, the model may also malfunction. An
    example of misclassification is when Step 2 Left is wrongly predicted as Step 7 Right as shown
    in Figure 9a and Figure 9b. One way to confirm if the model is learning only the hand features
    instead of the background is to plot a heatmap of the most sensitive pixels using gradient
    sensitivity methods.

    #### Performance Comparison with Three-Stream-Algorithm
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab5.PNG')
    st.write("""
    Additionally, if the sink and the hands are of similar colours, the model may also malfunction. An
    example of misclassification is when Step 2 Left is wrongly predicted as Step 7 Right as shown
    in Figure 9a and Figure 9b. One way to confirm if the model is learning only the hand features
    instead of the background is to plot a heatmap of the most sensitive pixels using gradient
    sensitivity methods.

    #### Performance Comparison with Three-Stream-Algorithm
    The three-stream-algorithm [4] consists of RGB Frames, Optical Flow and Histogram of
    Oriented Gradients models. These models were pre-trained on the ImageNet dataset then
    further trained on the Hand Wash dataset. The three-stream-algorithm was trained on the hand
    wash dataset with 3,504 video clips, while our proposed CNN-LSTM model only trained on 711
    video clips.

    Comparing the performances from Table 5, our proposed architecture was able to achieve about
    the same test accuracy as the three-stream-algorithm. Given our significantly smaller dataset, it
    is possible that our proposed solution is quite effective in achieving state-of-the-art performance.
    Since both the state-of-the-art model and our model use pre-trained neural networks that have
    shown relatively high accuracy scores, this proves the effectiveness of using transfer learning
    on hand washing datasets. The state-of-the-art model uses DenseNet-BC pretrained with
    ImageNet weights and our model uses pretrained AlexNet. These pretrained models might have
    been trained on similar datasets as the hand washing dataset, making them reusable and
    effective. Moreover, they do not have too many learnable parameters that will cause the models
    to overfit as compared to ResNet-50 that have a large number of parameters, hence resulting in
    a poor accuracy score during our experiment. Hence, although transfer learning tends to allow
    rapid progress and improves the performance of models, the right pretrained model must be
    used to ensure its effectiveness on the task that it is being trained on.

    However, given the significant difference in our dataset sizes, their hand wash dataset might
    contain a larger diversity of environmental variables. In addition, since their model will be able to
    capture more features from their larger dataset size, it might be more generalizable in
    classifying hand washing videos. In comparison, our dataset size might be too small for our
    model to be generalizable, hence causing it to overfit easily. Therefore, if we were to test our
    model on a larger test set with more variance of environmental factors, our test accuracy might
    not be able to perform as well.
    """)
    st.write("""
    ## Conclusion
    In conclusion, our proposed solution consists of a CNN-LSTM architecture with Alexnet
    backbone, and was able to achieve a high test accuracy of 88.5%, which is a comparable
    performance to the state-of-the-art models. While the model is unable to predict the correct
    classes all the time, it gives a good gauge about which of the 12 handwashing actions were
    done or left out. As such, the implementation of our solution would promote hygiene and healthy
    habits in order to protect oneself against the pandemic.

    ### Future Work
    An area of improvement is the expansion of the handwash dataset. Increasing the dataset with
    diverse environmental conditions such as different brightness or different background sinks
    would help make the model more robust and be able to generalize well on unseen data. With a
    larger dataset, it also becomes more possible to train the architecture from scratch and employ
    data augmentation techniques to further improve model robustness.

    As our model misclassified some of the steps, one method to tackle this is to have model
    interpretability methods such as using the LIME algorithm which attempts to identify the model
    features to understand how the predictions change. Once the features are identified, we can
    then tune the neural network to target those misclassified features to improve model accuracy.
    Moreover, as there are limitations to classifying coloured images, given different coloured sinks
    or different colour of the skin, 3D skeleton-based action recognition can be useful to analyse
    spatial joint relations and temporal posture dynamics. With the skeleton data, it might allow our
    model to learn more features of each action to improve the model accuracy [7].

    It may be good to also explore larger spatial dimensions, since from our experiments in Table 3,
    the larger the spatial dimensions, the higher the validation accuracy. By further investigating the
    tradeoffs, the extent to which the accuracy stops increasing can be determined and the most
    optimal spatial dimension can be used to further improve the model.
    State-of-the-art action recognition models such as 3D-ConvNet can also be explored for better
    accuracy of the model as it is able to learn spatiotemporal features by incorporating 3D filters
    and pooling kernels from a 2D architecture (DenseNet) which enables the model to be able to
    learn motion characteristics of the hand washing steps.

    Lastly, in order to correctly classify input videos that do not belong in any of the 12 classes, a
    binary classifier can be separately trained to distinguish these videos correctly. During
    prediction, the sample can be fed into the binary classifier, and only be inputted in the
    CNN-LSTM architecture if the binary classifier had predicted that one of the 12 classes were
    present
    """)
    st.write("""
    ## Old UI Demo
    """)
    st.video('https://www.youtube.com/watch?v=DLfKYGBf7oE')
    st.write("""
    ### References
    [1] Boshell, P. (2016). What Is The Correct Hand Washing Technique? Debgroup. Retrieved
    April 20, 2021, from https://info.debgroup.com/blog/what-is-the-correct-hand-washing-technique

    [2] Simonyan, K., Zisserman, A.: Two-stream convolutional networks for action recognition in
    videos. In: Advances in neural information processing systems. pp. 568–576 (2014)

    [3] M. E. Kalfaoglu, S. Kalkan, and A. A. Alatan, “Late temporal modeling in 3D CNN
    architectures with BERT for action recognition,” arXiv [cs.CV], 2020.

    [4] A. Nagaraj, M. Sood, C. Sureka, and G. Srinivasa, “Real-time Action Recognition for
    Fine-Grained Actions and The Hand Wash Dataset,” 2019.

    [5] real-timeAR, “Sample: Hand Wash Dataset,” Kaggle, 19-Apr-2020. [Online]. Available:
    https://www.kaggle.com/realtimear/hand-wash-dataset.

    [6] A. Palazzi, H. Yu, and S. Pini, “ndrplz/ConvLSTM_pytorch,” GitHub, 2017. [Online].
    Available: https://github.com/ndrplz/ConvLSTM_pytorch.

    [7] T. Huynh-The and D. Kim, "Data Augmentation For CNN-Based 3D Action Recognition on
    Small-Scale Datasets," 2019 IEEE 17th International Conference on Industrial Informatics
    (INDIN), 2019, pp. 239-244, doi: 10.1109/INDIN41052.2019.8972313.

    [8] Torch Contributors, LSTM Pytorch, 2019. [Online]. Available:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """)

