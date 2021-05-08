import streamlit as st

def app():
    st.title('Model')

    st.write('`A Custom AlexNet Model` is used in this application. For more information on architecture, please refer to home page ')

    st.write("""
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
