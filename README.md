# Voice-Activity-Decoder
A quick implementation of a Voice Activity Decoder (VAD) for automated speech processing, based on a Convolutional Neural Network implemented in Keras.

## Background and elements of the project:

This is a rather old project I did when I started working on audio data and deep learning, hence the scripts are far from being optimized (eg Pandas dataframes manipulations are not smartly taking advantage of the dedicated methods), but they are a working synthesis of many experiments led in various notebooks. I may refactor this code but this is not one of my projects as of now.

The scripts have the following roles:
- *data_handling_and_preparation*: essentially loads audio files, performs simple data augmentation with noise addition, applies some preprocessing to tag excerpts as speech or non-speech and converts them to log-spectrograms;
- *decision_cnn*: prepares the data and creates the Keras model (a CNN whose inputs are n timeframes of log-spectrograms), then trains it, saves it and displays its losses and accuracies;
- *decision_process_evaluation*: uses the neural network to decide of the parts considered as speech on the audio (the decisions at excerpt-level is then aggregated and postprocessed with decision smoothing and the hangover scheme), the difference of timestamps with the groundtruth data being also measured;
- *main*: gets the input from the user and returns the segments of speech detected by the VAD.

## Results:

From this very simple model, we could reach a validation accuracy of nearly 89% after 5 epochs of training:
![Accuracy](Results%20CNN.png)
The weird behavior of the training accuracy being below the validation accuracy is most likely due to the utilisation of Dropout layers.
