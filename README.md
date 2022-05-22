# Predicting Investment Based on Investor’s Facial Expression in a Startup Funding Pitch Using Support Vector Machine and Long-Short Term Memory Network
This repo contains code for a study in predicting investment based on investor's facial expression in a startup funding pitch. 
In this study, the use of Long Short-Term Memory (LSTM) neural network is proposed to be compared with Support Vector Machine as the baseline model 
with proven results in predicting audience engagement. Different frame selection methods are considered to shorten the sequence for the LSTM model. 
Both OpenFace and FaceReader, well-known video extraction analysis tools, are used and their predictive performance are compared. 
Different facial action unit sets are considered as features. The results show that the proposed LSTM model outperforms the baseline SVM model 
in predicting an investor's decision to invest. Different frame selection methods seem to have limited impact on the LSTM models' performance. 
FaceReader also outperforms OpenFace as facial units extraction tools in most cases. 
Finally, a more complete set of facial action units outperforms limited facial action units set referring to certain emotions 
when used to train an LSTM model.

## Experiment Overview
Overview of the experiment in regards to the dataset and the methods are displayed in the figure below.

<img width="654" alt="experimentoverview" src="https://user-images.githubusercontent.com/1264845/169686241-9d6a599a-1d1b-4041-a1f3-7f138e3554fc.png">

## Dataset
In this study, the dataset from a previous research “The promise of social signal processing for research on decision-making in 
entrepreneurial contexts” (Liebregts, Darnihamedani, Postma, & Atzmueller, 2020) will be used. The dataset consists of both OpenFace and FaceReader analysis 
output of pitching video recordings and surveys conducted during startup funding pitches from 2018 to 2021. 51 groups of students from 
startup and entrepreneurship courses in The Jheronimus Academy of Data Science (JADS) were asked to pitch their group’s business ideas 
to 3 investors. Both the pitcher and the 3 investors were recorded during each pitch session. After each pitch session, the 3 investors 
were asked to fill in questionnaires evaluating both the business ideas and the pitchers, and also their probability to invest on the 
business ideas.

## Preprocessing
### Temporal Features
An SVM model does not accept observations with additional sequential dimensions. Therefore, temporal features will need to be 
summarised for each observation’s sequential dimensions. Of many possible temporal features, mean and standard deviation 
are often preferred due to their simple and intuitive nature. In this study, the Python library TSfresh will be used to extract 
mean and standard deviation of sequential facial action unit values for each observation.

### Frame Selection
RNNs such as LSTM are suboptimal when used for modelling long sequences. Therefore, the following 
2 methods will be used to sample the long sequences of 3,500 frames into shorter sequences:
- Frame-sampling: extracting every 50th frame of each 3,500-frames sequence resulting in a 70-frames sequence
- Mean-sampling: extracting the mean values of every 50 frames of each 3,500-frames sequence resulting in a 70-frames sequence of mean values

### Normalization
In a classification task with facial action units as input, a squared min-max normalization is useful to optimize 
the training process. Min-max normalization will prevent the values from being too large or too small from each other. 
Squaring the min-max normalized values will then emphasize both the rewards and punishments for relevant and irrelevant features respectively.

## Models
This study will compare the performance of both SVM and LSTM in classifying an investor’s decision to invest based 
on the investor’s facial action units. Both models will be trained on 3 variations of facial action units features:
- All 16 AUs
- AU7 (Lid Tightener) + AU12 (Lip Corner Puller), AU5 (Upper Lid Raiser), AU25 (Lips Part) and AU26 (Jaw Drop) representing concentration, gestures and excitement
- AU6 (Cheek Raiser) + AU12 (Lip Corner Puller) and AU7 (Lid Tightener) representing enjoyment smile and attentional expressions 

Both OpenFace and FaceReader extractions of the related AUs will be used and compared.

## Results
As shown in table below, the highest performance based on the F1-score, 0.593, and AUC score, 0.600, belongs to the LSTM model trained 
on all 16 facial action unit types extracted by FaceReader and mean-sampled every 50 frames. In terms of AUC, it is 6\% higher 
than the baseline model SVM’s highest performance, 0.542, which is 4\% higher than chance-level performance.

<img width="690" alt="Screenshot 2022-05-22 at 10 58 06" src="https://user-images.githubusercontent.com/1264845/169687525-bfccc3ce-dbc4-4211-9f13-2f18b73caa19.png">

Further studies can explore improvements on the SVM model by considering temporal features other than mean and standard deviations. 
The LSTM model can be improved by further exploring optimal hyperparameters. Both models can be improved by feeding data with much 
more observations and more balanced class representations. Other facial expression features other than facial action units, such as 
head pose and eye gaze, can also be explored to further understand the extent of an investor’s facial expression’s predictive performance 
towards the investor’s decision to invest.

## Acknowledgement
Liebregts, W., Darnihamedani, P., Postma, E., & Atzmueller, M. (2020,
10). The promise of social signal processing for research on decisionmaking
in entrepreneurial contexts. Small Business Economics, 55.

OpenFace: https://github.com/TadasBaltrusaitis/OpenFace

FaceReader: https://www.noldus.com/facereader

TSfresh: https://tsfresh.readthedocs.io/en/latest/#


