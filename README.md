# DeepSchmatzing

Source code to replicate the results of:

"Using Representation Learning and Out-of-domain Data for a Paralinguistic Speech Task"
Benjamin Milde, Chris Biemann
In: Proceedings INTERSPEECH. Dresden, Germany, September 2015. 

(Please cite this paper if you use this project in your academic work)

It allows you to train an audio sequence classifier based on Convolutional Neural Networks for language idenfication using the Voxforge corpus and for Eating Condition classification. The output of several classification systems was submitted to the INTERSPEECH 2015 Computational Paralinguistics challenge. Using an ensemble of many slightly different models a final score of 75.85% unweighted average recall (UAR) for 7-way Eating Condition classification was reached. 

The Eating Condition classification task is to determine if a short utterance is either "No Food", "Banana", "Crisp", "Nectarine", "Haribo", "Apple", "Biscuit", based on training and testing data from the iHEARu-EAT corpus.  Speakers eat one of these specific food types while they speak and recite a poem.

We made use of transfer learning and data augmentation, i.e. transferring weights of a 7-way language identification system trained on Voxforge. Weights of the convolutional layers of the trained Voxforge LID network were used as initialisation. [Rubberband](http://breakfastquay.com/rubberband/) was used to generate additional pitch shifted variants of the training files (+/- 1 semitone, crisp level 6).

This code evolved under the time constrains of a challenge - good results were more important than good code quality. You should be able to replicate results, but don't expect easy to read code. 
