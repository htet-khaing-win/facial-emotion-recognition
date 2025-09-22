# Progress log will be written here


How to preprocess raw dataset CSVs into NumPy arrays for efficient training.

Normalization of image pixel values to the 0–1 range for stable neural network training.

Using train-test split with stratification to preserve class balance.

How to save and load .npy files for fast I/O in ML pipelines.

Visualizing random image samples with labels to sanity-check preprocessing.

Base Line and Augmented show healthy learning curve but accuracy can be improved.

Record baseline vs augmented performance (test accuracy).

Note validation accuracy improvement with checkpointing.

### Current Model

Positive emotion specialist - Happy/Surprise Good 

Negative emotion struggles - Fear/Disgust/Sad Bad

### End of Phase 1 
- All core preprocessing + baseline model experiments finished.  
- Augmentation and checkpointing working correctly.  
- Evaluation scripts confirm which classes need more attention.  

Next step → focus on deeper CNN + tuning hyperparameters.


lr = 0.0005 work best for the hyper parameter but still worse than the baseline which is 55% 


### Model	Validation Loss	Validation Accuracy
Baseline CNN	1.4071	0.5280
Augmented CNN	1.1130	0.5839
Deeper CNN	1.0340	0.6154
Deeper CNN with Callbacks	1.0119	0.6204
