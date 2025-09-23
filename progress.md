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


### End of Phase 2  
- CNN development completed with multiple variants.  
- Best performing CNN identified and saved.  
- All evaluation scripts in place (confusion matrix, comparison).  

Next → Week 3 (Transfer Learning with MobileNetV2).  
 
- Trained MobileNetV2 with frozen backbone.  
- Saved model as `mobilenetv2_frozen.h5`.  
- Test accuracy: ~37% (baseline transfer learning).  
- Next → fine-tune top layers of MobileNetV2 for higher accuracy.

- Fine-tuned top 30 layers of MobileNetV2.  
- Used low LR + callbacks (EarlyStopping, ReduceLROnPlateau).  
- Saved model as `mobilenetv2_finetuned.h5`.  
- Test accuracy: ~43% (~+6% improved over frozen baseline).  

 
- Compared CNNs vs MobileNetV2 (frozen + fine-tuned).  
- Saved comparison results in `results/`.  
- Clear bar chart visualization included.

###MobileNetV2 Transfer Learning Optimization

- Fixed critical learning rate issue: Changed from 1e-5 to 1e-4 (10x increase)
- Implemented image resizing: Upgraded input from 48x48 to 96x96 pixels
- Updated callbacks strategy: Added ModelCheckpoint with val_accuracy monitoring
- Results: MobileNetV2 accuracy improved from 43.37% → ~60% (+16.63% gain)
- Learning curves: Achieved healthy training with no overfitting, validation ≥ training accuracy
- Model ranking: Moved from 5th to 3rd position, now competitive with custom CNNs

Key learnings: Input size mismatch was major bottleneck for transfer learning. Pre-trained models need appropriate input dimensions to leverage ImageNet features effectively.

### Img size tweaking  
- Tried out both 96x96 and 224x224  
- Test accuracy: ~66% (improved over the 48x48 and 96x96).
- ran into overfitting but callback solved it
- - Saved model as `mobilenetv2_finetuned_224x224_best.h5`.  

