# Project Summary will be written here


## Phase 1 Summary  
- ✅ Data preprocessing complete (train/test split, one-hot labels).  
- ✅ Baseline CNN trained and saved.  
- ✅ Data augmentation introduced with improved generalization.  
- ✅ Implemented checkpointing for best model.  
- ✅ Evaluation: accuracy, loss curves, confusion matrix, classification report.  

**Key Insights:**  
- Augmentation reduces overfitting.  
- Certain emotions (happy, surprise) classified well; others (fear/ disgust/ sad) confused.  

### Performance Metrics:
- Baseline: 52.8% accuracy, 1.407 loss
- Augmented: 53.1% accuracy, 1.229 loss  
- Context: Human performance ~87%, Random ~14%

## Phase 2 Summary  
- ✅ Built deeper CNN with Dropout regularization.  
- ✅ Tuned hyperparameters (learning rate, batch size).  
- ✅ Added EarlyStopping + ReduceLROnPlateau for stable training.  
- ✅ Compared all CNNs: baseline, augmented, deeper, callbacks.  
- ✅ Best model: callbacks CNN with validation accuracy ~62%.  

**Key Insights:**  
- Dropout + callbacks improved stability.  
- Still confusion between similar emotions .  

