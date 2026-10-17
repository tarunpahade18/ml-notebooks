# ML Notebooks

A collection of my machine learning notebooks (built in Google Colab).

## Notebooks

### lung_cancer_detection.ipynb
CNN image classifier for lung cancer histopathological images, built with TensorFlow/Keras.
- Dataset: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) (Kaggle)
- 3-class softmax classifier: Conv2D + MaxPooling stack, Dense head, dropout, early stopping and learning-rate scheduling

### finetuned_me.ipynb
LoRA fine-tuning of Llama-2-7b-chat (NousResearch/Llama-2-7b-chat-hf) on IIT-JEE/NEET/AIIMS student question data.
- Dataset: [IITJEE NEET AIMS Students Questions Data](https://www.kaggle.com/datasets/mrutyunjaybiswal/iitjee-neet-aims-students-questions-data) (Kaggle, via kagglehub)
- Stack: transformers, peft (LoRA), trl SFTTrainer, bitsandbytes 4-bit quantization

### Car_price_prediction.ipynb
Car price regression on the Kaggle car price prediction challenge dataset.
- Dataset: [Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge) (Kaggle)
- scikit-learn: one-hot encoding, StandardScaler, HistGradientBoostingRegressor; model and scaler saved with joblib

### house_predictins.ipynb
California housing price regression.
- Data: California housing train/test CSVs (Colab sample data)
- scikit-learn: RandomForestRegressor / HistGradientBoosting, StandardScaler, joblib

### nlp.ipynb
NLP scratch notebook (transformers setup).

## Notes
- Kaggle downloads need your own Kaggle API token (`~/.kaggle/kaggle.json`).
- No trained model weights or datasets are committed here - notebooks only.
