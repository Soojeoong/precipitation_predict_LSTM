# precipitation predict using LSTM

### ✅Data
[kaggle weather-prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction)


### ✅LSTM cell
![LSTM cell](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/lstm.png)


### ✅Model Architecture
![Stacked LSTM](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/stack_lstm.png)


### ✅Result
|index|num_layers|hidden_size|sequence_length|batch_size|epochs|learning_rate|
|------|---------|-----------|---------------|----------|------|-------------|
|**id1**|1|1|1|4|100|0.001|
|**id2**|1|10|30|4|10|0.001|
|**id3**|3|10|30|4|10|0.001|



![id1](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/test_results_id1.png) | ![id2](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/test_results_id2.png) | ![id3](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/test_results_id3.png)
:---:|:---:|:---:|
id1|id2|id3|


It tends to predict close to the average (0) of precipitation.
