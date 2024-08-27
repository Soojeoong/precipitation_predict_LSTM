# precipitation predict using LSTM

### ✅데이터
[캐글 weather-prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction)


### ✅LSTM cell
![LSTM cell](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/lstm.png)


### ✅Model Architecture
![Stacked LSTM](https://github.com/Soojeoong/precipitation_predict_LSTM/blob/main/stack_lstm.png)


### ✅결과
|index|num_layers|hidden_size|sequence_length|batch_size|epochs|learning_rate|
|------|---------|-----------|---------------|----------|------|-------------|
|**id1**|1|1|1|4|100|0.001|
|**id2**|1|10|30|4|10|0.001|
|**id3**|3|10|30|4|10|0.001|

![id1]()
![id2]()
![id3]()

강수량의 평균(0)에 가깝게 예측하는 경향이 있다.