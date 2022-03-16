## Kaggle Feedback Prize - Evaluating Student Writing 15th solution

---

First of all, I would like to thank the excellent notebooks and discussions from https://www.kaggle.com/abhishek/two-longformers-are-better-than-1 @abhishek 
https://www.kaggle.com/c/feedback-prize-2021/discussion/308992 @hengck23 
https://www.kaggle.com/librauee/infer-fast-ensemble-models @librauee 
I learned a lot from their work. This is the second kaggle competition we have participated in, and although we are one short of gold, we are already very satisfied. In our work, I am mainly responsible for the training of the model, and @yscho1 is mainly responsible for the post-processing.

### Highlight
* In the final commit, we ensemble 6 debreta_xlarge, 6 longformer-large-4096, 2 funnel-large, 2 deberta-v3-large and 2 deberta-large. We set the max_length to 1600. We use Fast Gradient Method(FGM) to improve robustness and use Exponential Moving Average(EMA) to smooth training.

* Use optuna to learn all the hyperparameters in the post processing stage.

* CV results show that deberta-xlarge(0.7092) > deberta-large(0.7025) > deberta-large-v3(0.6842) > funnel-large(0.6798) = longformer-large-4096(0.6748)

* Merge consecutive predictions with same label, for example we merge [B-Lead, I-Lead, I-Lead], [B-Lead, I-Lead] into one single prediction. We only do this operation when the label is in ['Lead', 'Position', 'Concluding', 'Rebuttal'], since there are not consecutive predictions for these labels in the training data.

* Filter "Lead" and "Concluding". There are only one Lead label and  Concluding Label in almost all the trainging data, so we only keep the predictions that has higher score than threshold. Besides, we found that merge two Lead can increase cv further.
```python
concluding_df = sorted(concluding_df, key=lambda x: np.mean(x[4]), reverse=True)
new_begin = min(concluding_df[0][3][0], concluding_df[1][3][0])
new_end = max(concluding_df[0][3][-1], concluding_df[1][3][-1])
```

* Since the score is based on the overlap between prediction and ground truth, so we extend the predictions from word_list[begin:end] to word_list[begin - 1: end + 1]. Hoping the extended predictions can better hit ground truth and accross the 50% threshold.

* Scaling. The probabilities of each token are multiplied by a factor. The factors are obtained through genetic algorithm search.

* There are some other attempts but didn't work well. These attempts are included in the inference notebook.


### Code
* Inference and Post-processing: https://www.kaggle.com/telmazzzz/fb-inference-0-711-final?scriptVersionId=90185445
* Training:
```
# Model Training
bash script/run_Base_train_gpu.sh
# Model Predict
bash script/run_predict.sh
# Params Learning
bash script/run_params_test.sh
```
