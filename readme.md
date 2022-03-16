## Kaggle Feedback Prize - Evaluating Student Writing 15th solution

---

First of all, I would like to thank the excellent notebooks and discussions from https://www.kaggle.com/abhishek/two-longformers-are-better-than-1 @abhishek 
https://www.kaggle.com/c/feedback-prize-2021/discussion/308992 @hengck23 
https://www.kaggle.com/librauee/infer-fast-ensemble-models @librauee 
I learned a lot from their work. This is the second kaggle competition we have participated in, and although we are one short of gold, we are already very satisfied. In our work, I am mainly responsible for the training of the model, and @yscho1 is mainly responsible for the post-processing.

### Highlight
* In the final commit, we ensemble 6 debreta_xlarge, 6 longformer-large-4096, 2 funnel-large, 2 deberta-v3-large and 2 deberta-large. All the above models use max_length=1600, use Fast Gradient Method(FGM) to improve robustness and use Exponential Moving Average(EMA) to smooth training.
* Use optuna to learn weights between different class models
* CV results show that deberta-xlarge(0.7092) > deberta-large(0.7025) > deberta-large-v3(0.6842) > funnel-large(0.6798) = longformer-large-4096(0.6748)

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
