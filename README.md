<!--
 * @Author: dengshijiao
 * @Date: 2022-04-12 11:26:37
 * @LastEditTime: 2022-04-12 18:07:39
 * @Description: 
-->
# Unifai Test

This repo contains dataset, code and trained model for Unifai's Test

I've split 5000 data from train dataset for evaluating model and got 80.22% for the accucary of dev dateset. 

Hope it won't descend too much on test dataset :D


## Requirments
- python3.6 or higher
- numpy
- tensorflow
- scikit-learn
- nltk
- pandas

## Training

```shell
python3 src/cnn.py \
    --do_train=True \
    --do_test=False \
    --epochs=5
```

or 

```
python3 src/cnn.py \
    --do_train=True \
    --do_test=False  \
    --epochs=5 \
    --train_data=data_technical_test/train_technical_test.csv
```

Trainning with evaluation:

```
python3 src/cnn.py \
    --do_train=True \
    --do_test=False \
    --do_eval=True \
    --epochs=5 \
    --save_best_only=True \
    --batch_size=64 \
    --eval_data_size=5000 \
    --eval_data_size=3000
```

## Evaluation

```
python3 src/cnn.py \
    --do_train=False \
    --do_test=True
```

or 

```
python3 src/cnn.py \
    --do_train=False \
    --do_test=True \
    --test_data=data_technical_test/test_technical_test.csv
```

## Other

If we need to change the content of train data and test data, we can retrain another encoder for text and category for a better result. Don't forget to change the parameter `train_vector`
