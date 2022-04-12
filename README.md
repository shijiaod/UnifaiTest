# Unifai Test

This repo contains data, code and trained model for Unifai's Test

## Requirments
- python3.6 or higher
- numpy
- tensorflow
- scikit-learn
- nltk
- pandas

## Training

`python3 src/cnn.py --do_train=True --do_test=False --epochs=5` 

or 

`python3 src/cnn.py --do_train=True --do_test=False  --epochs=5 --train_data=data_technical_test/train_technical_test.csv`

Trainning with evaluation:

`python3 src/cnn.py --do_train=True --do_test=False --do_eval=True --eval_data_size=3000` 

## Evaluation

`python3 src/cnn.py --do_train=False --do_test=True` 

or 

`python3 src/cnn.py --do_train=True --do_test=False --train_data=data_technical_test/test_technical_test_2.csv`

## Other

If we need to change the content of train data and test data, we can retrain another encoder for text and category for a better result. Don't forget to change the parameter `train_vector`
