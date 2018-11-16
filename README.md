# Named Entitiy Recognition

Contributors:
- Luka Bašek - https://github.com/lbasek
- Luka Dulčić - https://github.com/ldulcic

Named entity recognition or entity extraction refers to a data extraction task that is responsible for finding and classification words of sentence into predetermined categories such as the names of persons, organizations, locations, expressions of times, etc.

### Example:

> ”Android Inc. was founded in Palo Alto, California, in October 2003 by Andy Rubin, Rich Miner, Nick Sears, and Chris White.” 

> ”**ORG**(Android Inc.) **O**(was founded in) **LOC**(Palo Alto), **LOC**(California), **O**(in) **TIME**(October 2003) **O**(by) **PER**(Andy Rubin), **PER**(Rich Miner), **PER**(Nick Sears), **O**(and) **PER**(Chris White).“ 

In the project, we used the Python programming language and the Keras library. We tested different architectures of recurrent neural networks that use *LSTM* and *GRU* memory cells. We also performed various experiments in which we searched for the optimal parameters of the neural network with the intent to accurately recognize and classify name entities. 

Dataset: `CoNLL2003`

Labels:   `B-PER, I-PER, B-LOC, I-LOC, B-MISC, I-MISC, B-ORG, I-ORG, O`

Used libraries: `keras, spacy, numpy, scikit-learn, matplotlib`


## Results:

#### Model results where additional features has been not used:

| Model     | Precision     | Recall  | F1     |
| --------- |:-------------:| -------:| ------:|
| LSTM      | 0.6487        | 0.7527  | 0.6821 |
| GRU       | 0.6985        | 0.7589  | 0.7161 |
| BI-LSTM   | 0.7995        | 0.7098  | 0.7414 |
| BI-GRU    | 0.8529        | 0.7372  | 0.7861 | 

#### Model accuracy and loss without additional features
<img src="val-acc-no-features.png" width="425"/> <img src="val-loss-no-features.png" width="425"/> 

---


#### Model results where additional features has been used:

| Model     | Precision     | Recall  | F1     |
| --------- |:-------------:| -------:| ------:|
| LSTM      | 0.8123        | 0.8162  | 0.8138 |
| GRU       | 0.8085        | 0.8071  | 0.8056 |
| BI-LSTM   | 0.8408        | 0.8616  | 0.8493 |
| BI-GRU    | 0.8610        | 0.8604  | 0.8604 | 

#### Model accuracy and loss with additional features
<img src="val-acc-features.png" width="425"/> <img src="val-loss-features.png" width="425"/> 


