## Co-Reference

* Transformer + pointer network + distillation

### Architecture of Model

<p align="center">
<img src="data/co-model.png" width="350">
</p>

## Requirement

* TensorFlow 1.14

## Distillation

mkdir bert_dir which download Base Bert weights into

## Train Data

If want train data, please contact me


## Quality of Sentence ReWriting

|Precision|F1|
|--|--|
|96%|93%|

* [Rouge-l](https://github.com/ne7ermore/deeping-flow/blob/master/coreference/common.py#L153)

## Result

```
Utterance1：北京今天天气怎么样
Utterance2：晴天
Utterance3：那明天呢

Utterance3`：那明天天气怎么样呢
```


```
Utterance1：你有么有看过约会之夜啊
Utterance2：没有约会之夜谁演的
Utterance3：盖尔加朵
Utterance4：盖尔加朵1984年出生和我姐一样大
Utterance5：电影评分84看吗

Utterance5`：电影约会之夜评分84看吗
```

## Citation
If you find this code useful for your research, please cite:
```
@misc{TaoDeepingFlow
  author = {Ne7ermore Tao},
  title = {deeping-flow},
  publisher = {GitHub},
  year = {2020},
  howpublished = {\url{https://github.com/ne7ermore/deeping-flow/tree/master/coreference}}
}
```

## Contact
Feel free to contact me if there is any question (Tao liaoyuanhuo1987@gmail.com).
Tao Ne7ermore/ [@ne7ermore](https://github.com/ne7ermore)