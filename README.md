# Sequence-to-Sequence G2P toolkit for PyTorch

Grapheme to Phoneme (G2P) is a function that generates pronunciations (phonemes) for words based on their written form (graphemes). 
It has an important role in automatic speech recognition systems, natural language processing and text-to-speech engines. 
This G2P model implements a transformer architecture on python [PyTorch](https://pytorch.org/) and  [FairSeq](https://fairseq.readthedocs.io/en/latest/).
This repo implements a G2P model with two APIs:
1. load_g2p_model: Loads the G2P model from disk.
2. decode_word: Outputs phonemes given a word. It optionally exposes phoneme stress information.

## Installation

This repo works on Python>=3.7.8 and uses poetry to install dependencies. Assuming `pyenv` and `poetry` is installed, the repo can be downloaded as follows:
```bash
cd g2p_seq2seq_pytorch/
pyenv virtualenv 3.7.8 g2p
pyenv activate g2p
poetry install
```

## Download the model

We provide a pretrained 3x3 layer transformer model with 256 hidden units [here](https://developer.cisco.com/fileMedia/download/5b20821d-f092-3b57-a438-546046ffaa61/).
The model should be named `20210722.pt`. Place the model file in the `g2p_seq2seq_pytorch/g2p_seq2seq_pytorch/models/` folder.

## How to use the APIs

```python
from g2p_seq2seq_pytorch.g2p import G2PPytorch
model = G2PPytorch()
model.load_model()
model.decode_word("amsterdam") # "AE M S T ER D AE M"
model.decode_word("amsterdam", with_stress=True) # "AE1 M S T ER0 D AE2 M"
```

## How to train/test the model

We use [CMUDict latest](https://github.com/cmusphinx/cmudict) for train and validation. Validation is ~10% of the total dataset. 
Note that CMUDict latest doesn't have any test splits. Note also that CMUDict latest has phoneme stress information.

We use [CMUDict PRONASYL 2007](https://sourceforge.net/projects/cmusphinx/files/G2P%20Models/phonetisaurus-cmudict-split.tar.gz/download)
test set for testing. Note that CMUDict PRONASYL 2007 doesn't have stress information.

1. Prepare the training/validation/test data for model ingestion. This step involves tokenization, 
   removing stop words and binarization of data
   
2. Train the model on the binarized data and generate predictions on the test data.

We cannot directly look at the output of the test evaluation results since the test set does not have the stress information. 
We have to remove that stress information from the generated output to directly compare to the test set. We do this since
we want the model to learn from the stress information even though we want to quantify it's performance on the test set.

```bash
cd scripts/
sh prepare-g2p.sh
sh train-and-generate.sh
```

## Evaluation of the model

We benchmarked the PyTorch model against the [CMUSphinx](https://github.com/cmusphinx/g2p-seq2seq) TensorFlow model with the following metrics: 
- Phonetic error rate (%): For each word, calculate the percentage of the total number of predicted phonemes that are correct when compared to the gold phonemes. Average this across all words. 
- Word error rate (%): For each word, compare the entire sequence of predicted phonemes to the gold phonemes. We calculate the percentage of words whose predicted phonemes are an exact match to the gold phonemes. 
- CPU Latency (milli-seconds): Time taken to execute the G2P function on a CPU instance.
- GPU Latency (milli-seconds): Time taken to execute the G2P function on a GPU instance.

| Architecture   | PER (%)  | WER (%)  | CPU Latency (ms)  | GPU Latency (ms)  |
|----------------|----------|----------|-------------------|-------------------|
| CMUSphinx      | 4.16     | 19.91    | 13.76             | -                 |
| PyTorch  | 5.26     | 23.80    | 10.19             | 5.41              |

More details on the benchmarking datasets can be found in our [blog post](https://blogs.cisco.com/developer/).