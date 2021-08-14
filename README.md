# Sequence-to-Sequence G2P toolkit for PyTorch

Grapheme to Phoneme (G2P) is a function that generates pronunciations (phonemes) for words based on their written form (graphemes). It has an important role in automatic speech recognition systems, natural language processing and text-to-speech engines. This tool uses a transformer model from [FairSeq]()

This repo implements a G2P model and APIs using the PyTorch framework. The two APIs expressed here are:
```
- load_g2p_model
- decode_word
```

The APIs also optionally expose phoneme stress information and beam search.

## Download the model

Download the model from this [link](https://cisco-my.sharepoint.com/:u:/p/vijayrk/EZWq-McCQdBOmu7GrGojWGwB1jsKfeN9xNICsaJmH4WrGg?e=oXhw22)
and place it in the `models/` directory.

## How to use the APIs

```python
from g2p_seq2seq_pytorch.g2p import G2PPytorch

model = G2PPytorch()
model.load_model('<path_to_model_checkpoint>', '<model_checkpoint_name>')
model.decode_word("test")
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
