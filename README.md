# Transformers-RNP
Predicting the mutation effect on the stability of Protein-RNA complex with deep learning-based model
![figure2](https://github.com/Jisheng-Liang/Transformers-RNP/assets/53801271/85821451-aa30-45b2-bb23-20507e88f567)

## Requirements and Environment
- python == 3.10
- pytorch == 1.12.1
- [HH-suite](https://github.com/soedinglab/hh-suite) for generating HHblits files from protein sequences (with the file suffix of .hhm)

## Dataset
* The use of ProNAB Dataset has been approved by the author. ProNAB Dataset will not be disclosed in this repository. You may try your own dataset on training and testing.
* HHblits files should be placed in `hhblits` folder, including both the original and mutant Protein HHblits matrices.
* Split the dataset into three files - 'train.csv', 'val.csv', and 'test.csv'. Each file contatins the original sequence, the mutant sequence, and the RNA sequence.
## Training
```
python main.py
```

## Testing
```
python test.py
```
## Citation
