# Transformers-RNP
Predicting the mutation effect on the stability of Protein-RNA complex with deep learning-based model
![figure2](https://github.com/Jisheng-Liang/Transformers-RNP/assets/53801271/85821451-aa30-45b2-bb23-20507e88f567)

## Requirements and Environment
- onnxruntime
- numpy
- pandas
- [HH-suite](https://github.com/soedinglab/hh-suite) for generating HHblits files from protein sequences (with the file suffix of .hhm)
- [AlphaFold2_predicted_PDB](https://alphafold.ebi.ac.uk/download) for downloading PDB files as predicted by AlphaFold2.
- ONNX version of Transformers-RNP dowmload(https://pan.baidu.com/s/1s00MiMwoWc56mHP20-Jdtg?pwd=4j9u password：4j9u)

## Dataset
* The use of ProNAB Dataset has been approved by the author. ProNAB Dataset will not be disclosed in this repository. You may try your own dataset on training and testing.
* HHblits files should be placed in `hhblits` folder, including both the original and mutant Protein HHblits matrices.
* Split the dataset into three files - `train.csv`, `val.csv`, and `test.csv`. Each file contatins the original sequence, the mutant sequence, and the RNA sequence. A sample file has been provided as `sample.csv`.
## Training
```
python main.py
```

## Testing
```
python test.py
```

## Inference
# 1. Generate Space-HHBlits file from HH-suite and AlphaFold2 result
```
python data/preprocess/space-hhblits.py
```
# 2. Input all the information for inference
For example,
```
python inference.py --ori_hhb ORIGINAL_PATH --mut_hhb MUTANT_PATH --model_path ONNX_PATH --original MEATMDQTQPLNEKQVPNSEGCYVWQVSDMNRLRRFLCFGSEGGTYYIEEKKLGQENAEALLRLIEDGKGCEVVQEIKTFSQEGRAAKQEPTLFALAVCSQCSDIKTKQAAFRAVPEVCRIPTHLFTFIQFKKDLKEGMKCGMWGRALRKAVSDWYNTKDALNLAMAVTKY --mutation 307I --rna cacugcuucccuugacuagccu
```

## Citation
