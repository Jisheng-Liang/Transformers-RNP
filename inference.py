import onnxruntime
import numpy as np
import argparse
from dataset import pro_encoder, shhblits, na_encoder, pdb_encoder

def main():
    args = parse_args()
    device_name = args.devices
    ori_dir = args.ori_hhb
    mut_dir = args.mut_hhb

    # ori_seq = "MEATMDQTQPLNEKQVPNSEGCYVWQVSDMNRLRRFLCFGSEGGTYYIEEKKLGQENAEALLRLIEDGKGCEVVQEIKTFSQEGRAAKQEPTLFALAVCSQCSDIKTKQAAFRAVPEVCRIPTHLFTFIQFKKDLKEGMKCGMWGRALRKAVSDWYNTKDALNLAMAVTKYKQRNGWSHKDLLRLSHIKPANEGLTMVAKYVSKGWKEVQEAYKEKELSPETEKVLKYLEATERVKRTKDELEIIHLIDEYRLVREHLLTIHLKSKEIWKSLLQDMPLTALLRNLGKMTADSVLAPASSEVSSVCERLTNEKLLKKARIHPFHILVALETYKKGHGNRGKLRWIPDTSIVEALDNAFYKSFKLVEPTGKRFLLAIDVSASMNQRVLGSILNASVVAAAMCMLVARTEKDSHMVAFSDEMLPCPITVNMLLHEVVEKMSDITMGSTDCALPMLWAQKTNTAADIFIVFTDCETNVEDVHPATALKQYREKMGIPAKLIVCAMTSNGFSIADPDDRGMLDICGFDSGALDVIRNFTLDLI"
    # mut_point = "108A, 109A, 184A"
    # rna_seq = "ggcugguccgaaggcagugguugccaccauuaauugauuacagacaguuacagacuucuuuguucuucuccccucccacugcuucccuugacuagccu"

    ori_seq = args.original
    mut_point = args.mutation
    rna_seq = args.rna

    mut_array = mut_point.split(", ")

    mut_seq = list(ori_seq)
    for mut in mut_array:
        pos = int(mut[:-1])-1
        aa = mut[-1]
        mut_seq[pos] = aa
    mut_seq = "".join(mut_seq)

    ori_seq = pro_encoder([ori_seq]); mut_seq = pro_encoder([mut_seq])
    ori_hhb, mut_hhb = shhblits(ori_dir,mut_dir)
    rna = na_encoder([rna_seq])
    onnx_input = {'x_ori': ori_seq, 'x_mut': mut_seq, 'x_orihhb': ori_hhb, 'x_muthhb': mut_hhb, 'x_na': rna}

    if device_name == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device_name == 'cuda:0':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    model_path = args.model_path
    onnx_model = onnxruntime.InferenceSession('trans.onnx', providers=providers)
    outputs = onnx_model.run(None, onnx_input)
    print("ddG: " + str(outputs[0][0][0]) + "kcal/mol")

def parse_args():
    parser = argparse.ArgumentParser(description='Inference ddg')
    parser.add_argument(
        '--devices', type=str, default='cpu',
        help='use CPU or GPU for inference')
    parser.add_argument(
        '--ori_hhb', default='data/hhblits/pro_ori_13179.hhm',
        help='the repository to save Space_HHBlits files of the original sequence')
    parser.add_argument(
        '--mut_hhb', default='data/hhblits/pro_mut_13179.hhm',
        help='the repository to save Space_HHBlits files of the mutation sequence')
    parser.add_argument(
        '--model_path', default='trans.onns',
        help='the repository to save Transformers-RNP inference model')
    parser.add_argument(
        '--original', default=None,
        help='the original sequence of the protein')
    parser.add_argument(
        '--mutation', default=None,
        help='the mutation information of the protein')
    parser.add_argument(
        '--rna', default=None,
        help='the sequence of the RNA')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()