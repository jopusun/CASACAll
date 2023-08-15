import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data

def dataset(dataset:str):
    '''
    dna_r9.4.1\n
    CP026085.1\n
    CP045741\n
    Tools\n
    Chiron
    '''
    if dataset == "dna_r9.4.1":
        input = torch.from_numpy(np.load("data/dna_r9.4.1/chunks.npy"))
        target = torch.from_numpy(np.load("data/dna_r9.4.1/ctcreferences.npy"))
        target_lengths = torch.from_numpy(np.int16(np.load("data/dna_r9.4.1/reference_lengths.npy")))
        valid_lengths = torch.from_numpy(np.int16(np.load("data/dna_r9.4.1/validation/reference_lengths.npy")))
        val_input = torch.from_numpy(np.load("data/dna_r9.4.1/validation/chunks.npy"))
        val_target = torch.from_numpy(np.load("data/dna_r9.4.1/validation/ctcreferences.npy"))
    if dataset == "dna_r9.4.1t":
        input = torch.from_numpy(np.load("data/dna_r9.4.1/chunks.npy")[:10000])
        target = torch.from_numpy(np.load("data/dna_r9.4.1/ctcreferences.npy")[:10000])
        target_lengths = torch.from_numpy(np.int16(np.load("data/dna_r9.4.1/reference_lengths.npy")[:10000]))
        valid_lengths = torch.from_numpy(np.int16(np.load("data/dna_r9.4.1/validation/reference_lengths.npy")))
        val_input = torch.from_numpy(np.load("data/dna_r9.4.1/validation/chunks.npy"))
        val_target = torch.from_numpy(np.load("data/dna_r9.4.1/validation/ctcreferences.npy"))

    if dataset == "CP026085.1":
        input = torch.from_numpy(np.load("data/Ecoli/CP026085.1/train/Chunk.npy"))
        target = torch.from_numpy(np.load("data/Ecoli/CP026085.1/train/Reference.npy"))
        target_lengths = torch.from_numpy(np.int16(np.load("data/Ecoli/CP026085.1/train/Reference_length.npy")))
        valid_lengths = torch.from_numpy(np.int16(np.load("data/Ecoli/CP026085.1/validation/Reference_length.npy")))
        val_input = torch.from_numpy(np.load("data/Ecoli/CP026085.1/validation/Chunk.npy"))
        val_target = torch.from_numpy(np.load("data/Ecoli/CP026085.1/validation/Reference.npy"))
    if dataset == "Tools":
        input = torch.from_numpy(np.load("data/Ecoli/Tools_polished/train/Chunk.npy"))
        target = torch.from_numpy(np.load("data/Ecoli/Tools_polished/train/Reference.npy"))
        target_lengths = torch.from_numpy(np.int16(np.load("data/Ecoli/Tools_polished/train/Reference_length.npy")))
        valid_lengths = torch.from_numpy(np.int16(np.load("data/Ecoli/Tools_polished/validation/Reference_length.npy")))
        val_input = torch.from_numpy(np.load("data/Ecoli/Tools_polished/validation/Chunk.npy"))
        val_target = torch.from_numpy(np.load("data/Ecoli/Tools_polished/validation/Reference.npy"))
    if dataset == "CP045741":
        input = torch.from_numpy(np.load("data/Ecoli/CP045741/train/Chunk.npy"))
        target = torch.from_numpy(np.load("data/Ecoli/CP045741/train/Reference.npy"))
        target_lengths = torch.from_numpy(np.int16(np.load("data/Ecoli/CP045741/train/Reference_length.npy")))
        valid_lengths = torch.from_numpy(np.int16(np.load("data/Ecoli/CP045741/validation/Reference_length.npy")))
        val_input = torch.from_numpy(np.load("data/Ecoli/CP045741/validation/Chunk.npy"))
        val_target = torch.from_numpy(np.load("data/Ecoli/CP045741/validation/Reference.npy"))
    if dataset == "Chiron":
        input = torch.from_numpy(np.load("data/Chiron/Chunk.npy")).float()
        target = torch.from_numpy(np.load("data/Chiron/Reference.npy"))
        target_lengths = torch.from_numpy(np.int16(np.load("data/Chiron/Reference_length.npy")))
        valid_lengths = torch.from_numpy(np.int16(np.load("data/Chiron/Reference_length.npy")))
        val_input = torch.from_numpy(np.load("data/Chiron/Chunk.npy")).float()
        val_target = torch.from_numpy(np.load("data/Chiron/Reference.npy"))
    torch_dataset = data.TensorDataset(input,target,target_lengths)
    val_dataset = data.TensorDataset(val_input,val_target,valid_lengths)
    return torch_dataset,val_dataset