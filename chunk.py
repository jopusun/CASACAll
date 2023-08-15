from signal import Signals
import h5py
import os
import numpy as np
import torch
filepath =[]
# Getting the current work directory (cwd)
thisdir = '/media/jopu/Data/jopu/tools_raw_data/TOCR21PDG00193_xBacGS_RawData/TOCR21PDG00193_1xBacGS/raw_data/DH5a/split/0/'

# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if file.endswith(".fast5") :
            filepath.append(os.path.join(r, file))




Chunks = []

class Chunktool():
    def __init__(self,filepath,chunk_size = 3600) -> None:
        super().__init__()
        self.filepath = filepath
        try:
            fast5_data = h5py.File(filepath, 'r')
        except IOError:
            assert IOError('Error opening file. Likely a corrupted file.')
        self.chunk_size = chunk_size
        self.HDFile = fast5_data
        self.event = fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
        self.raw_start = fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
    def load_file():
        pass
    def ChunkRaw(self,container):
        #todo:normalize
        
        rawpath = 'Raw/Reads/'
        fast5_data = self.HDFile
        chunk_size = self.chunk_size
        raw_start = self.raw_start
        try:
            raw = fast5_data['Raw/Reads/']
            for name in raw.keys():
                rawpath = rawpath+name

            signal = fast5_data[rawpath+'/Signal']
            for x in range(0,len(signal[()][raw_start:]),chunk_size):
                Chunk = signal[()][raw_start+x:raw_start+x+chunk_size]    
                if len(Chunk)==chunk_size:
                    container.append(Chunk)
                    print(np.size(Chunk))
            print(np.shape(container))
        except:  
            pass
    def ChunkRef(self):
        print()
        print(self.HDFile['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'][0][2])
        
    def QuilityBar(self):
        #self.HDFile['Analyses/Basecall_1D_000/Summary/basecall_1d_template'].attrs['mean_qscore']
        pass
    
Chunker = Chunktool(thisdir+'0a84b13f-c560-4fa9-b8d6-6a9ad498038c.fast5')

Chunker.ChunkRef()
#print(raw.keys())