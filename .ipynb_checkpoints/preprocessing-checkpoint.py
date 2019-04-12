import os
import gzip
import numpy as np
from collections import defaultdict
from rna_utils import *

TRAIN_DIR = '/30000/training_sample_0/sequences.fa.gz'
TEST_DIR  = '/30000/test_sample_0/sequences.fa.gz'

class RBPDataLoader:
    
    # Init RBP-31 dataset path through hard coding
    def __init__(self):
        self.RBP31 = {
            10: '10_PARCLIP_ELAVL1A_hg19',
            11: '11_CLIPSEQ_ELAVL1_hg19',
            12: '12_PARCLIP_EWSR1_hg19',
            13: '13_PARCLIP_FUS_hg19',
            14: '14_PARCLIP_FUS_mut_hg19',
            15: '15_PARCLIP_IGF2BP123_hg19',
            16: '16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters',
            17: '17_ICLIP_HNRNPC_hg19',
            18: '18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome',
            19: '19_ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome',
            1: '1_PARCLIP_AGO1234_hg19',
            20: '20_ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome',
            21: '21_PARCLIP_MOV10_Sievers_hg19',
            22: '22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome',
            23: '23_PARCLIP_PUM2_hg19',
            24: '24_PARCLIP_QKI_hg19',
            25: '25_CLIPSEQ_SFRS1_hg19',
            26: '26_PARCLIP_TAF15_hg19',
            27: '27_ICLIP_TDP43_hg19',
            28: '28_ICLIP_TIA1_hg19',
            29: '29_ICLIP_TIAL1_hg19',
            2 : '2_PARCLIP_AGO2MNASE_hg19',
            30: '30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters',
            31: '31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters',
            3: '3_HITSCLIP_Ago2_binding_clusters',
            4: '4_HITSCLIP_Ago2_binding_clusters_2',
            5: '5_CLIPSEQ_AGO2_hg19',
            6: '6_CLIP-seq-eIF4AIII_1',
            7: '7_CLIP-seq-eIF4AIII_2',
            8: '8_PARCLIP_ELAVL1_hg19',
            9: '9_PARCLIP_ELAVL1MNASE_hg19'
        }
        self.RBP31_TRAIN_DIR = '/30000/training_sample_0/sequences.fa.gz'
        self.RBP31_TEST_DIR  = '/30000/test_sample_0/sequences.fa.gz'
        self.RBP31_INDEX_RANGE = list(range(1, 32))
        self.RNA_ONEHOT      = defaultdict(lambda: np.array([.25,.25,.25,.25]))
        self.RNA_ONEHOT["A"] = np.array([1,0,0,0])
        self.RNA_ONEHOT["C"] = np.array([0,1,0,0])
        self.RNA_ONEHOT["G"] = np.array([0,0,1,0])
        self.RNA_ONEHOT["U"] = np.array([0,0,0,1])
    
    def get_RBP31_npz_file_name(self,index):
        file_name = "./RBP31_npz/" + self.RBP31[index] + ".npz"
        return file_name
    
    def get_RBP31_index_list(self):
        return self.RBP31_INDEX_RANGE
      
    def get_RBP31_path(self, index, isTrain=True):
        try:
            RBP = self.RBP31[index]
            if isTrain:
                path = "RBP31/" + RBP + self.RBP31_TRAIN_DIR
            else:
                path = "RBP31/" + RBP + self.RBP31_TEST_DIR
        except:
            raise ValueError("Not exists index {} in RBP31".format(index))
        return path
    
    def get_RBP31_path_list(self,isTrain=True):
        path_list = []
        for idx in self.RBP31_INDEX_RANGE:
            path = self.get_RBP31_path(idx,isTrain)
            path_list.append(path)
        return path_list
    
    def get_RBP31_seq_label(self,path):
        label_list = []
        seq_list = []
        seq = ""
        with gzip.open(path, 'r') as fp:
            for line in fp:
                line = line.decode()
                
                if line[0] == '>':
                    name = line[1:-1]
                    posi_label = name.split(';')[-1]
                    label = posi_label.split(':')[-1]
                    label_list.append(int(label))
                    if len(seq):
                        seq_list.append(seq)
                        seq = ""
                else:
                    d = line[:-1].replace("T","U")
                    d = d.upper()
                    seq = seq + d
                
            if len(seq):
                seq_list.append(seq)
        
                
        return np.array(seq_list), np.array(label_list)
    
    def get_RBP31_data(self,index, isTrain=True):
        path = self.get_RBP31_path(index, isTrain)
        seq_list, label_list = self.get_RBP31_seq_label(path)
        print("Load RBP31 path: {}".format(path))
        
        # Onehot Encoding
        onehot_seq_list = []
        for i, seq in enumerate(seq_list):
            onehot_seq = np.array([self.RNA_ONEHOT[s] for s in seq])
            onehot_seq_list.append(onehot_seq)
            if i % 1000 == 0: 
                print("onehot encoding processing... {} / {}".format(i,len(seq_list)))
            
        onehot_seq_list = np.array(onehot_seq_list)
        
        # RNA Secondary Structure
        ss_list = []
        onehot_ess_list = []
        adj_list = []
        db_list = []
        for i, seq in enumerate(seq_list):
            
            ss         = get_RNA_secondary_structure_dot_bracket(seq)
            #ess        = dotbracket_to_elements(ss, max_len=len(ss))
            #onehot_ess = elements_structure_encoding(ess)
            #adj        = RNASecondaryStructure2AdjacencyMatrix(ss, max_len=len(ss))
            
            #ss_list.append(ss)
            #onehot_ess_list.append(onehot_ess)
            #adj_list.append(adj)
            db_list.append(ss)
            
            if i % 1000 == 0: 
                print("secondary structure encoding processing... {} / {}".format(i,len(seq_list)))
        
        #ss_list = np.array(ss_list)
        onehot_ess_list = np.array(onehot_ess_list)
        adj_list = np.array(adj_list)
        db_list = np.array(db_list)
        
        return label_list, onehot_seq_list, onehot_ess_list, adj_list, db_list
    
    def save_RBP31_data(self,index):
        Y_train , X_train , E_train , A_train, DB_train = data_loader.get_RBP31_data(index,isTrain=True)
        print("Preprocess complete - RBP 31 - index {}".format(index))
        print("file_name : {}".format(self.RBP31[index]))
        print("X-train   : {}".format(X_train.shape))
        print("A-train   : {}".format(A_train.shape))
        print("E-train   : {}".format(E_train.shape))
        print("Y-train   : {}".format(Y_train.shape))
        print("----------")
        
        Y_test  , X_test  , E_test  , A_test, DB_test  = data_loader.get_RBP31_data(index,isTrain=False)
        print("Preprocess complete - RBP 31 - index {}".format(index))
        print("file_name : {}".format(self.RBP31[index]))
        print("X-test    : {}".format(X_train.shape))
        print("A-test    : {}".format(A_train.shape))
        print("E-test    : {}".format(E_train.shape))
        print("Y-test    : {}".format(Y_train.shape))
        print("----------")
        
        file_name = self.get_RBP31_npz_file_name(index)
        np.savez(file_name , 
                 X_train = X_train , A_train = A_train , E_train = E_train, Y_train = Y_train, DB_train = DB_train,
                 X_test  = X_test  , A_test  = A_test  , E_test  = E_test , Y_test  = Y_test , DB_test  = DB_test)
        
        print("npz file saved as {}\n".format(file_name))
        
    def preprocess_RBP31(self):
        rbp31_list = self.get_RBP31_index_list()
        for index in rbp31_list:
            self.save_RBP31_data(index)
    
    def load_RBP31_data(self, index):
        file_name = self.get_RBP31_npz_file_name(index)
        print("Loading the dataset {} ...".format(file_name))
        
        npz = np.load(file_name)
        X_train = npz["X_train"]
        A_train = npz["A_train"]
        E_train = npz["E_train"]
        Y_train = npz["Y_train"]
        DB_train = npz["DB_train"]
        
        print("Load complete - RBP 31 - index {}".format(index))
        print("X-train   : {}".format(X_train.shape))
        print("A-train   : {}".format(A_train.shape))
        print("E-train   : {}".format(E_train.shape))
        print("Y-train   : {}".format(Y_train.shape))
        print("----------")
        
        X_test = npz["X_test"]
        A_test = npz["A_test"]
        E_test = npz["E_test"]
        Y_test = npz["Y_test"]
        DB_test = npz["DB_test"]
        
        print("Load complete - RBP 31 - index {}".format(index))
        print("X-test    : {}".format(X_test.shape))
        print("A-test    : {}".format(A_test.shape))
        print("E-test    : {}".format(E_test.shape))
        print("Y-test    : {}".format(Y_test.shape))
        print("----------")
        
        print("Load dataset {} complete".format(file_name))
        
        return (X_train,A_train,E_train,Y_train, DB_train),(X_test,A_test,E_test,Y_test, DB_test)
    
    def load_dataset_RBP31_test(self):
        rbp31_list = self.get_RBP31_index_list()
        for index in rbp31_list:
            self.load_RBP31_data(index)
    
if __name__ == "__main__":
    data_loader = RBPDataLoader()
    data_loader.preprocess_RBP31()
    data_loader.load_dataset_RBP31_test()
    
    
    
    
    
    
    
    
    