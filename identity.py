import re
'''
calculate identity of basecaller from CIGAR produced by minimap2

'''



#load pad file, CIGAR is right after cg tag(there is a blankspace in the end of CIGAR)
class identity_calculater():
    def __init__(self, called_base = "") -> None:
        self.called_base = called_base

    def minimap2_call(identity,ref_path = ""):
        #identity()

        pass
    def identity(self,paf_path = ""):
        file = open(paf_path)
        paf = file.readlines()
        M_total = 0
        ID_total = 0
        for line in paf:
            CIGAR = line[line.find("cg:Z:")+5:-1]
            M_idx = []
            ID_idx = []

            M = 0
            ID = 0
            i = 0
            #indexing symbols
            for s in CIGAR:
                if s.isdigit():
                    pass
                elif s == "M":
                    M_idx.append(i)
                    i+=1
                elif (s == "I") or (s == "D"):
                    ID_idx.append(i)
                    i+=1
                else:
                    print("unknow:",s)

            for i in M_idx:
                M += int(re.split(r"[A-Z]",CIGAR)[i])
            for i in ID_idx:
                ID += int(re.split(r"[A-Z]",CIGAR)[i])
            #identity = M/(M+ID)
            M_total += M
            ID_total += ID
        identity = M_total/(M_total+ID_total)
        return identity
    def identity_eqx(self,paf_path = ""):
        file = open(paf_path)
        paf = file.readlines()
        M_total = 0
        I_total = 0
        D_total = 0
        X_total = 0
        for line in paf:
            CIGAR = line[line.find("cg:Z:")+5:-1]
            CIGAR = CIGAR.replace("=","M")
            M_idx = []
            I_idx = []
            D_idx = []
            X_idx = []

            M = 0
            I = 0
            D = 0
            X = 0
            i = 0
            #indexing symbols
            for s in CIGAR:
                if s.isdigit():
                    pass
                elif s == "M":
                    M_idx.append(i)
                    i+=1
                elif (s == "I"):
                    I_idx.append(i)
                    i+=1
                elif (s == "D"):
                    D_idx.append(i)
                    i+=1
                elif (s == "X"):
                    X_idx.append(i)
                    i+=1
                else:
                    print("unknow:",s)

            for i in M_idx:
                M += int(re.split(r"[A-Z]",CIGAR)[i])
            for i in I_idx:
                I += int(re.split(r"[A-Z]",CIGAR)[i])
            for i in D_idx:
                D += int(re.split(r"[A-Z]",CIGAR)[i])
            for i in X_idx:
                X += int(re.split(r"[A-Z]",CIGAR)[i])
            #identity = M/(M+ID)
            M_total += M
            I_total += I
            D_total += D
            X_total += X
        
        identity = M_total/(M_total+D_total+X_total)
        return M_total,I_total,D_total,X_total,identity

M,I,D,X,identity = identity_calculater().identity_eqx('alignment.paf')

print(I/(M+I+D+X))
print(D/(M+I+D+X))
print(X/(M+I+D+X))
print(identity)
print(M/(M+I+D+X))