# Tokens
ADENINE_TKN  = 'A'
CYTOSINE_TKN = 'C'
URACIL_TKN   = 'U'
GUANINE_TKN  = 'G'
THYMINE_TKN  = 'T'
INOSINE_TKN  = 'I'

ANY_NUCLEOTIDE_TKN = 'N'

# Tokens from https://en.wikipedia.org/wiki/FASTA_format
RNA_TOKENS = [ADENINE_TKN, CYTOSINE_TKN, GUANINE_TKN, THYMINE_TKN, INOSINE_TKN, "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", ANY_NUCLEOTIDE_TKN, "-"]
             #     5           6              7           8           9
#RNA_TOKENS = ['A', 'C', 'G', 'T', 'I', "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", ANY_NUCLEOTIDE_TKN, "-"]
             #  5    6    7    8    9

CLS_TKN  = "<cls>"  #0
PAD_TKN  = "<pad>"  #1
BOS_TKN  = "<bos>"  #没用
EOS_TKN  = "<eos>"  #2
UNK_TKN  = "<unk>"  #3
MASK_TKN = "<mask>" #4
