import pandas as pd
import numpy as np
import importlib
importlib.reload(fa)
import Func_automated_cloning as fa
import os

#Convert file to genebank format
input_file = './input_map.dna'
output_file = './output_map.gbk'

with open(input_file, 'rb') as input_file, open(output_file, 'w') as output_file:
    fa.snapgene_file_to_gbk(input_file, output_file)
    
# Example usage
input_file = './input_map.dna'
output_file = './output_map.gbk'
flank_5_prime = 'CGTTGGTGGGAGAAGAACGAACTGGTCACATCTCGTGGTAGCGGCGGTGGTGCGAGTGGTGGTGCG' #sequence directly upstream of insertion site, length doesn't matter --> make sure the plasmid map is saved in the correct orientation
flank_3_prime = 'GGCGGTAGCGCAGGCAGTGGAGCGGGCGGTAACACGATTAACATCGCTAAGAACGACTT'#sequence directly downstream of insertion site, length doesn't matter --> make sure the plasmid map is saved in the correct orientation

insert = df.loc[2, 'nt Sequence']  # DNA sequence to insert

fa.insert_sequence(input_file, output_file, flank_5_prime, flank_3_prime, insert)

#Example usage also adding a feature for the inserted sequence and different input files
genbank_files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.gb')]

start_ID = 225
IDs = []
for g in genbank_files:
    input_file = './pTA' + g + '.gb'
    for index, row in df['Substrate'].iterrows():

        output_file = './pTA' + str(start_ID) + '.gb'

        flank1 = row['flank1']
        flank2 = row['flank2']

        Seq = row['Seq']

        feature = row['Specification']

        fa.insert_sequence_with_feature(input_file, output_file, flank1, flank2, Seq, feature)
        IDs.append('pTA' + str(start_ID))
        start_ID += 1