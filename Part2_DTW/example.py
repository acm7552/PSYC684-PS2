#!/usr/bin/env python

#This file is used to generate the no_rec.txt and the yes_rec.txt files used in the DTW_algorithm final.py. 

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("audio/No.wav") #Put the file needed to be converted here!
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(mfcc_feat)

with open("no_rec.txt", "w") as f: # <--- name output file here
     for row in mfcc_feat:
         f.write(" ".join(map(str, row)) + "\n")
