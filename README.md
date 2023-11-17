# BIC-based-Mixture-Model-Defense-against-Data-Poisoning-Attacks-on-classifiers

Codes for the paper "A BIC based Mixture Model Defense against Data Poisoning Attacks on Classifiers".

dataset/full contains pre-processed feature matrices of TREC-05 corpus. Each feature matrix is named as Ham/Spam_Train/Test_#AttackHam_#AttackSpam.npy

learnedPara/full contains mixture model parameters learnt from the corresponding feature matrix.

Detection_2.py is the code for our BIC-based mixture model defense.

To reproduce the results
1. modify #AttackHamd and #AttackSpam passed to main() in Detection_2.py
2. run python Detection_2.py

The short version of this paper is accepted by IEEE International Workshop on Machine Learning for Signal Processing (MLSP 2023).
The complete version of this paper is under the second round of review of IEEE Transactions on Knowledge and Data Engineering (TKDE).

Cite the short version:
@inproceedings{DBLP:conf/mlsp/LiMXK23,
  author       = {Xi Li and
                  David J. Miller and
                  Zhen Xiang and
                  George Kesidis},
  title        = {A BIC-Based Mixture Model Defense Against Data Poisoning Attacks on
                  Classifiers},
  booktitle    = {33rd {IEEE} International Workshop on Machine Learning for Signal
                  Processing, {MLSP} 2023, Rome, Italy, September 17-20, 2023},
  year         = {2023},
}
