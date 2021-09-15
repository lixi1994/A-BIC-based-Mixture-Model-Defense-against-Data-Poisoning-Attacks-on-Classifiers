# BIC-based-Mixture-Model-Defense-against-Data-Poisoning-Attacks-on-classifiers

dataset/full contains pre-processed feature matrices of TREC-05 corpus. Each feature matrix is named as Ham/Spam_Train/Test_#AttackHam_#AttackSpam.npy

learnedPara/full contains mixture model parameters learnt from the corresponding feature matrix.

Detection_2.py is the code for our BIC-based mixture model defense.

To reproduce the results
1. modify #AttackHamd and #AttackSpam passed to main() in Detection_2.py
2. run python Detection_2.py
