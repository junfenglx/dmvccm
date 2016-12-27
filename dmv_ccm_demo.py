#!/usr/bin/python

# test/demo source code for DMV/CCM  (Klein&Manning 2004)
# and its implementation by Franco M. Luque
# see: http://www.cs.famaf.unc.edu.ar/~francolq/en/proyectos/dmvccm

# Adjust corpus directory to point to "wsj" (or another corpus).
# must contain trees in MRG form.
# Exact directory is important.
# delete all other files (e.g. READMEs) from this directory.
# David Reitter, Aug 2013, reitter@psu.edu



import nltk
import nltk.corpus

from nltk.corpus import ptb


from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import DependencyCorpusReader


from dmvccm import ccm
from dmvccm import dmv
from dmvccm.dmvccm import DMVCCM
import wsj10
import dep.dwsj as dwsj


CORPUS_DIR = "/Users/dr/Downloads/DMV/wsj_comb/parsed/mrg/wsj"
CORPUS_DIR = "/home/xtof/corpus/WSJ/wsj_comb"
CORPUS_DIR = '/home/junfeng/ptb_wsj/mrg/wsj'

dependency_treebank = LazyCorpusLoader(
     'dependency_treebank', DependencyCorpusReader, '.*\.dp',
     encoding='ascii')

tb = wsj10.WSJ10(basedir=CORPUS_DIR)
#print tb.get_trees()
#print tb.raw()
tb.print_stats()
# WSJ10 doesn't suit for DMV and DMVCCM

dtb = dwsj.DepWSJ10(basedir=CORPUS_DIR)
#print tb.get_trees()
#print tb.raw()
dtb.print_stats()

print('CCM')
m1 = ccm.CCM(dtb)
print(m1)
m1.train(1)
m1.test()

print('DMV')
m0 = dmv.DMV(dtb)
m0.train(1)
s = 'DT NNP NN VBD DT VBZ DT JJ NN'.split()
(t, p) = m0.dep_parse(s)
# t.draw()
t.pretty_print()

# with 10 iterations
# Sentences: 7422
# Micro-averaged measures:
#   Precision: 60.5
#   Recall: 76.8
#   Harmonic mean F1: 67.6

print('DMVCCM')
m2 = DMVCCM(dtb)
m2.train(1)
m2.test()
