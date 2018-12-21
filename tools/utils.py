# -*- coding: UTF-8 -*-

from __future__ import division
import os
import codecs
import math
import re
import itertools
import copy
import time
from collections import defaultdict


# =============================================================================
# ORIGINAL UTILS
# =============================================================================


# punctuation
punc_tag = {
    u'"':u'TRN',
    u'\'':u'APS',
    u',':u'UTR',
    u'.':u'NKT',
    u'\\':u'BSLH',
    u'/':u'SLH',
    u'-':u'DPH',
    u':':u'QNKT',
    u';':u'UNKT',
    u'?':u'SUR',
    u'!':u'LEP',
    u'«':u'ATRN',
    u'»':u'ZTRN',
    u'“':u'ATRN',
    u'”':u'ZTRN',
    u'``':u'ATRN',
    u"''":u'ZTRN',
    u'—':u'DSH',
    u'(':u'AZZ',
    u')':u'ZZZ',
    u'[':u'ADZ',
    u']':u'ZDZ',
    u'{':u'ASZ',
    u'}':u'ZSZ'}

# numerals
rex_num = re.compile('[\d]+[\.,‚]*[\d]*',re.U)

# time routings
def gettime(tmstr,tf='%Y-%m-%d, %H:%M:%S'):
    return time.strptime(tmstr,tf)

def saytime(tm=None,tf='%Y-%m-%d, %H:%M:%S'):
    if tm is None:
        tm = time.localtime()
    else:
        tm = time.localtime(gettime(tm))
    return time.strftime(tf,tm)

# ios
def get_lines(fn,enc='utf-8',strip=1,keep_emp=0,comm=None):
    lns = []
    for l in codecs.open(fn,'r',enc).readlines():
        #skip comments if required
        if comm:
            cidx = l.find(comm)
            if cidx+1:
                l = l[:cidx]
        #strip if required
        l = (strip and [l.strip()] or [l])[0]
        #skip empty lines if keep_emp is false
        if not (keep_emp or l): continue
        lns.append(l)
    return lns

# getting sentences from a file - specific action
def get_sens(fn,enc='utf-8',dlm='*_*',bad_tkn='?_?'):
    sens = []
    cs = []
    for l in get_lines(fn,enc,strip=1)+[dlm]:
        if l==dlm:
            if cs: sens.append(cs)
            cs = []
        elif not l==bad_tkn:
            cs.append(l)
    return sens


# ngram LM
class nglm():
    
    def __init__(self,N,seqs={},voc={},alpha=0.001,log=None):
        self.seqsize = N
        self.seqs = seqs
        self.voc = voc
        self.alpha = alpha
        self.log = log
        self.vocsize = len(voc)
    
    def build_ff(self,fn,enc='utf-8',dlm='\t',ndlm=' ',em='*'):
        for l in get_lines(fn,enc,strip=1,comm='#'):
            ents = l.split(dlm)
            seq,cnt = tuple(ents[:-1]),int(ents[-1])
            self.seqs[seq] = cnt
            pfx = seq[:-1]
            self.voc[pfx] = self.voc.get(pfx,0) + int(cnt)
        self.vocsize = self.seqsize==1 and len(self.seqs) or len(self.voc)

    def prb(self,s):
        p = 1.0*self.seqs.get(s,0) + self.alpha
        p /= (self.voc.get(s[:len(s)-1],0) + self.alpha*self.vocsize)
        return math.log(p)
    
    def chain_prb(self,seq):
        return sum([self.prb(s) for s in seq])

# text processing

# vowel regex
vwl = re.compile(u'[аәеёиоөуүұыіэюя]+',re.U|re.I)
def get_vowels(txt):
    return vwl.findall(txt)


# get ngrams from a sequence
def get_ngrams(N=1,seq=[],frnt=1,rear=1,atag='*'):
    if not seq:
        return []
    ret = []
    seq = frnt and (N-1)*['*'] + seq or seq
    seq = rear and seq + (N-1)*['*'] or seq
    for i in range(len(seq)-N+1):
        ret.append(tuple(seq[i:i+N]))
    return ret


# split morpheme into surface form and pos transition(s)
def split_morph(m,mdlm='_'):
    ms = {'sf':'?','lp':'?','rp':'?'}
    ents = m.split(mdlm)
    ms['sf'] = ents[0]
    try:
        ms['lp'] = ents[1]
    except:
        pass
    try:
        ms['rp'] = ents[2]
    except:
        pass
    return ms


# get a shallow parse
def make_shlw(seg,sdlm=' ',mdlm='_'):
    if seg=='*':
        return '*'
    mrphs = seg.split(sdlm)
    shlw = [mrphs[0]]
    lpos = split_morph(mrphs[0])['rp']
    for m in mrphs[1:]:
        ms = split_morph(m)
        if ms['rp']=='?':
            shm = mdlm.join([ms['sf'],lpos,lpos])
        else:
            shm = mdlm.join([ms['sf'],lpos,ms['rp']])
            lpos = ms['rp']
        shlw.append(shm)
    return sdlm.join(shlw)

        
# get sf of a parse
def get_parse_sf(p,sdlm=' ',mdlm='_',joiner=''):
    sf = []
    for m in p.split(sdlm):
        sf.append(split_morph(m)['sf'])
    return joiner.join(sf)


# get tag of a parse
def get_parse_tg(p,sdlm=' ',mdlm='_',joiner='-'):
    tg = []
    for m in p.split(sdlm):
        tg.append(mdlm.join(m.split(mdlm)[1:]))
    return joiner.join(tg)


# get segmentation of a parse
def get_parse_seg(p,sdlm=' ',mdlm='_'):
    return get_parse_sf(p,sdlm,mdlm,' ')


# get stem sf, tag and tag of the last IG
def split_stm_lig(txt,stm_sf=1,sdlm=' ',mdlm='_',stm_dlm=' ',tag_dlm='-'):
    mrphs = txt.split(sdlm)
    stm,stm_tg,lig = [],[],[]
    got_lig = 0
    for i in range(len(mrphs)-1,-1,-1):
        m = mrphs[i]
        ms = split_morph(m,mdlm)
        if (not got_lig) and (not ms['rp']=='?'):
            lig.insert(0,ms['rp'])
            got_lig = 1
        if got_lig:
            stm.insert(0,stm_sf and ms['sf'] or m)
            tg = ms['lp'] + (not ms['rp']=='?' and mdlm+ms['rp'] or '')
            stm_tg.insert(0,tg)
        else:
            lig.insert(0,ms['lp'])
    stm = len(stm)>1 and stm[:-1] or stm
    stm = (not stm_sf and sdlm or '').join(stm)
    stm = stm_sf and stm or mdlm.join(stm.split(mdlm)[:-1])
    return ''.join(stm),tag_dlm.join(stm_tg),tag_dlm.join(lig)


# get IG-s
def get_igps(txt,sdlm=' ',mdlm='_',ig_dlm='-'):
    root,parm = split_root_parm(txt,sdlm,mdlm)
    igps = [split_morph(root,mdlm)['rp']]
    igps_cp = [root]
    for m in parm.split(sdlm):
        ms = split_morph(m,mdlm)
        if not ms['sf']: continue
        if ms['rp']=='?':
            igps[-1] += ig_dlm + ms['lp']
            igps_cp[-1] += sdlm + m
        else:
            igps.append(ms['rp'])
            igps_cp.append(m)
    return igps,igps_cp


# get the root of an analysis
def get_root(txt,sdlm=' ',mdlm='_'):
    return split_morph(txt.split(sdlm)[0])


# split the root and the paradigm in a raw form (mixed with sf)
def split_root_parm(txt,sdlm=' ',mdlm='_'):
    ents = txt.split(sdlm)
    root = ents[0]
    parm = sdlm.join(ents[1:])
    return root,parm


# get pos paradigm
def get_pos_paradigm(txt,sdlm=' ',mdlm='_',joiner='-'):
    root,parm = split_root_parm(txt,sdlm,mdlm)
    #parm = [get_root(txt,sdlm,mdlm)['rp']]
    #for m in txt.split(sdlm)[1:]:
    #ms = split_morph(m,mdlm)
    #pe = ms['lp'] + (not ms['rp']=='?' and mdlm+ms['rp'] or '')
    #parm.append(pe)
    ret =  get_parse_tg(root,sdlm,mdlm)
    ret += parm and joiner + get_parse_tg(parm,sdlm,mdlm,joiner) or ''
    return ret


def get_cnts(fn,cdlm='_',log=None):
    cnts = {}
    if not fn:
        return cnts
    for l in get_lines(fn):
        try:
            ents = l.split(cdlm)
            sq = cdlm.join(ents[:-1])
            cnt = float(ents[-1])
        except:
            continue
        cnts[sq] = cnt
    return cnts


# =============================================================================
# HMM model
# =============================================================================

# HMM with deleted interpolation
class HMM_DI():

    def __init__(
            self,
            order=3,
            smoothing=None,
            count_delim='\t',
            sequence_delim='*_*',
            sequence_beg='<s>',
            sequence_end='</s>'):
        # N-gramm size
        self.order = order
        if self.order < 1 or self.order > 5:
            self.order = 1
        # smoothing vector (must be of length self.order + 2)
        self.smoothing = smoothing
        # counts delimeter (e.g. label N-gram<TAB>count)
        self.count_delim = count_delim
        # sequence (e.g. sentences) delimeter
        self.sequence_delim = sequence_delim
        # sequence start and end labels
        self.sequence_beg = sequence_beg
        self.sequence_end = sequence_end
        # state transistion table
        self.transitions = {}
        # state-observation table
        self.emissions = {}
        # vocabulary of states
        self.states = {}
        # vocabulary of observations
        self.observations = {}

    def load_model(self, model):
        ''' load a model from a file '''
        lines = open(model, 'r').readlines()
        # get the model's order
        self.order = int(lines[0].strip())
        # get smoothing vector - space separated
        self.smoothing = [float(lmb) for lmb in lines[1].strip().split()]
        # get the count delimiter (it supposed to be in quotes,
        # hence trimming one symbol from each side
        self.count_delim = lines[2].strip()[1:-1]
        # get sequence delimiter (again quotes trimming)
        self.sequence_delim = lines[3].strip()[1:-1]
        # get sequence beginning label (start state, default is '<s>')
        self.sequence_beg = lines[4].strip()[1:-1]
        # get sequence ending label (end state, default is '</s>')
        self.sequence_end = lines[5].strip()[1:-1]
        # get number of transitions
        N = int(lines[6].strip())
        for l in lines[7:7+N]:
            [transition, mle] = l.strip().split(self.count_delim)
            for state in transition.split():
                if state not in [self.sequence_beg, self.sequence_end]:
                    self.states[state] = 1
            self.transitions[tuple(transition.split())] = float(mle)
        # get number of state-observation pairs
        M = int(lines[7+N].strip())
        for i, line in enumerate(lines[8+N: 8+N+M]):
            [emission, mle] = line.strip().split(self.count_delim)
            tup = tuple(emission.split(' '))
            if len(tup) < 2:
                raise ValueError('Error loading model.\
                                 Line {}: too few state,\
                                 observation parameters'.format(i+7+N))
            elif len(tup) > 2:
                if len(tup) == 3 and not ''.join(tup[-2:]):
                    tup = [tup[0], ' ']
                else:
                    raise ValueError('Error loading model.\
                                     Line {}: consider removing spaces from\
                                     states or observations'.format(i+7+N))
            self.emissions[tup] = float(mle)

    def save_model(self, model):
        ''' load a model to a file '''
        fd = open(model, 'w')
        # zero-th line contains a single integer - the order of the model
        fd.write(f'{self.order}\n')
        # first line contains a single float - smoothing coefficient
        # (between 0 and 1)
        fd.write(f"{' '.join([str(lmb) for lmb in self.smoothing])}\n")
        # second line contains count delimiter enclosed into quotation marks
        fd.write(f'"{self.count_delim}"\n')
        # third line contains sequence delimiter enclosed into quotation marks
        fd.write(f'"{self.sequence_delim}"\n')
        # fourth line contains sequence beginning label
        # enclosed into quotation marks
        fd.write(f'"{self.sequence_beg}"\n')
        # fifth line contains sequence ending label
        # enclosed into quotation marks
        fd.write(f'"{self.sequence_end}"\n')
        # sixth line contains a single integer N -
        # number of state transition N-grams
        fd.write(f'{len(self.transitions)}\n')
        # next N lines contain state transition ngrams and
        # their frequences (MLE) delimitered by the count delimiter
        for transition, mle in sorted(
                self.transitions.items(),
                key=lambda x: x[1], reverse=True):
            fd.write(f"{' '.join(transition)}{self.count_delim}{mle:1.20f}\n")
        # next line contains a single integer M -
        # number of state-observation pairs
        fd.write(f'{len(self.emissions.values())}\n')
        # next M lines contain state state-observation pairs and
        # their frequences (MLE) delimitered by the count delimiter
        for emission, mle in sorted(
                self.emissions.items(),
                key=lambda x: x[1], reverse=True):
            fd.write(f"{' '.join(emission)}{self.count_delim}{mle:1.20f}\n")

    def train(
            self, trainfile,
            order=3,
            count_delim='\t',
            sequence_delim='*_*'):
        # N-gramm size
        self.order = int(order)
        if self.order < 1 or self.order > 5:
            self.order = 1
        # smoothing vector
        self.smoothing = (self.order + 2)*[0.0]
        # counts delimeter (e.g. label N-gram<TAB>count)
        self.count_delim = count_delim
        # sequence (e.g. sentences) delimeter
        self.sequence_delim = sequence_delim
        # state transistion table
        self.transitions = {}
        # state-observation table
        self.emissions = {}
        # vocabulary of states
        self.states = {}
        # vocabulary of observations
        self.observations = {}
        # read in the data and obtain counts
        transition_counts = {}
        emission_counts = {}
        input_length = 0.0
        buff = (self.order-1)*[self.sequence_beg]
        for line in open(trainfile, 'r').readlines():
            if not line.strip():
                continue
            input_length += 1
            # end of current sequence -
            # add sequence final markers and calc resulting transitions
            if line.strip() == self.sequence_delim:
                if self.order < 2:
                    continue
                for i in range(self.order - 1):
                    if buff:
                        ngrm = buff + [self.sequence_end]
                        for j in range(len(ngrm)):
                            pfx = tuple(ngrm[:len(ngrm)-j])
                            transition_counts[pfx] = transition_counts.get(
                                    pfx, 0.0) + 1
                    buff = buff[1:] + [self.sequence_end]
                buff = (self.order-1)*[self.sequence_beg]
                continue
            # get an observation-state pair
            [observ, state] = line.rstrip().split(self.count_delim)
            # update observations vocabulary
            self.observations[observ] = 1
            # update states vocabulary
            if state not in [self.sequence_beg, self.sequence_end]:
                self.states[state] = 1
            # count transistions
            ngrm = buff + [state]
            for i in range(len(ngrm)):
                pfx = tuple(ngrm[:len(ngrm)-i])
                transition_counts[pfx] = transition_counts.get(pfx, 0.0) + 1
            # count emissions
            emission_counts[state, observ] = emission_counts.get(
                    (state, observ), 0.0) + 1
            # update buffer
            if buff:
                buff = buff[1:] + [state]

        # compute MLEs and smoothing coeffcients for transitions
        lambdas = self.order*[0.0]
        for transition in transition_counts:
            if len(transition) < self.order:
                continue
            deleted = []
            for i in range(len(transition)):
                ngram = tuple(transition[:len(transition)-i])
                pfx = tuple(transition[:len(transition)-i-1])
                # calc mle
                self.transitions[ngram] = transition_counts.get(
                        ngram, 0.0)
                self.transitions[ngram] /= transition_counts.get(
                        pfx, input_length)
                # calc deleted mle
                if transition_counts.get(pfx, input_length) - 1 < 1:
                    deleted.insert(0, 0)
                else:
                    deleted.insert(0, transition_counts.get(
                            ngram, 0.0) - 1)
                    deleted[0] /= (transition_counts.get(
                            pfx, input_length) - 1)
            # adjust smoothing coefficients
            lambdas[deleted.index(
                    max(deleted))] += transition_counts[transition]
        # normalize and save smoothing coeffcients
        for i, lmb in enumerate(lambdas):
            self.smoothing[i] = lmb / sum(lambdas)

        # compute MLEs and smoothing coeffcients for emissions
        lambdas = 2*[0.0]
        for emission, count in emission_counts.items():
            [state, observ] = emission
            # calc mle
            self.emissions[emission] = count / transition_counts.get(
                    (state, ), count)
            # calc deleted mle
            if input_length > 1:
                deleted[0] = (transition_counts.get(
                        (state, ), count) - 1) / (input_length - 1)
            else:
                deleted[0] = 0
            if transition_counts.get((state, ), count) > 1:
                deleted[1] = (count - 1) / (
                        transition_counts.get((state, ), count) - 1)
            else:
                deleted[1] = 0
            deleted[1] = count - 1
            deleted[1] /= transition_counts.get((state, ), count) - 1
            deleted[1] = deleted[1] or 0
            lambdas[1 - int(deleted[0] > deleted[1])] += count
        # normalize and save smoothing coeffcients
        for i, lmb in enumerate(lambdas):
            self.smoothing[self.order+i] = lmb/sum(lambdas)

    def generate(self, observations):
        '''
        viterbi decoder
        '''
        def smoothed_emission(state, observ):
            if state == self.sequence_end:
                return 1.0
            ret = self.smoothing[self.order] * self.emissions.get(
                    (state, observ), 0)
            ret += self.smoothing[self.order+1] * self.transitions.get(
                    state, 0)
            return ret

        def smoothed_transition(states):
            ret = []
            for i in range(self.order):
                ret.append(self.smoothing[i] * self.transitions.get(
                        states[:i + 1], 0))
            return sum(ret)

        def backtrack(path):
            ret = []
            curstate = self.sequence_end
            for e in reversed(path):
                curstate = e[curstate]
                ret.insert(0, curstate)
            return ret

        ret = []
        # separate procedure for unigrams
        if self.order < 2:
            for o in observations:
                maxlike = [float('-inf'), None]
                for s in self.states:
                    like = smoothed_emission(s, o)
                    if like > maxlike[0]:
                        maxlike = [like, s]
                ret.append(maxlike[1])
            return ret

        # backpointers
        path = []
        # probabilities at previous and current steps
        prevporbs = {self.sequence_beg: math.log(1)}
        # states prefix
        state_pfx = (self.order-1) * [[self.sequence_beg]]
        # logarithm at zero
        LOGZERO = -1000
        for observ in observations + [self.sequence_end]:
            path.append({})
            currporbs = {}
            for state in (observ == self.sequence_end
                          and [self.sequence_end] or self.states):
                maxlogprob = float('-inf')
                for pfx in itertools.product(*tuple(state_pfx)):
                    tr_prob = smoothed_transition(pfx + (state, ))
                    em_prob = smoothed_emission(state, observ)
                    pp = prevporbs[pfx[-1]]
                    pp += tr_prob and math.log(tr_prob) or LOGZERO
                    pp += em_prob and math.log(em_prob) or LOGZERO
                    if maxlogprob < pp:
                        maxlogprob = pp
                        path[-1][state] = pfx[-1]
                currporbs[state] = maxlogprob
            state_pfx = state_pfx[1:] + [self.states.keys()]
            prevporbs = {k: v for k, v in currporbs.items()}
        return backtrack(path[1:])


# =============================================================================
# REGEX-based tokenizer
# =============================================================================

class TokenizeRex():
    '''
    Regex-based tokenizer. Fast, but does not perform sentence splitting.
    '''    
    def __init__(self):
        # regex that matches non-alphanums, non-hyphens, and non-spaces
        self.rex_split = re.compile(
                u'[^a-zа-яёәіңғүұқөһ\-\–\—\d\s]', re.U|re.I)
        # regex that matches leading untokenized hyphens, e.g. "Oh [-]yes"
        self.rex_hlead = re.compile(u'\s([\-\–\—]+)([^\s])', re.U)
        # regex that matches trailing untokenized hyphens, e.g. "yes[-] no"
        self.rex_htral = re.compile(u'([^\s])([\-\–\—]+)\s', re.U)
        # regex that matches *tokenized* multiple hyphen strings,
        # e.g. " [--] yes "
        self.rex_hmult = re.compile(u'\s([\-\–\—]{2, })\s', re.U)
    
    def tokenize(self, txt, lower=False):
        '''
        Returns a list of sentences. Each sentence is a list containing tokens.
        This particular implementation, however, 
        always returns a single sentence, i.e. performs no sentence splitting.
        If [lower] is True the input is lowercased *after* tokenization.
        '''
        # enclose non-alphanums, non-hyphens, and non-spaces into spaces
        spaced = self.rex_split.sub(' \g<0> ', txt)
        # detach hyphens
        dehyphened = self.rex_hlead.sub(' \g<1> \g<2>', u' %s'%spaced)
        dehyphened = self.rex_htral.sub('\g<1> \g<2> ', u'%s '%dehyphened)
        # break multi-hyphen strings
        while True:
            m = self.rex_hmult.search(dehyphened)
            if not m:
                break
            target = m.group(1)
            dehyphened = dehyphened[:m.start(1)]
            dehyphened += ' '.join(target) + dehyphened[m.end(1):]
        dehyphened = dehyphened.lower() if lower else dehyphened
        return [dehyphened.split()]


# =============================================================================
# HMM=based tokenizer
# =============================================================================

# character processing regex with replacements
CPREX = {
        # uppercase mathcer and replacer
        re.compile(u'[A-ZА-ЯЁӘІҢҒҮҰҚӨҺ]',re.U):'CAP',
        # lowercase mathcer and replacer
        re.compile(u'[a-zа-яёәіңғүұқөһ]',re.U):'LOW',
        # sentence-final punctuation matcher and replacer
        re.compile(u'[\.\?\!]',re.U):'SFL',
        # spaces (tab, whitespace, new line, carrier) matcher and replacer
        re.compile(u'\s',re.U):'SPC',
        # digit matcher and replacer
        re.compile(u'\d',re.U):'DIG'
        }


class TokenizerHMM():
    
    def __init__(self, implementation=HMM_DI, model=None):
        self.hmm = implementation()
        if model:
            self.hmm.load_model(model)
    
    def get_sequence(slef, txt):
        ret = []
        for c in txt:
            for rex, rep in CPREX.items():
                if rex.match(c):
                    c = rep
                    break
            ret.append(c)
        return ret
    
    def tokenize(self, txt, lower=False):        
        ret = []
        curr_sen = []
        curr_tok = []
        for i, label in enumerate(self.hmm.generate(self.get_sequence(txt))):
            char = txt[i]
            if label == 'S':
                if curr_tok:
                    curr_tok = ''.join(curr_tok)
                    curr_tok = curr_tok.lower() if lower else curr_tok
                    curr_sen.append(curr_tok)
                if curr_sen:
                    ret.append(curr_sen)
                    curr_sen = []
                curr_tok = [char]
            elif label == 'T':
                if curr_tok:
                    curr_tok = ''.join(curr_tok)
                    curr_tok = curr_tok.lower() if lower else curr_tok
                    curr_sen.append(curr_tok)
                curr_tok = [char]
            elif label == 'I':
                curr_tok.append(char)
            elif label == 'O':
                if curr_tok:
                    curr_tok = ''.join(curr_tok)
                    curr_tok = curr_tok.lower() if lower else curr_tok
                    curr_sen.append(curr_tok)
                curr_tok = []
        if curr_tok:
            curr_tok = ''.join(curr_tok)
            curr_tok = curr_tok.lower() if lower else curr_tok
            curr_sen.append(curr_tok)
            ret.append(curr_sen)
        return ret


# =============================================================================
# DD analyzer
# =============================================================================

class AnalyzerDD():

    def __init__(self, md={}, tm={}, sfx={},
                 unts=['R_X'],
                 prn_sgs=True, oov=False,
                 sdlm=' ', mdlm='_', log=None):
        self.md = md
        self.tm = tm
        self.sfx = sfx
        self.unts = unts
        # self.parms = parms
        self.plm = None
        self.prn_sgs = prn_sgs
        self.oov = oov
        self.sdlm = sdlm
        self.mdlm = mdlm
        self.log = log

    def load_model(self, mdl_dir):
        # build morpheme and transition dictionaries
        self.getff_md(os.path.join(mdl_dir, 'md'))
        self.getff_tm(os.path.join(mdl_dir, 'tm'))
        # build suffix-paradigm mappings
        self.getff_sfx(os.path.join(mdl_dir, 'sfx'))

    # get morpheme dictionary from file
    def getff_md(self, fn, enc='utf-8', dlm='\t', mdlm=None):
        mdlm = mdlm and mdlm or self.mdlm
        for l in get_lines(fn, enc, strip=1):
            # morpheme mapping
            [t1, t2] = l.split(dlm)
            self.md[t2] = self.md.get(t2, {})
            self.md[t2][t1] = 1

    # get transition dictionary from file
    def getff_tm(self, fn, enc='utf-8', dlm='\t', mdlm=None):
        mdlm = mdlm and mdlm or self.mdlm
        for l in get_lines(fn, enc, strip=1):
            # morpheme mapping
            [t1, t2] = l.split(dlm)
            self.tm[t1] = self.tm.get(t1, {})
            self.tm[t1][t2] = 1

    # get sufix-paradigm mapping from file
    def getff_sfx(self, fn, enc='utf-8', dlm='\t', mdlm=None):
        for l in get_lines(fn, enc, strip=1):
            [sf, sfx] = l.split(dlm)
            self.sfx[sf] = self.sfx.get(sf, {})
            self.sfx[sf][sfx] = 1

    # get tags for unsegmented inputs from file
    def getff_unts(self, fn, enc='utf-8'):
        self.unts = get_lines(fn, enc, strip=1)

    # returns segementation on shallow morphs
    def segment(self, pfx, ret={}, cpos='*', cseq=''):
        # roots must have at least one vowel
        # (achronyms are handled by the anlysis)
        if not (pfx and get_vowels(pfx)):
            return
        for m in self.tm.get(cpos, []):
            # check for root case
            if m.split(self.mdlm)[0] == 'R':
                # if we got a suitable root or oov roots are fine - save
                if self.oov or pfx in self.md[m]:
                    root = pfx + self.mdlm + m
                    if cseq:
                        new_anl = root + self.sdlm + cseq
                    else:
                        new_anl = root
                    ret[new_anl] = 1
                continue
            # iterate through the surface forms of the morpheme
            for msf in self.md[m]:
                # get full morpheme - with sf; and left pos
                mor = msf + self.mdlm + m
                # update morph. seq
                if cseq:
                    new_seq = mor + self.sdlm + cseq
                else:
                    new_seq = mor
                if pfx.endswith(msf):
                    # we got a suitable sf
                    new_pfx = pfx[:-1*len(msf)]
                    # no vowel in a candidate root - skip
                    if not get_vowels(new_pfx):
                        continue
                    # skip if we got unseen suffix and prune mode is on
                    if self.prn_sgs:
                        sf = get_parse_sf(new_seq,
                                                self.sdlm, self.mdlm, '')
                        tg = get_parse_tg(new_seq,
                                                self.sdlm, self.mdlm, '-')
                        if tg not in self.sfx.get(sf, []):
                            continue
                    # continue recursively into the depth
                    self.segment(new_pfx, ret, m, new_seq)

    # returns analyses including all root-word possibilities
    def analyze(self, tkn, top=0):
        # punctuation
        if tkn in punc_tag:
            tkn + self.mdlm + 'R_' + punc_tag[tkn]
            return False, [tkn + self.mdlm + 'R_' + punc_tag[tkn]]
        # numerals
        if not rex_num.sub('', tkn):
            return False, [tkn + '_R_SN']
        # get segmentations
        sgs = {}
        self.segment(tkn, sgs)
        anls = list(sgs.keys())
        # unsegmented input tag by special tags
        if not anls:
            for t in self.unts:
                anls.append(tkn + self.mdlm + t)

        return bool(sgs), anls


# =============================================================================
# HMM-based tagger
# =============================================================================

class TaggerHMM():

    def __init__(
            self,
            mode='IP', cw=3,
            lyzer=None, transi=None, emissi=None,
            lkp={}, pc={},
            sen_dlm='*_*', seg_dlm=' ', mor_dlm='_', mor_jnr='-',
            gen_dlm='\t', ng_dlm=' ', pc_dlm='~@~',
            em='*', ph='#', de='utf-8', log=None):

        # general stuff
        self.mode = mode
        self.cw = cw
        # analyzer and transition, emission LM objects
        self.lyzer = lyzer or AnalyzerDD()
        self.transi = transi
        self.emissi = emissi
        # init dictionaries
        self.lkp = lkp
        self.pc = pc
        self.own_lkp = {}
        # sentence dlm
        self.sen_dlm = sen_dlm
        # segementation dlm (default in parenthesis): men_R_SIM( )i_C4
        self.seg_dlm = seg_dlm
        # morpheme dlm (default in parenthesis): men(_)R(_)SIM i(_)C4
        self.mor_dlm = mor_dlm
        # morpheme joiner (default in parenthesis): R(-)SIM(-)C4
        self.mor_jnr = mor_jnr
        # general delimeter (default in parenthesis): wrd(\t)tag
        self.gen_dlm = gen_dlm
        # ngram delimeter (default in parenthesis): w1( )w2( )w3
        self.ng_dlm = ng_dlm
        # empty marker for N-grams (default in parenthesis): (*) (*) wrd
        self.em = em
        # pre-computed data delimeter (default in parenthesis): w(~@~)f1(~@~)f2
        self.pc_dlm = pc_dlm
        # pre-computed data header-mark (default in parenthesis): (# )h1~@~h2
        self.ph = ph
        # default encoding: utf-8
        self.de = de
        # set log file descriptor
        self.log = log

    def load_model(self, mdl_dir):
        # create a transition LM instance
        self.new_transi(
                # pass transition smoothing coefficient
                smth=0.01)
        # build the transition LM: depending on MODE consider
        # either IG- or paradigm-based transition
        tfn = (self.mode.count('I') and 'ligs.' or 'prms.')
        tfn += str(self.cw) + 'gram'
        self.buildff_transi(os.path.join(mdl_dir, tfn))
        # create an emission LM instance
        self.new_emissi(
                # pass emission smoothing coefficient
                smth=0.01)
        # build the emission LM: depending on MODE consider
        # either wrd|prm or stm-prm|IG probabilities
        efn = self.mode.count('P') and 'wrd_tag' or 'stm_lig'
        self.buildff_emissi(os.path.join(mdl_dir, efn))
        # populate the look-up dictionary
        self.getff_lkp(os.path.join(mdl_dir, 'lkps'))

    # LM ROUTINES: TRANSITION
    def new_transi(self, cw=None, smth=1, log=None):
        cw = cw or self.cw
        log = log or self.log
        self.transi = nglm(cw, {}, {}, smth, log)

    def buildff_transi(self, fn, enc=None, dlm=None, ndlm=None, em=None):
        enc = enc or self.de
        dlm = dlm or self.gen_dlm
        ndlm = ndlm or self.ng_dlm
        em = em or self.em
        self.transi.build_ff(fn, enc, dlm, ndlm, em)

    def set_transi(self, t):
        self.transi = t

    # LM ROUTINES: EMISSION
    def new_emissi(self, smth=1, log=None):
        log = log or self.log
        self.emissi = nglm(2, {}, {}, smth, log)

    def buildff_emissi(self, fn, enc=None, dlm=None, ndlm=None, em=None):
        enc = enc or self.de
        dlm = dlm or self.gen_dlm
        ndlm = ndlm or self.ng_dlm
        em = em or self.em
        self.emissi.build_ff(fn, enc, dlm, ndlm, em)

    def set_emissi(self, e):
        self.emissi = e

    # DICTIONARIES: LOOK-UP and PRE-COMPUTED DATA
    def getff_lkp(self, fn, enc=None, dlm=None):
        enc = enc or self.de
        dlm = dlm or self.gen_dlm
        for l in get_lines(fn, enc, strip=1):
            [sf, tg, cnt] = l.split(dlm)
            self.lkp[sf] = self.lkp.get(sf, []) + [tg]

    def getff_pc(self, fn, enc=None, pdlm=None, ph=None):
        enc = enc or self.de
        pdlm = pdlm or self.pc_dlm
        ph = ph or self.ph
        cats = {}
        for l in get_lines(fn, enc, strip=1):
            if l.startswith(ph):
                if not cats:
                    for i, cat in enumerate(l[1:].split(self.pc_dlm)):
                        cats[cat] = i
                continue
            elif not cats:
                continue
            vals = l.split(pdlm)
            lyses = vals[cats['[lyses]']:]
            self.pc[vals[0]] = lyses

    def tag_sentence(self, s):

        def vbi(anls):
            # pre- and append "*" - empty ngram chars
            shft = self.cw - 1
            # empty entry
            emp = {'anl': self.em, 'emprb': 0.0, 'trtag': self.em}
            ems = [[emp] for i in range(shft)]
            anls = ems + anls + ems
            # viterbi prob-paths list
            vp = []
            # initial prb-s and path
            for i in range(shft):
                vp.append([(0.0, (i+1)*[self.em])])
            # calculate max prb path
            for i in range(shft, len(anls)):
                cc = [[float('-inf'), ''] for e in range(len(anls[i]))]
                for tup in itertools.product(
                        *(range(len(anls[j])) for j in range(i-shft, i))):
                    for ca, a in enumerate(anls[i]):
                        prb, pth, seq = 0.0, [], ()
                        for k, t in enumerate(tup):
                            seq += (anls[i - shft + k][t]['trtag'], )
                            if not pth:
                                [prb, pth_copy] = vp[i - shft + k][t]
                                pth = copy.copy(pth_copy)
                            else:
                                pth.append(anls[i - shft + k][t]['anl'])
                        seq += (a['trtag'], )
                        pth.append(a['anl'])
                        prb += self.transi.prb(seq)
                        if cc[ca][0] < prb:
                            cc[ca] = [prb, pth]
                vp.append(cc)
            return vp[-1][0][-1][shft:-shft]

        # get all analyses
        anls = self.analyze_sentence(s)
        return vbi(anls)

    # analyse a sentnece
    def analyze_sentence(self, s):
        # analyze words
        anls = []
        for w in s:
            # check own look-up first
            wa = self.own_lkp.get(w, {})
            if wa:
                # already formatted for tagging - save and continue
                anls.append(wa)
                continue
            # if not in own look-up, check pc
            wa = self.pc.get(w, [])
            # if not in pc, check look-ups
            wa = wa and wa or self.lkp.get(w, [])
            # if still unlucky - analyze
            wa = wa and wa or self.lyzer.analyze(w)[-1]
            # format for tagging and save
            tmp = []
            for a in wa:
                fa = {'anl': a}
                wrd = get_parse_sf(a, self.seg_dlm, self.mor_dlm)
                tag = get_parse_tg(
                        a, self.seg_dlm, self.mor_dlm, self.mor_jnr)
                trtag = tag
                if self.mode.count('I'):
                    trtag = get_igps(
                            a, self.seg_dlm, self.mor_dlm, self.mor_jnr)[0][-1]
                    if self.mode == 'I':
                        wrd = a[:a.rfind(
                                trtag.split('-')[0])].rstrip(self.mor_dlm)
                        tag = trtag
                fa['trtag'] = trtag
                fa['emprb'] = self.emissi.prb((tag, wrd))
                tmp.append(fa)
            anls.append(tmp)
            # update own look-up
            self.own_lkp[w] = tmp
        return anls


# =============================================================================
# NB-based NER
# =============================================================================

class NB():

    def __init__(self, mdl_fn):
        self.classes = []
        self.lcontext = 3
        self.rcontext = 3
        self.trm_probs = defaultdict(float)
        if mdl_fn:
            self.load_model(mdl_fn)

    def load_model(self, fn):
        lines = open(fn, 'r').readlines()
        self.lcontext = int(lines[0])
        self.rcontext = int(lines[1])
        classes = {}
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            [trm, lbl, prb] = line.split('\t')
            self.trm_probs[trm, lbl] = float(prb)
            classes[lbl] = 1
        self.classes = sorted(list(classes.keys()))

    def save_model(self, fn):
        fd = codecs.open(fn, 'w', 'utf-8')
        fd.write(f'{self.lcontext}\n')
        fd.write(f'{self.rcontext}\n')
        for k, prb in self.trm_probs.items():
            [trm, lbl] = k
            fd.write('{}\n'.format('\t'.join(
                    map(lambda x: str(x), [trm, lbl, prb]))))

    def fex(self, anls):
        ret = {}
        for i in range(len(anls)):
            root = get_root(anls[i])
            lem = root['sf'].lower()
            pos = root['rp']
            if not pos == 'ZEQ':
                continue
            curdoc = {}
            # add left context features
            for j in range(i - self.lcontext, i):
                if j >= 0:
                    # print(utils.get_root(anls[j]))
                    root = get_root(anls[j])
                    lem = root['sf'].lower()
                    pos = root['rp']
                    curdoc[f'-{lem}'] = 1
                    curdoc[f'-{pos}'] = 1
            # current word features
            root = get_root(anls[i])
            lem = root['sf'].lower()
            pos = root['rp']
            curdoc[lem] = 1
            curdoc[f'+{lem[:2]}'] = 1
            curdoc[f'+{lem[-2:]}'] = 1
            # add right context features
            for j in range(i, i + self.rcontext):
                if j + 1 < len(anls):
                    root = get_root(anls[j + 1])
                    lem = root['sf'].lower()
                    pos = root['rp']
                    # curdoc[f'+{j + 1 - i}{lem}'] = 1
                    # curdoc[f'+{j + 1 - i}{pos}'] = 1
                    curdoc[f'+{lem}'] = 1
                    curdoc[f'+{pos}'] = 1
            ret[i] = list(curdoc.keys())
        return ret

    def predict(self, anls):
        ret = len(anls)*['_']
        for tokid, doc in self.fex(anls).items():
            prbs = defaultdict(float)
            for tok in doc:
                for c in self.classes:
                    prbs[c] += self.trm_probs.get(
                            (tok, c), self.trm_probs['<OOV>', c])
            # add priors
            mxp = (float('-inf'), self.classes[0])
            for c in self.classes:
                prbs[c] += self.trm_probs.get(('<PRR>', c), 0.0)
                if prbs[c] > mxp[0]:
                    mxp = [prbs[c], c]
            # normalize probabilities
            ret[tokid] = mxp[1]
        return ret


# =============================================================================
# Tagset manipulations
# =============================================================================

KLC2UD = {'ABE': 'Case=Abe',
          'C2': 'Case=Gen',
          'C3': 'Case=Dat',
          'C3SIM': 'Case=Dat',
          'C4': 'Case=Acc',
          'C5': 'Case=Loc',
          'C6': 'Case=Abl',
          'C7': 'Case=Ins',
          'C7SIM': 'Case=Ins',
          'CMP': 'Degree=Cmp',
          'EQU': 'Case=Equ',
          'ETB_ESM': 'vbType=Adj',
          'ETB_ETU': 'vbType=Ger',
          'ETB_KSE': 'vbType=Cvb',
          'ETK_ESM': 'vbType=Adj',
          'ETK_ETB': 'vbNeg=True',
          'ETK_ETU': 'vbType=Ger',
          'ETK_KSE': 'vbType=Cvb',
          'ETPK_ESM': 'vbType=Adj',
          'ETPK_ETB': 'vbNeg=True',
          'ETPK_ETU': 'vbType=Ger',
          'ETPK_KSE': 'vbType=Cvb',
          'ETP_ESM': 'vbType=Adj',
          'ETP_ETB': 'vbNeg=True',
          'ETP_ETU': 'vbType=Ger',
          'ETP_KSE': 'vbType=Cvb',
          'ET_ESM': 'vbType=Adj',
          'ET_ETB': 'vbNeg=True',
          'ET_ETU': 'vbType=Ger',
          'ET_KSE': 'vbType=Cvb',
          'LATT': 'Case=Latt',
          'M2': 'vbMood=Imp',
          'M3': 'vbMood=Desi',
          'M4': 'vbMood=Cond',
          'N1': 'Number=Pl',
          'N1S': 'Number=Pl',
          'P1': 'Person=1',
          'P2': 'Person=2',
          'P3': 'Person=3',
          'P4': 'Person=2F',
          'P5': 'Person=1Pl',
          'P6': 'Person=2Pl',
          'P7': 'Person=3',
          'P8': 'Person=2PlF',
          'R_APS': 'UPOS=PUNCT',
          'R_ATRN': 'UPOS=PUNCT',
          'R_AZZ': 'UPOS=PUNCT',
          'R_BOS': 'UPOS=X',
          'R_BSLH': 'UPOS=PUNCT',
          'R_DPH': 'UPOS=PUNCT',
          'R_ELK': 'UPOS=NOUN',
          'R_ET': 'UPOS=VERB',
          'R_ETB': 'UPOS=AUX',
          'R_ETD': 'UPOS=VERB',
          'R_ETK': 'UPOS=AUX',
          'R_ETP': 'UPOS=VERB',
          'R_ETPK': 'UPOS=AUX',
          'R_LEP': 'UPOS=PUNCT',
          'R_MOD': 'UPOS=ADJ',
          'R_NKT': 'UPOS=PUNCT',
          'R_OS': 'UPOS=INTJ',
          'R_QNKT': 'UPOS=PUNCT',
          'R_SE': 'UPOS=ADJ',
          'R_SH': 'UPOS=ADP',
          'R_SIM': 'UPOS=PRON',
          'R_SLH': 'UPOS=PUNCT',
          'R_SN': 'UPOS=NUM',
          'R_SUR': 'UPOS=PUNCT',
          'R_SYM': 'UPOS=SYM',
          'R_TRN': 'UPOS=PUNCT',
          'R_UNKT': 'UPOS=PUNCT',
          'R_US': 'UPOS=ADV',
          'R_UTR': 'UPOS=PUNCT',
          'R_X': 'UPOS=X',
          'R_ZE': 'UPOS=NOUN',
          'R_ZEQ': 'UPOS=PROPN',
          'R_ZHL': 'UPOS=CONJ',
          'R_ZTRN': 'UPOS=PUNCT',
          'R_ZZZ': 'UPOS=PUNCT',
          'S1': 'Poss=1',
          'S2': 'Poss=2',
          'S3': 'Poss=3',
          'S3SIM': 'Poss=3',
          'S4': 'Poss=2F',
          'S5': 'Poss=1Pl',
          '#S9': 'Poss=0',
          'SML': 'Case=Sml',
          'T1': 'vbTense=Aor',
          'T2': 'vbTense=Fut',
          'T2NEG': 'vbTense=Fut',
          'T3': 'vbTense=Pst',
          'T3E': 'vbTense=Pst',
          'V1': 'vbVcRefx=True',
          'V2': 'vbVcRefx=True',
          'V3': 'vbVcRefx=True',
          'V4': 'vbVcRefx=True'}


def klc2conll(surface, analysis, tokid, conllx=False):
    ret = (8*['_'] + 2*['0']) if conllx else 10*['_']
    ret[0] = str(tokid)
    ret[1] = surface
    morphs = analysis.split()
    gramms = [(m.split('_')[0], '_'.join(m.split('_')[1:])) for m in morphs]
    lemm = gramms[0][0]
    upos = KLC2UD.get(gramms[0][1],f'UPOS={gramms[0][1]}')
    upos = upos.split('=')[1]
    if upos == 'PROPN':
        lemm = surface[:len(lemm)]
    ret[2] = lemm
    ret[3:5] = [upos, upos]
    feats = []
    for gr in gramms[1:]:
        if gr[1] in KLC2UD:
            feats.append(KLC2UD[gr[1]])
    ret[5] = '|'.join(feats) or '_'
    return '\t'.join(ret)
    
