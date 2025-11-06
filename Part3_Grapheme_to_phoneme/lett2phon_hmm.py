#!/usr/bin/python

import random
import sys
import math

# set this to 1 if you want to see more output
verbose = 0

#################################
#### LETTER-to-PHONEME RULES ####
#################################

# These are some very likely letter-to-phoneme mappings.
## TO DO ##
## You can add additional likely letter-to-phoneme correspondences here
## to improve the alignments from which you derive your mapping rules.

#baserules = {'F':'F'}

# Adding some base rules based on the Phoneme set at http://www.speech.cs.cmu.edu/cgi-bin/cmudict:
# Several ARPAbet consonents appear as only one letter or set of letters.
# Using this baserules decreases error from .7 to around ~.55 by itself which is pretty good. - AM
baserules = {   
             'B':'B',
             'D':'D',
             #'EE': 'IY', an entry like this won't do anything since this only works for single letters. -AM
             'F':'F',
             'G':'G',
             'M':'M',
             'N':'N',
             'P':'P',
             'R':'R',
             'T':'T',
             'V':'V',
             'Z':['Z','ZH'], # could be Z (as in zee) or ZH (as in seizure). -AM
            # may want to avoid using vowels, since it makes lots of assumptions we'd otherwise like to learn - AM.
            # did not help error. - AM
            #  'A': ['AE', 'AH', 'EY'],
            #  'E': ['EH', 'IY'],
            #  'I': ['IH', 'AY'],
            #  'O': ['OW', 'AO', 'AA'],
            #  'U': ['UW', 'AH', 'Y UW']
             }

# Global smoothing value for emission probabilities to avoid zero probability
SMOOTHING_VALUE = 1e-6

## TO DO ##
## Other ideas:
## set up some rules that allow a letter to map to one of several phones
## set up some conditional rules, e.g., C-> CH before H
## To implement these ideas, you will also need to update the calc_distance function!

# could be good if there were rules for letter pairs that always result in the given phoneme(s):
# otherrules = {'EE': 'IY',
#               'TH': ['DH','TH],
#   }

# the dict newrules stores the new mappings that your alignment will produce.
# This will be a dictionary mapping a letter to  a list of possible phonemes.
# Keys will be single letters, values will be lists of possible phonemes.
# We will use this set of new mapping rules to
# to guess the pronunciations of words we haven't seen.
newrules = {}

##################
### FUNCTIONS ####
##################

# This function compares a letter and phoneme to see how well they match

## TO DO ##
## You can change the way the distance is calculated when performing
## the alignment. 
## Ideas: 
## * assign different values (e.g., 2 instead of 1, .5 for matches
## that are possible but not super likely, etc.).
## * implement something to allow for conditional distances (e.g., 
## C:CH only before H, S:Z at the end of a word after a vowel)
## * add functionality to encourage vowel letters to align to vowel phonemes

def calc_distance(a, b):

    # If it's in a baserule, then the distance is 0 because they should probably match.
    # Otherwise, make the distance 2 because it's probably a very bad idea.
    if a in baserules.keys():
        if baserules[a] == b:
            return 0.0
        # if we put a list in baserules and the phoneme is in it, could return a small value - AM
        elif b in baserules[a]:
            return 0.5
        else:
            return 2.0
        
    # do the same for the trained rules, but even lower distances since these were learned, not assumed. - AM
    elif a in newrules.keys():
        if newrules[a] == b:
            return 0.0
        elif b in newrules[a]:
            return 0.25
        else:
            return 1.0  
    # Otherwise, it's not clear, so assign a smaller penalty
    else:
        return 1.0


# This function performs dynamic time warping to align
# the spelling of a word with the pronunciation for that word.
def getalignment(s1, s2):

    if verbose == 1:
        print("Aligning " + s1 + " to " + s2)

    spelling = list(s1)  # word spelling (letters)
    pronunciation = s2.split()  # word pronunciation (phonemes)

    n = len(pronunciation)
    N = len(spelling)

    D = [[0 for i in range(N)] for j in range(n)]
    BT = {}

    # This is the same code you used for you last problem set.
    for i in range(0,n):
        for j in range(0,N):
            mydist = calc_distance(spelling[j], pronunciation[i])
            if j==0 and i==0:
                D[0][0] = mydist
                BT[(0,0)] = (-1,-1) 
            else:
                mymin = 10000000
                if i==0:
                    mymin = D[i][j-1]
                    BT[(i,j)] = (i,j-1)
                elif j==0:
                    mymin = D[i-1][j]
                    BT[(i,j)] = (i-1,j)
                else:
                    mymin = min(D[i-1][j-1], D[i-1][j], D[i][j-1])
                    minval = D[i-1][j-1]
                    minidxs = (i-1, j-1)
                    if D[i-1][j] < minval:
                        minval = D[i-1][j]
                        minidxs = (i-1, j)
                    if D[i][j-1] < minval:
                        minval = D[i][j-1]
                        minidxs = (i, j-1)
                    BT[(i,j)] = minidxs
                D[i][j]=mydist + mymin

                            
    if verbose == 1:
        print("Overall distance is:" + str(D[n-1][N-1]))
            

    # Just a reminder...
    # spelling uses variables j,N
    # pronunciation uses variables i,n

    # This determines the alignment using the backtrace.
    # Substitutions get added to the set of "new" rules that you are learning.
    # Deletions also get as a rule X->NULL (i.e., a rule to delete X).
    startn = n-1
    startN = N-1

    while startn > -1 and startN > -1:
        backup = BT[(startn, startN)]
        if backup == (startn-1, startN-1):
            if verbose == 1:
                print("Substitution: "+ spelling[startN]+ " with "+ pronunciation[startn])
            if spelling[startN] in newrules.keys():
                newrules[spelling[startN]].append(pronunciation[startn])
            else:
                newrules[spelling[startN]] = [pronunciation[startn]]
        if backup == (startn-1, startN):
            if verbose == 1:
                print("Insert: "+ spelling[startN])
        if backup == (startn, startN-1):
            if verbose == 1:
                print("Delete: "+ spelling[startN])
            if spelling[startN] in newrules.keys():
                newrules[spelling[startN]].append("NULL")
            else:
                newrules[spelling[startN]] = ["NULL"]
        startn = backup[0]
        startN = backup[1]



# Levenshtein distance
# Used to calculate quality of your guessed pronunciations.
def levenshtein(s1, s2):
    # s1: J and j
    # s2: I and i

    J = len(s1)
    I = len(s2)

    D = [[0 for j in range(J)] for i in range(I)]

    for i in range(0,I):
        for j in range(0,J):
            mydist = 0
            if s1[j] != s2[i]:
                mydist = 1

            if j==0 and i==0:
                D[0][0] = mydist
            else:
                mymin = 10000000
                if i==0:
                    mymin = D[i][j-1]
                elif j==0:
                    mymin = D[i-1][j]
                else:
                    mymin = min(D[i-1][j-1], D[i-1][j], D[i][j-1])
                D[i][j]=mydist + mymin

    return D[I-1][J-1]



###################
#### MAIN PART ####
###################

#####################################
### Train letter-to-phoneme model ###
#####################################

# Pass 1: Create mappings
# Read in a training lexicon.
# f = open(sys.argv[1])
f = open('trainprons.txt')
for line in f:
    parts = line.strip().split("  ")
    w = parts[0]
    p = parts[1]

    # Align the letters to the phonemes with dtw.
    getalignment(w, p)

f.close()

# Go through list of mappings for each letter and
# calculate the probability of mapping from a letter
# to each of the possible phonemes it was aligned to
# in training.
probs = {}
for k,v in newrules.items():
    probs[k] ={}
    for i in set(v):
        iprob = v.count(i) / float(len(v))
        probs[k][i] = iprob

    if verbose == 1:
        print(k + "\t"),
        print(v)

# Pass 2: Create bigrams (HMM)

# Phoneme transition matrix
transition_counts = {}

f = open('trainprons.txt')
for line in f:
    parts = line.strip().split("  ")
    # Only grab correct pronunciations
    p = parts[1].strip().split() 
    
    # Padding for first phoneme transition
    p_sequence = ['<START>'] + p
    
    for i in range(len(p_sequence) - 1):
        p_i = p_sequence[i]
        p_j = p_sequence[i+1]
        
        # Initialize counts
        if p_i not in transition_counts:
            transition_counts[p_i] = {}
        if p_j not in transition_counts[p_i]:
            transition_counts[p_i][p_j] = 0
            
        # Count bigram
        transition_counts[p_i][p_j] += 1
f.close()

# Convert counts to probabilities
transition_probs = {}
for p_i, next_p_counts in transition_counts.items():
    total_count = sum(next_p_counts.values())
    
    transition_probs[p_i] = {}
    for p_j, count in next_p_counts.items():
        transition_probs[p_i][p_j] = count / float(total_count)
# Adding more proper smoothing for unseen transitions may further reduce error rate. Arbitrary smoothing is added in the testing section, but calculating smoothing values mathematically would result in more optimal probability estimates. 

####################################
### Test letter-to-phoneme model ###
####################################

# Use these mappings trained above to generate pronunciations for unseen words.

# Option 1: Randomly pick one of the mappings that was found
# by the aligner. Ones that were found more often will
# get picked more often. Do this ten times for every test word.

## TO DO ##
##
## If you added conditional mappings in your alignment, you 
## will need code to implement that here.
##
## You can also change the way you select the mapping, e.g.,
## * always pick the most frequent mapping
## * ignore infrequent mappings
## * ignore unlikely mappings (e.g., vowel:consonant mappings)
## * disallow sequences of more than one vowel in your output

# This stores the actual pron - guessed pron pairs to be evaluated.
# It is a list of tuples (actual pron, guessed pron)
results = [] 

# Get the list of all unique possible phonemes (HMM states)
all_phonemes = set()
for v in probs.values():
    all_phonemes.update(v.keys())
all_phonemes.discard('NULL') # Discard NULL as an emitted state

f = open('testprons.txt')
for line in f:
    parts = line.strip().split("  ")
    word = parts[0]
    pron = parts[1]

    # Viterbi initialization for first letter
    viterbi_scores = {}
    backpointers = {}
    L_0 = word[0]

    for p in all_phonemes:
        # P(Phoneme (P) | L_0)
        # Using log probabilities to avoid underflow
        emission_log_prob = -100.0
        if L_0 in probs and p in probs[L_0]:
             emission_log_prob = math.log(probs[L_0][p])
        elif L_0 in probs and 'NULL' in probs[L_0]:
             # Pass NULL, as current framework does not handle NULL values. Proper implementation of NULL probability handling could further reduce error rate
             pass 
        
        # P(<START> -> current phoneme (P_k))
        transition_log_prob = math.log(transition_probs.get('<START>', {}).get(p, SMOOTHING_VALUE)) 
        
        viterbi_scores[(0, p)] = emission_log_prob + transition_log_prob
        backpointers[(0, p)] = None

    # Viterbi recursion for each letter
    for i in range(1, len(word)):
        current_letter = word[i]
        
        for p_k in all_phonemes:
            
            # P(current phoneme (P_k) | current letter (L_i))
            emission_log_prob = -100.0
            if current_letter in probs and p_k in probs[current_letter]:
                emission_log_prob = math.log(probs[current_letter][p_k])
            
            max_score = -float('inf')
            best_prev_phoneme = None
            
            # Transition score P(current phoneme (P_k) | previous phoneme (P_j))
            for p_j in all_phonemes:
                
                # Use log transition prob
                transition_log_prob = math.log(transition_probs.get(p_j, {}).get(p_k, SMOOTHING_VALUE))
                
                # Total score for the current path
                current_score = viterbi_scores[(i-1, p_j)] + transition_log_prob
                
                # Save best score
                if current_score > max_score:
                    max_score = current_score
                    best_prev_phoneme = p_j
            
            # Final score for current phoneme at current letter
            viterbi_scores[(i, p_k)] = max_score + emission_log_prob
            backpointers[(i, p_k)] = best_prev_phoneme

    # Find highest score for last letter
    last_letter_index = len(word) - 1
    max_final_score = -float('inf')
    last_phoneme = None
    
    for p in all_phonemes:
        score = viterbi_scores[(last_letter_index, p)]
        if score > max_final_score:
            max_final_score = score
            last_phoneme = p

    # Use backtrace to construct the optimal phoneme sequence
    guess_sequence = []
    current_phoneme = last_phoneme
    for i in range(last_letter_index, -1, -1):
        if current_phoneme:
             guess_sequence.append(current_phoneme)
        # Find the previous phoneme from backpointers
        current_phoneme = backpointers[(i, current_phoneme)]
    
    # Create output guess
    guess_sequence.reverse()
    guess = " ".join(guess_sequence) + " "
    
    # Append single best guess
    results.append((pron, guess))


########################
### Evaluate outoput ###
########################

# Calculate the Levenshtein distance (min edit distance)
# between each pronunciation you guessed and the correct one.

# Final output is total distance over all pairs
# divided by the total number of phones in the correct prons.

totalphones = 0
totallev = 0
for pron,guess in results:
    lev = levenshtein(pron.strip().split(), guess.strip().split())
    print(pron + "\t\t" + guess + "\t\t" + str(lev))
    totallev += lev
    totalphones += len(pron.strip().split())



totalerr = float(totallev) / float(totalphones)
print("The total error is: " + str(totalerr))
f.close()

