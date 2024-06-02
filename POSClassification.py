import conll
import spacy
from spacy.tokens import Doc
import pandas as pd
from sklearn.metrics import classification_report
#########################################FUNCTIONS##########################################################
# Given a label and the mapping, convert the label
def label_conversion(iob, etype, mapping):
    # There is the possibility in which one of the two
    # part of the label is absent, so I managed that cases
    if(iob == ''):
        label = mapping[etype]
    elif(etype == ''):
        label = iob
    else:
        label = iob + '-' + mapping[etype]
    return label
# Function that calculates the accuracy
# accuracy = Right_pred / Total_labels
# returns the accuracy
# This method was the calculation of accuracy for 1.1
# before the clarification of the 28th on Piazza
def acc(doc, corpus, mapping):
    # Dictionaries that contains the values for 
    # good predictions 'ok' and total ones 'total'
    ok = {
        'O': 0,
        'I-LOC': 0,
        'B-LOC': 0,
        'I-PER': 0,
        'B-PER': 0,
        'I-ORG': 0,
        'B-ORG': 0,
        'I-MISC': 0,
        'B-MISC': 0
    }
    total = {
        'O': 0,
        'I-LOC': 0,
        'B-LOC': 0,
        'I-PER': 0,
        'B-PER': 0,
        'I-ORG': 0,
        'B-ORG': 0,
        'I-MISC': 0,
        'B-MISC': 0
    }
    for ref, pred in zip(corpus, doc.sents):
        for tt, pt in zip(ref,pred):
            # True label of the token
            tlabel = (tt[0].split(" "))[3]
            # Predicted label
            plabel = label_conversion(pt.ent_iob_, pt.ent_type_, mapping)
            if(plabel == tlabel):
                ok[tlabel] +=1
            total[tlabel] +=1
    accuracies = {}
    total_t = 0
    total_n = 0
    for key in ok:
        r = ok[key]/total[key]
        total_n += total[key]
        total_t +=ok[key]
        accuracies[key] = (r)
    accuracies['total'] = (total_t/total_n)
    ok['total'] = total_t
    total['total'] = total_n
    return accuracies, ok, total
# Evaluate spacy ner model using sklearn classification report
# This is the actual accuracy evaluation
def sklearn_acc(doc, corpus, mapping):
    refs = []
    hyps = []
    for ref, pred in zip(corpus, doc.sents):
        for tt, pt in zip(ref,pred):
            # True label of the token
            tlabel = (tt[0].split(" "))[3]
            # Predicted label
            plabel = label_conversion(pt.ent_iob_, pt.ent_type_, mapping)
            refs.append(tlabel)
            hyps.append(plabel)
    return classification_report(refs,hyps)
# Function that gives in output the references
def take_refs(corpus):
    def tuple_transformation(t):
        res = t[0].split(" ")
        return (res[0], res[3])
    return [[tuple_transformation(w) for w in s] for s in corpus]
# Function that gives in output the hypothesis
def take_hyps(mapping, doc):
    hyp = [[(w.text,(label_conversion(w.ent_iob_, w.ent_type_, mapping))) for w in s] for s in doc.sents]
    return hyp
# Function that groups named entities (as stated in the second request)
# Input: str/Doc
# Output: list(list(chunks_ents))
# EX:
#   input : "Apple's Steve Jobs died in 2011 in Palo Alto, California."
#   output: [['ORG', 'PERSON'], ['DATE'], ['GPE'], ['GPE']]
def ent_grouping(txt, is_string = True):
    # If is_string=True then the txt must be parsed
    if(is_string == True):
        doc = nlp(txt)
    else:
        doc = txt
    res = []
    # Variable in order to "jump" some loop cycles
    counter = 0
    for ent in doc.ents:
        # Flag that checks if an ent is found in a noun chunk
        flag = False
        if(counter==0):
            for chunk in doc.noun_chunks:
                c = []
                for ne in chunk.ents:
                    if(ent.text == ne.text):
                        flag = True
                        # Set the counter in order to avoid to add
                        # a group multiple times to the result
                        if(counter == 0):
                            counter = len(chunk.ents) - 1
                    c.append(ne.label_)
                if(c!= [] and flag==True):
                    res.append(c)
                    break
                elif(flag == False and counter == 0):
                    # This if covers the possibility that an entity
                    # is not in doc.noun_chunks
                    res.append([ent.label_])
                    break 
        else:
           counter = counter-1
    return res
# Function that fixes eventual segmentation errors
# Input: str
# Output: list of tuples
# Ex:
#   input : "Apple's Steve Jobs died in 2011 in Palo Alto, California."
#   output: [('Apple', 'B-ORG'), ("'s", 'O'), ('Steve', 'B-PERSON'), ('Jobs', 'I-PERSON'), ('died', 'O'), ('in', 'O'), ('2011', 'B-DATE'), ('in', 'O'), ('Palo', 'B-GPE'), ('Alto', 'I-GPE'), (',', 'O'), ('California', 'B-GPE'), ('.', 'O')]
def fix_segm(txt):
    doc = nlp(txt)
    # List of final ents of the doc
    res = []
    # List of tokens that must be ignored during the outer for-loop
    # in order to avoid "duplicant-ents"
    ignore = []
    for i, token in enumerate(doc):
        # This little section is for the correct labeling
        if(token.ent_iob_ == ''):
            label = token.ent_type_
        elif(token.ent_type_ == ''):
            label = token.ent_iob_
        else:
            label = token.ent_iob_ + '-' + token.ent_type_
        #####
        if(token.dep_ == 'compound'):
            for j, ent in enumerate(doc.ents):
                # if a compund is not in the same entity of it's head
                if((token in ent) and (not(token.head in ent))):
                    if(token.head == doc[i+1]):
                        res.append(spacy.tokens.Span(doc,i, i+1,label))
                        ignore.append(token.head)
                    elif(token.head == doc[i-1]):
                        ignore.append(token.head)
                        res.append(spacy.tokens.Span(doc,i-1, i,label))
                    break
                elif(token in ent):
                    ignore.append(token.head)
                    res.append(spacy.tokens.Span(doc,ent.start, ent.end,label))
                    break
        elif(token not in ignore):
            res.append(spacy.tokens.Span(doc,i, i+1,label))
    # Set the ents of the doc to the list of right ents
    doc.set_ents(res)
    return doc
# Takes in input a corpus taken from conll.read_corpus_conll()
# and executes the two points of the first request of the assignment
def firstRequest(corpus, mapping):
    docstart = base[0]
    # Remove the docstart lines
    tst_corpus = [sent for sent in base if(sent!=docstart)]
    words = [(w[0]).split(" ")[0] for s in tst_corpus for w in s]
    # Use spacy pipeline to process the text with custom words
    # in order to preserve the tokens
    doc = Doc(nlp.vocab, words=words)
    # Assure the right division in sentences
    # This index is for the position in the doc file (list of all the words)
    doc_pos = 0
    for s in tst_corpus:
        for i,w in enumerate(s):
            if(i==0):
                doc[doc_pos].is_sent_start = True
            doc_pos+=1
    # I ignored the parser for one simple reason:
    # spacy parser messes up the phrase division
    # I imposed in the for cycle above.
    # (For the first two requests doesn't matter 
    # because only extracts the dep_relations)
    for name, process in nlp.pipeline:
        if(process.name!='parser'):
            doc = process(doc)
    # First point
    hola = sklearn_acc(doc, tst_corpus, mapping)
    print(hola)
    # Second point
    # Take the references in the form of (token, NE_label)
    refs = take_refs(tst_corpus)
    #print(refs[0])
    # Now I must take the hypothesis in the form (token, NE_label)
    hyps = take_hyps(mapping, doc)
    #print(hyps[0])
    # So the problem is spacy that does something and changes the
    # Phrases because I imposed the same divisions but after processing
    # spacy changes them.
    # Calculate the requested table as shown in the lab's notebook
    results = conll.evaluate(refs, hyps)
    # Table
    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl.round(decimals=3)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(pd_tbl)
# Takes in input a string/doc and does the second request of the assignment
# If you want to input a Doc, set also is_string=False
def secondRequest(txt, is_string=True):
    res = ent_grouping(txt, is_string = True)
    # Dictionary with frequencies
    freq = {}
    for c in res:
        key = ""
        for i,t in enumerate(c):
            if(i!=(len(c)-1)):
                key = key + (t+",")
            else:
                key = key + (t)
        if(key in freq):
            freq[key] +=1
        else:
            freq[key] = 1
    # Sort the dictionary
    freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1])}
    print(res)
    print(freq)
# Function that takes in input a string and fixes the segmentation of
# the doc if needed
def thirdRequest(txt):
    res = fix_segm(txt)
    printed = []
    for ent in res.ents:
        printed.append((ent.text, ent.label_))
    print(printed)
########## First Request #######################
# Path of the file
tst_file = "data/test.txt"
nlp = spacy.load('en_core_web_sm')
# Get test corpus
base = conll.read_corpus_conll(tst_file)
# Mapping from spacy to original label
# Taken from OntoNotes Release 5.0
# TIME and DATE are mapped in 'O' due to the fact
# that in the dataset are mapped as 'O'
mapping = {
    "PERSON": "PER",
    "NORP": "MISC",
    "FAC": "MISC",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "O",
    "TIME": "O",
    "PERCENT": "O",
    "MONEY": "O",
    "QUANTITY": "O",
    "ORDINAL": "O",
    "CARDINAL":  "O"
}
firstRequest(base, mapping)
########################## Second request ###############################
test = "Apple's Steve Jobs died in 2011 in Palo Alto, California."
secondRequest(test)
########################## Third request ######################################
thirdRequest(test)
