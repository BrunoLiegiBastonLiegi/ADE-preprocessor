import torch, re
from transformers import AutoTokenizer




def find_sublist(l, pattern):
    """
    Parameters:
    l (list): input list 
    pattern (list): sublist to find in l

    Returns:
    span ((int,int)): span of pattern in l 
    """
    matches = []
    for i in range(len(l)):
        if l[i] == pattern[0] and l[i:i+len(pattern)] == pattern: 
            return (i, i+len(pattern))


        
def create_entity_labels(e_type, entity):
    """
    Parameters:
    e_type (list[str]): list of entity types, e.g. ['LOC','PER','DRUG']
    entity (list[int]): list containing the vocabulary index of each token forming the entity (Actually, simply the lenght of the list would be enough for this scope)

    Returns
    labels (list[str]): the entity labels in BIOES scheme, e.g. ['B-PER','I-PER','E-PER']
    """
    if len(entity) == 1:
        return ['S-' + e_type]
    else:
        return ['B-' + e_type] + ['I-' + e_type for i in range(len(adv_eff)-2)] + ['E-' + e_type]

        
def annotate_sent(tokenized_sent, entities, types):
    """
    Parameters:
    tokenized_sent (list[int]): sentence tokenized and mapped to vocabulary indexes using a pretrained tokenizer from transformers library 
    entities (list[list[int]]): list of all entities contained in tokenized_sent, each one tokenized using a pretrained tokenizer from transformers library 
    types (list[str]): list containing the relative type of each entity in entities

    Returns:
    sent (list[str]): sentence annotated with the BIOES scheme
    """
    e_labels = []
    for i in range(len(entities)):
        e_labels.append( { 'labels' : create_entity_labels(types[i], entities[i]), 'span' : find_sublist(tokenized_sent, entities[i]) } )
    e_labels.sort(key=lambda x: x['span'][0])
    sent = []
    i = 0
    while i < len(tokenized_sent):
        if len(e_labels) != 0:
            if i == e_labels[0]['span'][0]:
                sent += e_labels[0]['labels']
                i += len(e_labels[0]['labels'])
                e_labels.pop(0)
            else:
                sent += ['O']
                i += 1
        else:
            sent += ['O']
            i += 1
    return sent




######## MAIN ########



# input file to process
input_f = 'ADE-Corpus-V2/DRUG-AE.rel'

# extracting sentences and original entities annotations
sents = []
with open(input_f, 'r') as f:
    for x in f:
        split = re.split('\|', x)
        sents.append({ 'sentence' : split[1], 'AE' : split[2], 'DRUG' : split[5] })  # formatting: [sentence, adverse-effect, drug]
    
# import the pretrained tokenizer
model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model)

# loop over each sentence and write the results to file
with open(out_f, 'w') as f:
    for s in sents:
        print('### SENT\n',s)
        # We tokenize both the sentence and the entities, mapping also each token to the relative vocabulary index, in order to easily
        # find their position in the sentence using the find_sublist() function defined above.
        # The tokenized sentence and entities are then used, together with their relative entity type, inside the annotate_sent() function
        # to obtain the BIOES annotation of the sentence
        
        # tokenize the sentence
        tokenized_sent = tokenizer(s['sentence'])['input_ids'][1:-1] #remember to remove CLS and SEP!
        # tokenize the AE entity
        adv_eff = tokenizer(s['AE'])['input_ids'][1:-1]
        # tokenize the DRUG entity
        drug = tokenizer(s['DRUG'])['input_ids'][1:-1]
        # create the annotated sentence in BIOES scheme
        ann = annotate_sent(tokenized_sent, [adv_eff,drug], ['AE','DRUG'])
        f.write(s['sentence'] + '\|')
        [ f.write() ] 
        print('# ANNOTATION\n', ann)
    
    
    
