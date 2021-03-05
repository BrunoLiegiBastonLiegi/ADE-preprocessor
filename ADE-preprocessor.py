import torch, re, json, pickle
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from itertools import product



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
            matches.append((i, i+len(pattern)))#return (i, i+len(pattern))
    return matches




# ------------------------ NER tagging schemes ----------------------------------------



class Scheme(ABC):
    
    @abstractmethod
    def __init__(self):
        pass

    @property
    def space_dim(self):
        return len(self.tag2index)
        
    def to_tensor(self, *tag, index=False):
        if index:
            return torch.tensor([ self.tag2index[i] for i in tag ]) # return only index of the class
        else:
            t = torch.zeros((len(tag),self.space_dim))
            j = 0
            for i in tag:
                t[j][self.tag2index[i]] = 1
                j += 1
            return t                                               # return 1-hot encoding tensor
        
    def to_tag(self, tensor, index=False):
        if index:
            return int(torch.argmax(tensor))
        else:
            return self.index2tag[int(torch.argmax(tensor))]

    @abstractmethod
    def tag(self, entity, e_type):
        pass
    
    
class BIOES(Scheme):
    
    def __init__(self, entity_types):
        self.e_types = entity_types
        self.tag2index = {}
        i = 0
        for e in self.e_types:
            self.tag2index['B-' + e] = i
            i +=1
            self.tag2index['I-' + e] = i
            i +=1
            self.tag2index['E-' + e] = i
            i +=1
            self.tag2index['S-' + e] = i
            i +=1
        self.tag2index['O'] = i
        self.index2tag = {v: k for k, v in self.tag2index.items()}

    def tag(self, entity, e_type):
        if len(entity) == 1:
            return ['S-' + e_type]
        else:
            return ['B-' + e_type] + ['I-' + e_type for i in range(len(entity)-2)] + ['E-' + e_type]
        


        

# ------------------------ Annotation object ----------------------------------------

        


class Annotation(object):
    
    def __init__(self, sent, entities, relations, entity2embedding, tokenizer=None, ner_scheme=None):
        
        self.sent = {'sentence': sent}
        self.ent2emb = entity2embedding
        
        # tokenizer
        if tokenizer != None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
        self.sent['tokenized'] = self.tokenizer(self.sent['sentence'], add_special_tokens=False)['input_ids']

        # NER tagging scheme
        if ner_scheme != None:
            self.scheme = ner_scheme
        else:
            self.scheme = BIOES(list(dict.fromkeys(list(entities.values()))))

        # entities
        self.entities = {}
        duplicates_check = {} # check for entities with the same tokenization but different span
        for k, v in entities.items():
            tokenized_ent = self.tokenizer(k, add_special_tokens=False)['input_ids']
            try:
                duplicates_check[str(tokenized_ent)] += 1
            except:
                duplicates_check[str(tokenized_ent)] = 0
            span = find_sublist(self.sent['tokenized'], tokenized_ent)[duplicates_check[str(tokenized_ent)]]
            tag = self.scheme.tag(tokenized_ent, v)
            self.entities[k] = {'type': v, 'tokenized': tokenized_ent, 'span': span, 'tag': tag, 'embedding': self.ent2emb[k]}
        # check for overlapping entities
        self.overlapping_entities = False
        for v in self.entities.values():
            span1 = list(range(v['span'][0], v['span'][1]))
            for w in self.entities.values():
                if v != w:
                    span2 = list(range(w['span'][0], w['span'][1]))
                    if len(find_sublist(span1, span2)) != 0 or len(find_sublist(span2, span1)) != 0 :
                        self.overlapping_entities = True
            
        # relations
        possible_relations = list(product(self.entities.keys(), self.entities.keys())) # all possible relations
        [ possible_relations.pop(i*len(self.entities.keys())) for i in range(len(self.entities.keys())) ]
        self.relations = { k: {
            'type': v,
            'tokenized': (self.entities[k[0]]['tokenized'], self.entities[k[1]]['tokenized']),
            'span': (self.entities[k[0]]['span'], self.entities[k[1]]['span'])
        } for k, v in relations.items() }
        for r in possible_relations: # add the NO_RELATIONS 
            try:
                tmp = self.relations[r]
            except:
                self.relations[r] = {
                    'type': 'NO_RELATION',
                    'tokenized': (self.entities[r[0]]['tokenized'], self.entities[r[1]]['tokenized']),
                    'span': (self.entities[r[0]]['span'], self.entities[r[1]]['span'])
                }
                
        # tag sentence
        self.sent['tag'] = []
        e_sorted = list(self.entities.values())
        e_sorted.sort(key=lambda x: x['span'][0])
        i = 0
        while i < len(self.sent['tokenized']):
            if len(e_sorted) != 0 and i == e_sorted[0]['span'][0]:
                tmp = e_sorted.pop(0)
                self.sent['tag'] += tmp['tag']
                i += len(tmp['tag'])
            else:
                self.sent['tag'] += ['O']
                i += 1

        self.annotation = {'sentence': self.sent, 'entities': self.entities, 'relations': self.relations}

    def get_annotation(self):
        assert self.overlapping_entities == False, "Expected non-overlapping entities"
        return self.annotation

    def json(self, of=None, **kwargs):
        tmp = { str(k): v for k,v in self.annotation['relations'].items() }
        conversion = self.annotation
        conversion['relations'] = tmp
        if of == None:
            print(json.dumps(conversion, **kwargs))
        else:
            json.dump(conversion, fp=of, **kwargs)




    

######## MAIN ########


# import the pretrained tokenizer
model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model)

# input file to process
input_f = 'ADE-Corpus-V2/DRUG-AE.rel'

# load entity graph embeddings
# these were obtained by disambiguation with metamap
# combined with some pretrained Methathesaurus embeddings
# (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6568073/)
with open('entity2embedding.pkl', 'rb') as f:
    entity2embedding = pickle.load(f)

sents = {}
# extracting sentences and original entities annotations
with open(input_f, 'r') as in_f:
    for x in in_f:
        split = re.split('\|', x)
        try:
            tmp = sents[split[1]]
            sents[split[1]]['entities'][split[2]] = 'AE'
            sents[split[1]]['entities'][split[5]] = 'DRUG'
            sents[split[1]]['relations'][(split[2], split[5])] = 'ADVERSE_EFFECT_OF'
        except:
            sents[split[1]] = {'entities': {split[2]: 'AE', split[5]: 'DRUG'},
                               'relations': {(split[2], split[5]): 'ADVERSE_EFFECT_OF'}}

# build new annotations
ann = []
c = 0 # counter for sentences with overlapping entities
for s in sents.keys():
    
    try:
        a = Annotation(sent=s,
                       entities=sents[s]['entities'],
                       relations=sents[s]['relations'],
                       entity2embedding=entity2embedding,
                       tokenizer=tokenizer)
        ann.append(a.get_annotation())
        #a.json(indent=4)
    except:
        c += 1

print('> Found ', c, 'sentences with overlapping entities or missing entity graph embeddings, discarded them.')
        

# pickle everything
out_f = 'DRUG-AE_BIOES.pkl'
with open(out_f, 'wb') as o_f:
    pickle.dump(obj=ann, file=o_f)
            
            
#for s in sents.keys():
 #   ann = Annotation(sent=s, entities=sents[s]['entities'], relations=sents[s]['relations'], tokenizer=tokenizer).json(indent=4)

#s = 'Lithium treatment was terminated in 1975 because of lithium intoxication with a diabetes insipidus-like syndrome.'
#s = 'In patients with swallowing dysfunction and pneumonia, a history of mineral oil use should be obtained and a diagnosis of ELP should be considered in the differential diagnoses if mineral oil use has occurred.'
#s = 'These in vitro findings and clinical course suggest that TRAb/TBII without thyroid-stimulating activity may develop in patients with amiodarone-induced destructive thyroiditis, as reported in patients with destructive thyroiditis, such as subacute and silent thyroiditis.'
#s = 'We describe a case of disseminated muscular cysticercosis followed by myositis (fever, diffuse myalgia, weakness of the lower limbs, and inflammatory reaction around dying cysticerci) induced by praziquantel therapy, an event not described previously.'
#ss = sents[s]
#print(ss)
#Annotation(s,entities=ss['entities'],relations=ss['relations'],tokenizer=tokenizer).json(indent=4)


# k-fold cross validation
from sklearn.model_selection import KFold
import numpy as np

k = 10
kf = KFold(n_splits=k, shuffle=True)

ann = np.array(ann)
folds = {}
f = 0
for train_index, test_index in kf.split(ann):
    folds['fold_' + str(f)] = { 'train': ann[train_index], 'test': ann[test_index] }
    f += 1

# pickle everything
out_f = 'DRUG-AE_BIOES_' + str(k) + '-fold.pkl'
with open(out_f, 'wb') as o_f:
    pickle.dump(obj=folds, file=o_f)
