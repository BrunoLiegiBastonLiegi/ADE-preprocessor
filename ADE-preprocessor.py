import torch, re, json, pickle
from transformers import AutoTokenizer
from abc import ABC, abstractmethod



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
    
    def __init__(self, sent, entities, relations, tokenizer=None, ner_scheme=None):
        
        self.sent = {'sentence': sent}
        
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
            self.entities[k] = {'type': v, 'tokenized': tokenized_ent, 'span': span, 'tag': tag}
            
        # relations
        self.relations = { k: {'type': v, 'tokenized':(self.entities[k[0]]['tokenized'], self.entities[k[1]]['tokenized']), 'span':(self.entities[k[0]]['span'], self.entities[k[1]]['span']) } for k, v in relations.items() } 
        
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
        return self.annotation

    def json(self, of=None, **kwargs):
        tmp = { str(k): v for k,v in self.annotation['relations'].items() }
        conversion = self.annotation
        conversion['relations'] = tmp
        if of == None:
            print(json.dumps(conversion, **kwargs))
        else:
            json.dump(conversion, fp=of, **kwargs)
            
            '''
    def pandas(self):
        pd = {}
        for k, v in self.annotation['sentence'].items():
            if k == 'sentence':
                pd['sent'] = v
            elif k == 'tokenized' or k == 'tag' :
                pd[k + '_sent'] = v
        for k, v in self.annotation['entities'].items():
            tmp = 'type'
            for i, j in v.items():
                if i == 'type':
                    pd[j] = k
                    tmp = j
                elif i == 'tokenized' or i == 'tag':
                    pd[i + '_' + tmp] = j
        for k, v in self.annotation['relations'].items():
            tmp = 'type'
            for i, j in v.items():
                if i == 'type':
                    pd['relation_type'] = j
                    pd['head'] = k[0]
                    pd['tail'] = k[1]
                    tmp = j
                elif i == 'tokenized':
                    pd[i + '_head'] = j[0]
                    pd[i + '_tail'] = j[1]
        return pd
'''



    

######## MAIN ########


# import the pretrained tokenizer
model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model)

# input file to process
input_f = 'ADE-Corpus-V2/DRUG-AE.rel'

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
for s in sents.keys():
    ann.append(Annotation(sent=s, entities=sents[s]['entities'], relations=sents[s]['relations'], tokenizer=tokenizer).get_annotation())

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


