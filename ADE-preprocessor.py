import torch, re, json, pandas
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
            return (i, i+len(pattern))





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
        for k, v in entities.items():
            tokenized_ent = self.tokenizer(k, add_special_tokens=False)['input_ids']
            span = find_sublist(self.sent['tokenized'], tokenized_ent)
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

    def json(self, pretty=True):
        tmp = { str(k): v for k,v in self.annotation['relations'].items() }
        conversion = self.annotation
        conversion['relations'] = tmp
        return json.dumps(conversion, indent=4) if pretty else json.dumps(conversion)

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




    

######## MAIN ########


# import the pretrained tokenizer
model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model)

# input file to process
input_f = 'ADE-Corpus-V2/DRUG-AE.rel'

#out_f = 'test.test'

df = []
# extracting sentences and original entities annotations
with open(out_f, 'w') as o_f:
    with open(input_f, 'r') as in_f:
        for x in in_f:
            #print(x)                       # WARNING: some sentences occur multiple times!!! should we remove the duplicates?
            split = re.split('\|', x)
            sent = split[1]
            ents = { split[2]: 'AE', split[5]: 'DRUG' }
            rel = { (split[2], split[5]): 'HAS_ADVERSE_EFFECT' }
            #ann = Annotation(sent=sent, entities=ents, relations=rel, tokenizer=tokenizer).json(pretty=True)
            #json.dump(ann, o_f)
            df.append(Annotation(sent=sent, entities=ents, relations=rel, tokenizer=tokenizer).pandas())

#print(ann)
#print(json.load(ann))

df = pandas.DataFrame(df)
df.to_pickle('DRUG-AE.pkl')
