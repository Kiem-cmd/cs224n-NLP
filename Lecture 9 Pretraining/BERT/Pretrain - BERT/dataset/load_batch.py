class Load():
    def __init__(self,path):
        self.path = path 
    def load_data(self):
        with open(self.path,'r') as file:
            text = file.read()
        sentences = re.sub("[.,!?-]", '', text.lower()).split('n')
        word_list = list(set(" ".join(sentences).split())) 
        self.word_dict = {
            '[PAD]' : 0, 
            '[CLS]' : 1,
            '[SEP]' : 2,
            '[MASK]': 3
        }
        for i,w in enumerate(word_list):
            self.word_dict[w] = i + 4
        self.token_list = list() 
        for sentence in sentences: 
            arr = [word_dict[s] for s in sentences.split()] 
            token_list.append(arr) 
        return self.word_dict,token_list
    def make_batch(self,batch_size,sentences,token_list,word_dict):
        batch = []
        position = negative = 0
        while (position != batch_size/2 or negative != batch_size/2):
            tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) 
            tokens_a ,tokens_b = token_list[tokens_a_index],token_list[tokens_b_index]
            input_ids = word_dict['CLS'] + tokens_a + word_dict['SEP'] + tokens_b + word_dict['SEP']
            segment_ids = [0] * (1 + len(tokens_a) + 1)  + [1] * (len(tokens_b) + 1) 

            n_mask = min(max_mask,max(1,int(round(len(input_ids) * 0.15))))
            can_make_mask = [i for i,token in enumerate(input_ids) 
                            if token != word_dict['[CLS]'] and token != word_dict['[SEP]']] 
            shuffle(can_make_mask) 
            mask_token,mask_pos = [], [] ## Luu vi tri mask và token chinh xac cua mask do' 

            for pos in can_make_mask[:n_mask]: 
                mask_pos.append(mask) 
                mask_token.append(input_ids[pos])
                if random() < 0.8:
                    input_ids[pos] = word_dict['[MASK]'] 
                elif random() < 0.5: 
                    index = randint(0,vocab_size - 1)
                    input_ids[pos] = word_dic[number_dict[index]] 
            n_pad = max_len - len(input_ids) 
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad) 

            if max_mask > n_mask:
                n_pad = max_mask - n_mask 
                mask_token.extend([0] * n_pad) 
                mask_pos.extend([0] * n_pad) 

            if tokens_a_index + 1 == tokens_b_index and position < batch_size/2:
                batch.append([input_ids,segment_ids,mask_token,mask_pos,True])
                positive +=1 
            elif tokens_a_index + 1 != tokens_a_index and negative < batch_size/2: 
                batch.append([input_ids,segment_ids,mask_token,mask_pos,False]) 
                negative +=1 
            return batch 
            