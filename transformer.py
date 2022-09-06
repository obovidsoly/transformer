from turtle import Turtle
import torch
import torch.nn as nn
import math

'''
2022.08.21 CCLab Study 과제

Transformer Implementation

Transformer architecture의 핵심은
1. multi-head attention
2. Masking

이고, 구현에서도 손이 제일 가는 부분입니다.

따라서 진행하셔야할 부분은,

1. Positional Encoding 구현
2. MultiHeadAttention 구현
(추가적으로, 가능하다면, 구현된 transformer를 이용해 Translation, Summarization 등 Task에 학습을 진행해보는 것입니다.)


'''

class PosEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

    def forward(self, seq_len):
        device="cuda" if torch.cuda.is_available() else "cpu"

        Pos=torch.arange(seq_len).reshape((seq_len,1)).to(device)
        I=torch.arange(self.config.embedding_dim).reshape((1,self.config.embedding_dim)).to(device)
        P=Pos/torch.pow(10000,I/self.config.embedding_dim)
        P[:,0::2]=torch.sin(P[:,0::2])
        P[:,1::2]=torch.cos(P[:,1::2])

        return P


class Transformer(nn.Module):
    def __init__(self, config,pad_idx,max_seq_len):
        super().__init__()
        self.config = config
        self.pad_idx=pad_idx
        self.max_seq_len=max_seq_len
        self.shared_word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=pad_idx)                
        self.encoder = TransformerEncoder(config, shared_word_embedding=self.shared_word_embedding)
        self.decoder = TransformerDecoder(config, shared_word_embedding=self.shared_word_embedding)
        self.linear=nn.Linear(config.hidden_dim,config.vocab_size)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids,train):
        batch_size=enc_input_ids.size(-2)
        enc_output = self.encoder(input_ids=enc_input_ids, attention_mask=enc_attention_mask)
        dec_output=self.decoder(dec_input_ids,enc_output,train)
        output=self.softmax(self.linear(dec_output))    # (batch_size,seq_len,vocab_size)
        output=torch.argmax(output,dim=-1) # (batch_size,seq_len)
        predict=output
        seq_len=1
        
        if not train:
            while (output[:,-1]==self.pad_idx).sum() != batch_size and seq_len<self.max_seq_len:
                output=torch.argmax(self.softmax(self.linear(self.decoder(output,enc_output,train))),dim=-1)
                predict=torch.cat((predict,output),dim=-2)
                seq_len+=1
        
        return predict


class TransformerConfig:
    def __init__(self):
        self.vocab_size = 50265
        self.embedding_dim=self.hidden_dim = 512
        self.head_num = 8
        self.head_dim = 64
        self.encoder_layer_num = 6
        self.decoder_layer_num = 6
        
class TransformerEncoder(nn.Module):
    def __init__(self, config, shared_word_embedding):
        super().__init__()
        self.config = config
                
        self.word_embedding = shared_word_embedding
        self.pos_embedding = PosEncoding(config)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(config) for i in range(config.encoder_layer_num)])

    def forward(self, input_ids, attention_mask):
        input_repre = self.word_embedding(input_ids)
        seq_len=input_repre.size(-2)
        input_repre += self.pos_embedding(seq_len)

        for layer in self.encoder_layers:
            input_repre = layer(input=input_repre, attention_mask=attention_mask)
            
        output = input_repre
        return output
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.multi_head_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.hidden_dim)
        
        self.linear_1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        
    def forward(self, input, attention_mask):
        mha_output = self.layernorm(input + self.multi_head_attention(input=input, attention_mask=attention_mask))
        layer_output = self.layernorm(mha_output + self.linear_2(self.relu(self.linear_1(mha_output))))
        
        return layer_output    
        
class TransformerDecoder(nn.Module):
    def __init__(self, config, shared_word_embedding):
        super().__init__()
        self.config = config                
                
        self.word_embedding = shared_word_embedding
        self.pos_embedding = PosEncoding(config)
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(config) for i in range(config.encoder_layer_num)])

    def create_self_attention_mask(self,seq_len):
        mask=torch.ones((seq_len,seq_len))
        for row in range(mask.size(0)):
            mask[row,row+1:]=0

        return mask

    def forward(self, input_ids, enc_output,train):
        input_repre = self.word_embedding(input_ids)
        seq_len=input_repre.size(-2)
        input_repre += self.pos_embedding(seq_len)
        self_attention_mask=None

        if train:
            self_attention_mask=self.create_self_attention_mask(seq_len).to("cuda" if torch.cuda.is_available() else "cpu")

        for layer in self.decoder_layers:
            input_repre = layer(input=input_repre, enc_output=enc_output, dec_attention_mask=self_attention_mask,train=train)
            
        output=input_repre
        return input_repre


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.masked_attention = MultiHeadAttention(config)
        self.enc_dec_cross_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.hidden_dim)
        
        self.linear_1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

    def forward(self, input, enc_output,dec_attention_mask,train):
        if train:
            masked_mha_output = self.layernorm(input + self.masked_attention(input=input,
                                                                                attention_mask=dec_attention_mask, 
                                                                                encoder_output=None))
        else:
            masked_mha_output = self.layernorm(input + self.masked_attention(input=input,
                                                                                attention_mask=None, 
                                                                                encoder_output=None))
        
        cross_mha_output = self.layernorm(masked_mha_output + self.enc_dec_cross_attention(input=masked_mha_output,
                                                                                            attention_mask=None,
                                                                                            encoder_output=enc_output))
        layer_output = self.layernorm(cross_mha_output + self.linear_2(self.relu(self.linear_1(cross_mha_output))))
        
        return layer_output


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_Q=nn.Linear(config.embedding_dim,config.hidden_dim)
        self.linear_K=nn.Linear(config.embedding_dim,config.hidden_dim)
        self.linear_V=nn.Linear(config.embedding_dim,config.hidden_dim)
        self.softmax=nn.Softmax(dim=-1)
        
    def forward(self, input, attention_mask=None, encoder_output=None):
        head_dim=self.config.head_dim
        head_num=self.config.head_num
        hidden_dim=self.config.hidden_dim
        Q=K=V=None

        # train==True, attention_mask==None, encoder_output==valid -> encoder-decoder attention
        # train==True, attention_mask==valid, encoder_output==None -> masked self attention
        # train==False, attention_mask==None, encoder_output==None -> decoder self attention
        # train==False, attention_mask==None, encoder_output==valid -> encoder-decoder attention
        # train==False, attention_mask==valid, encoder_output==None -> encoder masked self attention

        #encoder_output == None -> self attention
        if encoder_output is None:
            Q=self.linear_Q(input)
            K=self.linear_K(input)
            V=self.linear_V(input)
        else:
            Q=input
            K=V=encoder_output
        
        batch_size=Q.size(0)
        seq_len_Q,seq_len=Q.size(1),V.size(1)
        Q=Q.reshape(batch_size,seq_len_Q,head_num,head_dim).transpose(-3,-2)
        K=K.reshape(batch_size,seq_len,head_num,head_dim).transpose(-3,-2)
        V=V.reshape(batch_size,seq_len,head_num,head_dim).transpose(-3,-2) #(batch_size, head_num, seq_len, hidden_size/head_num)

        attention_score=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(head_dim)
        if attention_mask is not None:
            attention_score=attention_score.masked_fill(attention_mask==0,-1e15)
        attention_score=self.softmax(attention_score)

        attention_value=torch.matmul(attention_score,V)
        attention_value=attention_value.transpose(-3,-2)    # (batch_size, seq_len, head_num, hidden_size/head_num)
        attention_value=attention_value.reshape(batch_size,seq_len_Q,hidden_dim)

        return input+attention_value

device="cuda" if torch.cuda.is_available() else "cpu"

model_config = TransformerConfig()
model = Transformer(config=model_config,pad_idx=0,max_seq_len=40).to(device)

batch_size=1
seq_len=128

enc_input_ids_rand = torch.randint(0, 10, (batch_size,seq_len)).to(device)
enc_attention_mask = torch.ones((seq_len,seq_len)).to(device)
dec_input_ids_rand = torch.randint(0, 10, (batch_size,seq_len)).to(device)

output = model(enc_input_ids=enc_input_ids_rand, 
               enc_attention_mask=enc_attention_mask,
               dec_input_ids=dec_input_ids_rand,train=False)

print(output.shape)