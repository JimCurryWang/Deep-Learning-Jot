import torch
from torch import nn

'''
torch.nn.Embedding(
    num_embeddings, ---> size of the dictionary of embeddings == vocab_size
    embedding_dim, --->  the size of each embedding vector

    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    _weight=None
)



REF:: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
'''


def Examples():
    # an Embedding module containing 10 tensors of size 3
    embedding = nn.Embedding(10,3)
    print(embedding.weight)

    # a batch of 2 samples of 4 indices each
    inp = torch.LongTensor([
            [1,2,4,5],
            [4,3,2,9]
        ])

    emb = embedding(inp)

    print(emb)

    '''
    tensor([[[-0.3706, -0.0240, -0.6057], --> 1
             [ 0.8399,  0.5721,  0.1688], --> 2
             [ 0.2958,  1.0252,  0.7635], --> 4
             [-0.1053, -0.2161,  1.2853]],

            [[ 0.2958,  1.0252,  0.7635], --> 4
             [ 0.6627,  0.6380,  0.7199], --> 3
             [ 0.8399,  0.5721,  0.1688], --> 2 
             [-0.3977,  0.3657, -0.7857]]], grad_fn=<EmbeddingBackward>)
    '''



def Examples_from_pretrained():
    '''
    tensor([ 
            [1.0000, 2.3000, 3.0000], ---> 0
            [4.0000, 5.1000, 6.3000]  ---> 1 
        ])
    '''
    # FloatTensor containing pretrained weights
    weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    embedding = nn.Embedding.from_pretrained(weight)
    embed.weight.requires_grad = False
    
    print(embedding.weight)

    # Get embeddings for index 1
    input = torch.LongTensor([1])
    out = embedding(input) # tensor([[ 4.0000,  5.1000,  6.3000]])
    print(out)


if __name__ == '__main__':
    # Examples()
    Examples_from_pretrained()




