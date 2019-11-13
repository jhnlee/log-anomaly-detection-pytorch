import torch
import torch.nn as nn


class RnnEncoder(nn.Module):
    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim):
        super(RnnEncoder, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                            embedding_dim=embedding_hidden_dim,
                                            padding_idx=vocab.index('<PAD>'))
        self.gru_layer = nn.GRU(input_size=embedding_hidden_dim,
                                hidden_size=gru_hidden_dim,
                                batch_first=True,
                                bidirectional=True)

    def forward(self, input_sequence):

        # input_sequence = (B x L)
        # input_length : (B)
        # embedding_hidden : (h)

        #batch_size, max_len = input_sequence.shape

        embedded = self.embedding_layer(input_sequence)  # (B x L x h)
        encoder_outputs, encoder_hidden = self.gru_layer(embedded)  # (B x L x 2H), (2*N x B x H)

        return encoder_hidden


