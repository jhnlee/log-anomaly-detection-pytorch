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

        return encoder_outputs, encoder_hidden
    
    
class RnnDecoder(nn.Module):    
    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim, attention_method='dot'):
        super(Decoder, self).__init__()
        self.gru_layer = nn.GRU(embedding_hidden_dim, gru_hidden_dim)
        self.attention_layer = Attention(attention_method, gru_hidden_dim * 2)
        self.concat_layer = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)
        self.fc_layer = nn.Linear(gru_hidden_dim, len(vocab))

    def forward(self, target_embedded, encoder_outputs, last_hidden, encoder_mask):

        # target_embedded : (1 x B x h)
        # encoder_outputs : (B x L x H)
        # last_hidden : (1 x B x H)

        decoder_output, last_hidden = self.gru_layer(target_embedded, last_hidden)  # (1 x B x H)

        attention_weight = self.attention_layer(decoder_output, encoder_outputs, encoder_mask)  # (B x 1 x L)
        context = attention_weight.bmm(encoder_outputs)  # (B x 1 x L) * (B x L x H) = (B x 1 x H)

        concat_input = torch.cat((decoder_output.squeeze(0), context.squeeze(1)), 1)  # (B x 2H)
        concat_output = torch.tanh(self.concat_layer(concat_input))  # (B x H)

        out = self.fc_layer(concat_output).unsqueeze(0)  # (1 x B x V)

        return out, last_hidden


