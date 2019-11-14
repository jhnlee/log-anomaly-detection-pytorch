import torch
import torch.nn as nn


class BiGruEncoder(nn.Module):
    """ Bi-directional GRU Encoder for AutoEncoder """
    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim):
        super(BiGruEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                            embedding_dim=embedding_hidden_dim,
                                            padding_idx=vocab.index('<PAD>'))
        self.gru_layer = nn.GRU(input_size=embedding_hidden_dim,
                                hidden_size=gru_hidden_dim,
                                batch_first=True,
                                bidirectional=True)
        self.fc = nn.Linear(in_features=gru_hidden_dim * 2,
                            out_features=gru_hidden_dim)

    def forward(self,
                input_sequence):  # input_sequence = (B x L)

        # input_length : (B)
        # embedding_hidden : (h)

        batch_size, max_len = input_sequence.shape

        embedded = self.embedding_layer(input_sequence)  # (B x L x h)
        encoder_outputs, encoder_hidden = self.gru_layer(embedded)  # (B x L x 2H), (2*N x B x H)

        # Sum forward and backward hidden state   (B x L x 2H)  =>  (B x L x H)
        encoder_outputs = encoder_outputs[:, :, :self.gru_hidden_dim] + encoder_outputs[:, :, self.gru_hidden_dim:]

        # Concat forward and backward hidden states
        encoder_hidden = encoder_hidden.view(self.num_hidden_layer, 2, batch_size, self.gru_hidden_dim)[-1]
        encoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), 1)  # (2 x B x H)  =>  (B x 2H)
        encoder_hidden = self.fc(encoder_hidden).unsqueeze(0)  # (B x 2H)  =>  (1 x B x H)

        return encoder_outputs, encoder_hidden  # (B x L x H), (1 x B x H)


class GruDecoder(nn.Module):
    """ GRU Decoder for AutoEncoder """
    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim, attention_method='dot'):
        super(GruDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                            embedding_dim=embedding_hidden_dim,
                                            padding_idx=vocab.index('<PAD>'))
        self.gru_layer = nn.GRU(input_size=embedding_hidden_dim,
                                hidden_size=gru_hidden_dim)
        self.attention_layer = Attention(method=attention_method,
                                         hidden_size=gru_hidden_dim * 2)
        self.concat_layer = nn.Linear(in_features=gru_hidden_dim * 2,
                                      out_features=gru_hidden_dim)
        self.fc_layer = nn.Linear(in_features=gru_hidden_dim,
                                  out_features=len(vocab))

    def forward(self,
                target_embedded,  # target_embedded : (1 x B x h)
                encoder_outputs,  # encoder_outputs : (B x L x H)
                last_hidden,      # last_hidden : (1 x B x H)
                encoder_mask):    # encoder_mask : (B x L)

        decoder_output, last_hidden = self.gru_layer(target_embedded, last_hidden)  # (1 x B x H)

        # Calculate attention weights
        attention_weight = self.attention_layer(decoder_output, encoder_outputs, encoder_mask)  # (B x 1 x L)
        context = attention_weight.bmm(encoder_outputs)  # (B x 1 x L) * (B x L x H) = (B x 1 x H)

        # Concatenate attention output and decoder output
        concat_input = torch.cat((decoder_output.squeeze(0), context.squeeze(1)), 1)  # (B x 2H)
        concat_output = torch.tanh(self.concat_layer(concat_input))  # (B x H)

        # Predict next word
        out = self.fc_layer(concat_output).unsqueeze(0)  # (1 x B x V)

        return out, last_hidden


class Attention(nn.Module):
    """ Implementation of various attention score function """
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif self.method == 'concat':  # Additive
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

        elif self.method == 'dot':
            pass

    def forward(self, hidden, encoder_outputs, encoder_mask):

        attn_energies = self.score(hidden, encoder_outputs, encoder_mask)
        attn_weight = torch.softmax(attn_energies, 2)

        return attn_weight  # (B x 1 x L)

    def score(self, hidden, encoder_outputs, encoder_mask):
        # encoder outputs : (B x L x H)
        hidden = hidden.transpose(0, 1)  # (B x 1 x H)
        if self.method == 'dot':
            # TODO : scaled dot product attention
            energy = torch.bmm(hidden, encoder_outputs.transpose(1, 2))  # (B x 1 x L)

        elif self.method == 'general':
            energy = torch.bmm(hidden, self.attn(encoder_outputs).transpose(1, 2))  # (B x 1 x L)

        elif self.method == 'concat':
            seq_length = encoder_outputs.shape[1]
            concat = torch.cat((hidden.repeat(1, seq_length, 1), encoder_outputs), 2)
            energy = self.v(torch.tanh(self.attn(concat))).transpose(1, 2)  # (B x 1 x L)

        else:
            raise ValueError("Invalid attention method")

        # Mask to pad token
        energy = energy.masked_fill(encoder_mask.unsqueeze(1) == 0, -1e10)

        return energy
