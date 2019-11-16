import torch
import torch.nn as nn


class GruAutoEncoder(nn.Module):
    """
    GRU AutoEncoder

    Encoder : Bi-Directional GRU
    Decoder : GRU
    """
    def __init__(self, vocab, embedding_hidden_dim, num_hidden_layer, gru_hidden_dim, device,
                 dropout_p=0.1, attention_method='dot'):
        super(GruAutoEncoder, self).__init__()
        self.device = device
        self.vocab_size = len(vocab)
        self.encoder = BiGruEncoder(vocab, embedding_hidden_dim, gru_hidden_dim, num_hidden_layer, dropout_p)
        self.decoder = GruDecoder(vocab, embedding_hidden_dim, gru_hidden_dim, attention_method)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.index('<PAD>'))

    def forward(self,
                encoder_mask,  # (B x L)
                encoder_input,  # (B x L)
                decoder_input, ):  # (B x L)

        batch_size, max_len = encoder_input.shape
        decoder_input = decoder_input.transpose(0, 1)  # (B x L) => (L x B)

        encoder_outputs, hidden = self.encoder(encoder_input)  # (B x L x H), (2*N x B x H)
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).to(self.device)  # (L x B x V)

        for t in range(max_len):
            out, hidden = self.decoder(decoder_input=decoder_input[t].unsqueeze(0),
                                       encoder_outputs=encoder_outputs,
                                       last_hidden=hidden,
                                       encoder_mask=encoder_mask, )
            outputs[t] = out.squeeze(0)

        loss = self.loss_fn(outputs.view(-1, self.vocab_size),  # (L x B x V) => (L*B x V)
                            encoder_input.reshape(-1))  # (B x L) => (L*B)

        return outputs, loss


class BiGruEncoder(nn.Module):
    """ Bi-directional GRU Encoder for AutoEncoder """

    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim, num_hidden_layer, dropout_p):
        super(BiGruEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                            embedding_dim=embedding_hidden_dim,
                                            padding_idx=vocab.index('<PAD>'))
        self.gru_layer = nn.GRU(input_size=embedding_hidden_dim,
                                hidden_size=gru_hidden_dim,
                                num_layers=num_hidden_layer,
                                dropout=dropout_p if num_hidden_layer > 1 else 0,
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

    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim, attention_method):
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
                decoder_input,  # (1 x B)
                encoder_outputs,  # (B x L x H)
                last_hidden,  # (1 x B x H)
                encoder_mask):  # (B x L)

        embedded = self.embedding_layer(decoder_input)  # (1 x B x H)
        decoder_output, last_hidden = self.gru_layer(embedded, last_hidden)  # (1 x B x H), (2*N x B x H)

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
    """
    Implementation of various attention score function
    참고: https://wikidocs.net/22893
    """

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif self.method == 'concat':  # Luong(additive) Attention
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

        elif self.method == 'dot':
            pass

    def forward(self,
                decoder_outputs,  # (1 x B x H)
                encoder_outputs,  # (B x L x H)
                encoder_mask):  # (B x L)

        attn_energies = self.score(decoder_outputs, encoder_outputs, encoder_mask)
        attn_weight = torch.softmax(attn_energies, 2)

        return attn_weight  # (B x 1 x L)

    def score(self, decoder_outputs, encoder_outputs, encoder_mask):
        """ Attention score functions """
        decoder_outputs = decoder_outputs.transpose(0, 1)
        if self.method == 'dot':
            # (B x 1 x H) x (B x H x L) = (B x 1 x L)
            energy = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))
        elif self.method == 'general':
            # (B x 1 x H) x (B x H x L) = (B x 1 x L)
            energy = torch.bmm(decoder_outputs, self.attn(encoder_outputs).transpose(1, 2))
        elif self.method == 'concat':
            seq_length = encoder_outputs.shape[1]
            # (B x L x H) ; (B x L x H) => (B x L x 2H)
            concat = torch.cat((decoder_outputs.repeat(1, seq_length, 1), encoder_outputs), 2)
            # (B x L x 2H) =attn=> (B x L x H) =v=> (B x L x 1)
            energy = self.v(torch.tanh(self.attn(concat))).transpose(1, 2)  # (B x 1 x L)
        else:
            raise ValueError("Invalid attention method")

        # Mask to pad token
        energy = energy.masked_fill(encoder_mask.unsqueeze(1) == 0, -1e10)

        return energy
