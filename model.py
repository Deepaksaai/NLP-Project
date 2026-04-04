import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    BiLSTM Encoder.
    Reads input sequence in both directions and produces
    hidden states for each token.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()

        # Embedding layer trained from scratch
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # index 0 is padding token
        )

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Project BiLSTM output (hidden_dim * 2) back to hidden_dim
        # This is needed because BiLSTM doubles the hidden size
        # and the decoder expects hidden_dim not hidden_dim * 2
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: (batch_size, src_len) — token indices
        Returns:
            outputs: (batch_size, src_len, hidden_dim * 2) — all hidden states
            hidden: (num_layers, batch_size, hidden_dim) — final hidden state
            cell: (num_layers, batch_size, hidden_dim) — final cell state
        """

        # Embed input tokens
        embedded = self.dropout(self.embedding(src))
        # embedded: (batch_size, src_len, embed_dim)

        # Pass through BiLSTM
        outputs, (hidden, cell) = self.bilstm(embedded)
        # outputs: (batch_size, src_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell: (num_layers * 2, batch_size, hidden_dim)

        # Combine forward and backward hidden states
        # hidden has shape (num_layers * 2, batch, hidden_dim)
        # We need to merge forward and backward for each layer
        # to get (num_layers, batch, hidden_dim)

        num_layers = hidden.shape[0] // 2
        hidden_combined = []
        cell_combined = []

        for i in range(num_layers):
            # Forward hidden state for layer i
            h_fwd = hidden[2 * i]
            # Backward hidden state for layer i
            h_bwd = hidden[2 * i + 1]
            # Concatenate and project
            h_combined = torch.tanh(
                self.fc_hidden(torch.cat([h_fwd, h_bwd], dim=1))
            )
            hidden_combined.append(h_combined)

            c_fwd = cell[2 * i]
            c_bwd = cell[2 * i + 1]
            c_combined = torch.tanh(
                self.fc_cell(torch.cat([c_fwd, c_bwd], dim=1))
            )
            cell_combined.append(c_combined)

        # Stack back to (num_layers, batch, hidden_dim)
        hidden = torch.stack(hidden_combined, dim=0)
        cell = torch.stack(cell_combined, dim=0)

        return outputs, hidden, cell


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention.
    At each decoder step calculates how much to focus
    on each encoder hidden state.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        # Attention layers
        # encoder_outputs are hidden_dim * 2 (bidirectional)
        # decoder hidden is hidden_dim
        self.attn = nn.Linear(
            (hidden_dim * 2) + hidden_dim,
            hidden_dim
        )
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch_size, hidden_dim) — current decoder state
        encoder_outputs: (batch_size, src_len, hidden_dim * 2)
        Returns:
            attention_weights: (batch_size, src_len)
        """

        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        # so we can compare it with each encoder output
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # decoder_hidden: (batch_size, src_len, hidden_dim)

        # Concatenate decoder hidden with each encoder output
        combined = torch.cat([decoder_hidden, encoder_outputs], dim=2)
        # combined: (batch_size, src_len, hidden_dim * 2 + hidden_dim)

        # Calculate energy scores
        energy = torch.tanh(self.attn(combined))
        # energy: (batch_size, src_len, hidden_dim)

        # Reduce to scalar score per position
        attention_scores = self.v(energy).squeeze(2)
        # attention_scores: (batch_size, src_len)

        # Convert to probabilities
        attention_weights = F.softmax(attention_scores, dim=1)

        return attention_weights


class Decoder(nn.Module):
    """
    LSTM Decoder with Attention.
    Generates output tokens one at a time using
    attention-weighted context from the encoder.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()

        self.attention = Attention(hidden_dim)

        # Embedding for target tokens
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # LSTM takes embedded token + context vector as input
        # context vector is hidden_dim * 2 (from bidirectional encoder)
        self.lstm = nn.LSTM(
            input_size=(hidden_dim * 2) + embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Final output projection to vocabulary
        # Input is decoder output + context vector + embedded token
        self.fc_out = nn.Linear(
            hidden_dim + (hidden_dim * 2) + embed_dim,
            vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_token, encoder_outputs, hidden, cell):
        """
        trg_token: (batch_size,) — current target token
        encoder_outputs: (batch_size, src_len, hidden_dim * 2)
        hidden: (num_layers, batch_size, hidden_dim)
        cell: (num_layers, batch_size, hidden_dim)
        Returns:
            prediction: (batch_size, vocab_size)
            hidden: updated hidden state
            cell: updated cell state
            attention_weights: (batch_size, src_len)
        """

        # Add sequence dimension to token
        trg_token = trg_token.unsqueeze(1)
        # trg_token: (batch_size, 1)

        # Embed the token
        embedded = self.dropout(self.embedding(trg_token))
        # embedded: (batch_size, 1, embed_dim)

        # Calculate attention using top layer of decoder hidden state
        attention_weights = self.attention(hidden[-1], encoder_outputs)
        # attention_weights: (batch_size, src_len)

        # Apply attention weights to encoder outputs
        attention_weights = attention_weights.unsqueeze(1)
        # attention_weights: (batch_size, 1, src_len)

        context = torch.bmm(attention_weights, encoder_outputs)
        # context: (batch_size, 1, hidden_dim * 2)

        # Concatenate embedded token with context vector
        lstm_input = torch.cat([embedded, context], dim=2)
        # lstm_input: (batch_size, 1, embed_dim + hidden_dim * 2)

        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)

        # Prepare final prediction input
        # Concatenate output, context, and embedded token
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(
            torch.cat([output, context, embedded], dim=1)
        )
        # prediction: (batch_size, vocab_size)

        attention_weights = attention_weights.squeeze(1)

        return prediction, hidden, cell, attention_weights


class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model combining Encoder, Attention, and Decoder.
    This is the complete BiLSTM + Attention summarization model.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch_size, src_len) — input chunk tokens
        trg: (batch_size, trg_len) — target summary tokens
        teacher_forcing_ratio: probability of using real target
                               token vs model prediction at each step
        Returns:
            outputs: (trg_len, batch_size, vocab_size) — predictions
        """

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        # Store decoder outputs
        outputs = torch.zeros(
            trg_len, batch_size, vocab_size
        ).to(self.device)

        # Encode the source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # First decoder input is the <SOS> token
        decoder_input = trg[:, 0]

        # Generate output tokens one at a time
        for t in range(1, trg_len):

            # Decode one step
            output, hidden, cell, _ = self.decoder(
                decoder_input, encoder_outputs, hidden, cell
            )

            # Store prediction
            outputs[t] = output

            # Teacher forcing — use real token or model prediction
            use_teacher_forcing = (
                torch.rand(1).item() < teacher_forcing_ratio
            )

            if use_teacher_forcing:
                # Use real target token as next input
                decoder_input = trg[:, t]
            else:
                # Use model's own prediction as next input
                decoder_input = output.argmax(1)

        return outputs


def build_model(vocab_size, device, embed_dim=256, hidden_dim=512,
                num_layers=2, dropout=0.3):
    """
    Builds and returns the complete Seq2Seq model.
    All hyperparameters are set here for easy adjustment.
    """

    encoder = Encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    decoder = Decoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    model = Seq2Seq(encoder, decoder, device)

    # Initialize weights
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    model.apply(init_weights)

    # Print model size
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Model built successfully")
    print(f"Total trainable parameters: {total_params:,}")

    return model


# Quick test
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test with dummy data
    vocab_size = 30000
    batch_size = 4
    src_len = 200
    trg_len = 50

    model = build_model(vocab_size, device)
    model = model.to(device)

    # Create dummy input
    src = torch.randint(1, vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(1, vocab_size, (batch_size, trg_len)).to(device)

    # Forward pass
    output = model(src, trg)
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({trg_len}, {batch_size}, {vocab_size})")
    print("\nModel test passed successfully")
