#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        
        super(CharDecoder, self).__init__()             # Initialize as an nn.Module
        
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features=len(target_vocab.char2id), bias=True)
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id), embedding_dim=char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        
        # computed the *sum* of cross-entropy losses of all the words in the batch
        # not take into account pad character when compute loss
        self.loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char2id['<pad>'])

        ### END YOUR CODE


    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        
        char_emb = self.decoderCharEmb(input)
        hidden, dec_hidden = self.charDecoder(char_emb, dec_hidden)
        scores = self.char_output_projection(hidden) 
        
        return scores, dec_hidden
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same. The "length" is the length of char sequence.  
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        x = char_sequence[:-1]                              # the <END> token should not be in the input; shape (length-1, batch)
        scores, dec_hidden = self.forward(x, dec_hidden)    # call Forward pass of character decoder; scores shape (length-1, batch, vocab_size)
        
        targets = char_sequence[1:]                         # the <START> token should not be in the output; shape (length-1, batch)

        targets = targets.reshape(-1)                       # squeeze into 1D (length-1 * batch_size)
        scores = scores.reshape(-1, scores.shape[-1])       # (length-1 * batch_size, vocab_size)

        ce_loss = self.loss(scores, targets)
        
        return ce_loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        # initial constant
        batch_size = initialStates[0].shape[1]
        start_index = self.target_vocab.start_of_word
        end_index = self.target_vocab.end_of_word
        
        # initial state
        dec_hidden = initialStates

        # initialize two variables
        current_chars = torch.tensor([start_index] * batch_size, device=device).unsqueeze(0)    # char for each entry in a batch (1, batch_size) <- unsqueeze for the LSTM dim
        decodeTuple = [['', False] for _ in range(batch_size)]                                  # output words for each entry (output string, if this entry has already reached the end)

        for t in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)
            prob = torch.softmax(scores, dim=2)
            current_chars = torch.argmax(scores, dim=2)         # pick the char with the highest probability as the next char

            char_indices = current_chars.detach().squeeze(0)    # returns a new Tensor, detached from the current graph
            for i, char_index in enumerate(char_indices): 
                if not decodeTuple[i][1]:                       # this entry in the batch has not reached the end
                    if char_index == end_index:
                        decodeTuple[i][1] = True                # flip the end flag
                    else:                        
                        decodeTuple[i][0] += self.target_vocab.id2char[char_index.item()]   # append the predicted char 
        
        decodedWords = [item[0] for item in decodeTuple]
        
        return decodedWords
        
        ### END YOUR CODE

