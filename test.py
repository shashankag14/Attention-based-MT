import torch
import random
import os

import data
import utils
import model_attn

DEBUG = 0

test_data_in = data.corpus.test_in.to(utils.device)
test_data_out = data.corpus.test_out.to(utils.device)

if DEBUG :
  print("test_data_in", test_data_in.shape)
  print("test_data_out", test_data_out.shape)

encoder = model_attn.EncoderRNN(data.corpus.dictionary_in.n_word, utils.args.hsize).to(utils.device)
encoder.load_state_dict(torch.load(os.path.join(utils.saved_model, "encoder_weights.pt")))
encoder.eval()

attn_decoder = model_attn.AttnDecoderRNN(utils.args.hsize, data.corpus.dictionary_out.n_word, dropout_p=0.1).to(utils.device)
attn_decoder.load_state_dict(torch.load(os.path.join(utils.saved_model, "decoder_weights.pt")))
attn_decoder.eval()

def test(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = torch.tensor(sentence, dtype=torch.long, device=utils.device).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(utils.args.sent_maxlen, encoder.hidden_size, device=utils.device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[data.SOS_token]], device=utils.device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(utils.args.sent_maxlen, utils.args.sent_maxlen)

        for di in range(utils.args.sent_maxlen):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == data.EOS_token:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append(data.corpus.dictionary_out.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def testRandomly(encoder, decoder, n=10):
    print("Index of . : ", data.corpus.dictionary_out.word2idx['.'])
    rand_nums = random.sample(range(test_data_in.size(0)), n)
    test_file = open(os.path.join(utils.output_path, "test_output.txt"), "w") 
    test_file.write("Some randomly sampled machine translated outputs on test data :\n\n") 
    for i in range(n):
        sample = rand_nums.pop()
        input_data = test_data_in[sample].tolist()
        target_data = test_data_out[sample].tolist()
        output_words, attentions = test(encoder, decoder, input_data)

        input_words = []
        target_words = []
        for idx in range(utils.args.sent_maxlen):
          input_words.append(data.corpus.dictionary_in.idx2word[input_data.pop(0)])
          target_words.append(data.corpus.dictionary_out.idx2word[target_data.pop(0)])
        
        input_sent = ' '.join(input_words)
        target_sent = ' '.join(target_words)
        
        test_file.write('>' + input_sent+ '\n')
        test_file.write('=' + target_sent+ '\n')
        print('>', input_sent)
        print('=', target_sent)
        
        output_sentence = ' '.join(output_words)
        test_file.write('<' + output_sentence + '\n\n')
        print('<', output_sentence)
        print('')
    test_file.close()
    print("Samples saved in the local path !")

testRandomly(encoder, attn_decoder)