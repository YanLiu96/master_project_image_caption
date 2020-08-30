# evaluate the performance of model on test dataser: get the BLEU-1 to BLEU-4 scores
# METEOR ROUGE CIDEr 
# Time: 20 min 
# resnet101 beam search 3:{'Bleu_1': 0.7077848412851242, 'Bleu_2': 0.5352381686387186, 'Bleu_3': 0.4011979950852196, 'Bleu_4': 0.30209710516184746, 'METEOR': 0.24550618207549812, 'ROUGE_L': 0.5225619634651749, 'CIDEr': 0.9404193445802925}
# resnet152 beam search 1:{'Bleu_1': 0.7067479161146974, 'Bleu_2': 0.532390949850118, 'Bleu_3': 0.3880338877519845, 'Bleu_4': 0.2819539230760915, 'METEOR': 0.2451758871431663, 'ROUGE_L': 0.5203647122740788, 'CIDEr': 0.9378385373606865}
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms

# import the dataset pipline
from evaluate.pycocoevalcap.bleu.bleu import Bleu
from evaluate.pycocoevalcap.rouge.rouge import Rouge
from evaluate.pycocoevalcap.cider.cider import Cider
from evaluate.pycocoevalcap.meteor.meteor import Meteor
#from evaluate.pycocoevalcap.spice.spice import Spice

from train.dataset import *
#from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

hyp={}
ref={}
data_folder = '/saved_data'
data_name = 'trained_models'

# Load word map (word2ix)
with open('saved_data/word2index/word2idx.json', 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")#,
        #(Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def evaluate(pre_save, beam_size):
    if pre_save=='pre_save':
        with open('saved_data/ft_152/ft_resnet152_bs1_test_hyp.json', 'r') as j:
            hyp = json.load(j)
        with open('saved_data/ft_152/ft_resnet152_bs1_test_ref.json', 'r') as j:
            ref = json.load(j)
        print('Begin evaluation:')
        print(calc_scores(ref,hyp))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        checkpoint = 'saved_data/trained_models/resnet152_best_checkpoint_trained_models.pth.tar'
        # Load model
        checkpoint = torch.load(checkpoint,map_location=str(device))
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
        # DataLoader
        loader = torch.utils.data.DataLoader(
            MyDataset('saved_data/', 'test', transform=transforms.Compose([normalize])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        references = list() # store anotation (true captions) of the image 
        hypotheses = list() # store prediction for that image

        # For each image
        for i, (image, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc="Evaluation on the test dataset")):
            img_id =i
            k = int(beam_size)
            image = image.to(device)  # (1, 3, 256, 256)

            # Encode
            encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            # print(img_captions) 5 captions for every image
            ref[img_id]= list(
                map(lambda c: ' '.join([rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]),
                    img_caps)) 
            references.append(img_captions)
            # Hypotheses 
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            #print(hypotheses)
            hyp[img_id]= [' '.join([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])]
            assert len(references) == len(hypotheses)

        # Calculate BLEUscores
        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        with open(os.path.join('saved_data/evaluation','test_ref.json'),'w') as f1:
            json.dump(ref, f1)
        with open(os.path.join('saved_data/evaluation','test_hyp.json'),'w') as f1:
            json.dump(hyp, f1)
        print(calc_scores(ref,hyp))
        #bleu4 = corpus_bleu(references, hypotheses)
        #print("\nBLEU-4 score is %.4f." %bleu4)
