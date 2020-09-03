# evaluate the performance of model on test dataser: get the BLEU-1 to BLEU-4 scores
# METEOR ROUGE CIDEr 
# Time: 20 min 
# model with resnet101 beam search 3:{'Bleu_1': 0.7077848412851242, 'Bleu_2': 0.5352381686387186, 'Bleu_3': 0.4011979950852196, 'Bleu_4': 0.30209710516184746, 'METEOR': 0.24550618207549812, 'ROUGE_L': 0.5225619634651749, 'CIDEr': 0.9404193445802925}
# model with resnet152 beam search 1:{'Bleu_1': 0.7067479161146974, 'Bleu_2': 0.532390949850118, 'Bleu_3': 0.3880338877519845, 'Bleu_4': 0.2819539230760915, 'METEOR': 0.2451758871431663, 'ROUGE_L': 0.5203647122740788, 'CIDEr': 0.9378385373606865}
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
from evaluate.pycocoevalcap.spice.spice import Spice

from train.dataset import *
from tqdm import tqdm

hyp={}
ref={}
data_folder = '/saved_data'

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
        (Spice(), "SPICE")
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
    if pre_save=='pre_save': # if you dont want to waste time make sure you donwload below file
        with open('saved_data/evaluation/resnet_152_bs3_test_hyp.json', 'r') as j:
            hyp = json.load(j)
        with open('saved_data/evaluation/resnet_152_bs3_test_ref.json', 'r') as j:
            ref = json.load(j)
        print('Begin evaluation:')
        print(calc_scores(ref,hyp))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        # make sure the path to your model is right!!
        checkpoint = 'saved_data/trained_models/optimal_model.pth.tar'
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

            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
            seqs = k_prev_words  # (k, 1)
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)

            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)

                awe, _ = decoder.attention(encoder_out, h)

                gate = decoder.sigmoid(decoder.f_beta(h))
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

                scores = decoder.fc(h) 
                scores = F.log_softmax(scores, dim=1)

                scores = top_k_scores.expand_as(scores) + scores

                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

                prev_word_inds = top_k_words / vocab_size
                next_word_inds = top_k_words % vocab_size

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
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
            ref[img_id]= list(
                map(lambda c: ' '.join([rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]),
                    img_caps)) 
            references.append(img_captions)
            # Hypotheses 
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            #print(hypotheses)
            hyp[img_id]= [' '.join([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])]
            assert len(references) == len(hypotheses)

        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        
        # save the references and hypotheses. Them can be used to caculate scores
        with open(os.path.join('saved_data/evaluation','test_ref.json'),'w') as f1:
            json.dump(ref, f1)
        with open(os.path.join('saved_data/evaluation','test_hyp.json'),'w') as f1:
            json.dump(hyp, f1)
        print(calc_scores(ref,hyp))
