import time
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
# get the encoder, decoder, attention models
from models.encoder import Encoder
from models.decoder_Attention import DecoderWithAttention
# import the dataset pipline
from train.dataset import MyDataset
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # whether using gpu

data_folder = '/saved_data' 
data_name = 'trained_models'

training_loss=[]
training_time=[]
training_acc=[]

val_loss=[]
val_time=[]
val_acc=[]
val_bleu4=[]

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of LSTM
dropout = 0.5

cudnn.benchmark = True

# Training parameters
start_epoch = 0 
epochs = 30  # epochs number
epochs_since_improvement = 0  # number of epochs that no improvement in validation BLEU
batch_size = 32 # batch size
workers = 1 

encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # decoder learning rate

grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention'
best_bleu4 = 0.  # BLEU-4 score
print_freq = 100  # print (ouput) training/validation stats every __ batches
fine_tune_encoder = False  # whether fine-tune the encoder
checkpoint = 'saved_data/trained_models/checkpoint_trained_models.pth.tar'  # path to checkpoint: None means the first training. otherwise continue previous traning

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients to avoid explosion of gradients.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    record the value of loss, accuracy, time
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Adjust the learning rate by times a shrink factor.
    """
    print("\nDecay learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves the checkpoint of models into .pth.tar file
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'saved_data/trained_models/checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store this best model
    if is_best:
        filepath='saved_data/trained_models/best_checkpoint_' + data_name + '.pth.tar'
        torch.save(state, filepath)

def main_train():
    """
    Training on train set and do validation on val dataset.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    with open('saved_data/word2index/word2idx.json', 'r') as j: # load vocabulary
        word_map = json.load(j)

    # Whether using pretrained checkpoint to continue training
    if checkpoint is None:
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder) # whether fine-tune the encoder
        # Adam Optimization algorithm
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
    else:
        print('Continue previous training')
        checkpoint = torch.load(checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch'] + 1
        print(start_epoch)
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    decoder = decoder.to(device) # Move decoder to GPU, if available
    encoder = encoder.to(device) # # Move encoder to GPU, if available

    # Use CrossEntropy Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # Normalize the image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Load train and validation dataset
    train_loader = torch.utils.data.DataLoader(
        MyDataset('saved_data/', 'train', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        MyDataset('saved_data/', 'val', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Training epoches
    for epoch in range(start_epoch, epochs):
        # Decrease the learning rate if there is no improvement for 8 consecutive epochs, and early stop the training if no improvement after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        # begin training
        train(train_loader=train_loader, 
              encoder=encoder,
              decoder=decoder,
              criterion=criterion, # loss function
              encoder_optimizer=encoder_optimizer, # Adam Optimization algorithm
              decoder_optimizer=decoder_optimizer, # Adam Optimization algorithm
              epoch=epoch)

        # Validating on val dataset
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs that has no improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        print ('The bleu4 score of epoch {} is {}\n'.format(epoch,best_bleu4))
    with open('saved_data/evaluation/training_time.json','w') as f1:
        json.dump(training_time, f1)
    with open('saved_data/evaluation/training_loss.json','w') as f2:
        json.dump(training_loss, f2)
    with open('saved_data/evaluation/training_acc.json','w') as f3:
        json.dump(training_acc, f3)
    with open('saved_data/evaluation/val_time.json','w') as f4:
        json.dump(val_time, f4)
    with open('saved_data/evaluation/val_loss.json','w') as f5:
        json.dump(val_loss, f5)
    with open('saved_data/evaluation/val_acc.json','w') as f6:
        json.dump(val_acc, f6)
    with open('saved_data/evaluation/val_bleu.json','w') as f7:
        json.dump(val_bleu4, f7)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Training function

    :param criterion: loss function
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    """
    decoder.train()  # dropout and batchnorm is used in decoder
    encoder.train() # image feature extraction

    #batch_time = AverageMeter()  # forward prop. + back prop. time
    #data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    print('Begin training')
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        #data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # encoder, decoder Forward function.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:] # targets are all words between <start> and  <end>

        # pack_padded_sequence can remove pads(0) that not decode at
        scores= pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets= pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets) # Calculate loss

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        #batch_time.update(time.time() - start)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, top5=top5accs))
    training_time_1epoch= time.time() - start
    # get the loss and accuracy of training
    print('\n * Training: Time - {training_time_1epoch:.3f}, LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(
                training_time_1epoch=training_time_1epoch,
                loss=losses,
                top5=top5accs))
    training_loss.append(round(losses.avg,4))
    training_time.append(round(training_time_1epoch,4))
    training_acc.append(round(top5accs.avg,4))

def validate(val_loader, encoder, decoder, criterion):
    """
    validation on val dataset: calculate bleu4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    # batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            # batch_time.update(time.time() - start)

            # start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        validation_time = time.time()-start
        print(
            '\n * Validation: Time - {validation_time:.3f}, LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                validation_time=validation_time,
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
        val_loss.append(round(losses.avg,4))
        val_time.append(round(validation_time,4))
        val_acc.append(round(top5accs.avg,4))
        val_bleu4.append(round(bleu4,4))
    return bleu4
