import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from alive_progress import alive_bar

def set_dropout(model, drop_rate=0.5):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout2d):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def standard_prediction(model, X):
    model = model.eval()
    logits = model(Variable(X))[0]
    pred = torch.argmax(logits.squeeze(), dim=0).cpu().detach().float().unsqueeze(0)
    
    return pred

def predict_dist(model, X, T=100):
    
    model = model.train()

    softmax_out_stack = []

    with alive_bar(T, title=f'MC-Dropout:') as bar:
        for mc_i in range(T):
            
            logits = model(Variable(X))[0]
            softmax_out = F.softmax(logits, dim=1)
            
            del logits
            torch.cuda.empty_cache()

            # remove batch dim
            softmax_out = softmax_out.squeeze(0)

            softmax_out_stack.append(softmax_out)

            bar.text('[MC-it: ' + str(mc_i + 1) + ']')
            bar()

    softmax_out_stack = torch.stack(softmax_out_stack)  

    return softmax_out_stack

def monte_carlo_dropout_proc(model, x, T=1000, dropout_rate=0.5):

    standard_pred = standard_prediction(model, x)

    set_dropout(model, drop_rate=dropout_rate)
    
    softmax_dist = predict_dist(model, x, T)
    
    pred_std = torch.std(softmax_dist, dim=0)
    
    del softmax_dist
    torch.cuda.empty_cache()

    pred_std = pred_std.gather(0, standard_pred.long()).squeeze(0)
    
    model = model.eval()

    return pred_std


