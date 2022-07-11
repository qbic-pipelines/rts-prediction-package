import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_dropout(model, drop_rate=0.5):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def standard_prediction(model, X):
    model = model.eval()
    logits = model(Variable(X))[0]
    pred = torch.argmax(logits.squeeze(), dim=0).cpu().detach().float().unsqueeze(0)
    
    return pred

def predict_dist(model, X, T=1000):
    
    model = model.train()

    softmax_out_stack = []
    for mc_i in range(T):
        print("MC-it: " + str(mc_i))
        logits = model(Variable(X))[0]
        print("logits: " + str(logits.shape))
        softmax_out = F.softmax(logits, dim=1)
        print("softmax_out: " + str(softmax_out.shape))

        del logits
        torch.cuda.empty_cache()

        # remove batch dim
        softmax_out = softmax_out.squeeze(0)

        softmax_out_stack.append(softmax_out)

    softmax_out_stack = torch.stack(softmax_out_stack)  
    print("softmax_out_stack:" + str(softmax_out_stack.shape))

    return softmax_out_stack

def monte_carlo_dropout_proc(model, x, T=1000, dropout_rate=0.5):

    standard_pred = standard_prediction(model, x)
    print("standard_pred shape:" + str(standard_pred.shape))

    set_dropout(model, drop_rate=dropout_rate)
    
    softmax_dist = predict_dist(model, x, T)
    
    pred_std = torch.std(softmax_dist, dim=0)
    
    del softmax_dist
    torch.cuda.empty_cache()
    
    print("pred std shape:" + str(pred_std.shape))

    pred_std = pred_std.gather(0, standard_pred.long()).squeeze(0)
    print("pred std gather shape:" + str(pred_std.shape))

    model = model.eval()

    return pred_std


