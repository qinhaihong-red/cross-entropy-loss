cross entropy loss

### binary-classification cross entropy:

$E(w)=-\sum_{n}^{N}(t_{n}\log y_{n}+(1-t_{n})\log (1-y_{n})) $

$\bar{E(w)}=\frac{E(w)}{N}$


- without activation:nn.BCELoss()(nn.Sigmoid()(X),t)
- with activation:nn.BCEWithLogitsLoss()(X,t)


### multi-classification cross entropy:
$E(w)=-\sum_{n}^{N} \sum_{k}^{K} t_{nk}\log y_{nk}$

$\bar{E(w)}=\frac{E(w)}{N}$

  
- without activation:nn.NLLLoss()(nn.LogSoftmax()(X),t)
- with activation:nn.CrossEntropyLoss()(X,t)
