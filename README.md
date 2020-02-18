### cross entropy loss

- binary-classification cross entropy:
  - without activation:nn.BCELoss()(nn.Sigmoid()(X),t)
  - with activation:nn.BCEWithLogitsLoss()(X,t)
- multi-classification cross entropy:
  - without activation:nn.NLLLoss()(nn.LogSoftmax()(X),t)
  - with activation:nn.CrossEntropyLoss()(X,t)