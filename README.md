# AICity_task2

TF implementation:



log:


 20190402: MobileNet 224x224x3 with cosine loss, use half training set, lr=0.1, train FC only, 1000 iter - no convergence
 
 
 20190403: MobileNet 224x224x3 with 0/1 CE loss, use full training set, lr=0.01, train full model, 1000 iter - CE~0.6
 
 
 20190408: Custom network 128x128x3 with concated inputs and 0/1 CE loss, use full training set, lr=0.01/0.005/0.001, 70k iter - CE~0.2
 
 data may need preprocessing as well
