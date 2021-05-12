import torch

model = LSTM_v0_CUDA(*args, **kwargs)
model.load_state_dict(torch.load("./Weights/LSTM_v0_CUDA_0512"))
model.eval()

model = LSTM_v0_CUDA(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()