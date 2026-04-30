import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Transformer
from data import sentence, make_data, MyDataSet, src_vocab_size, tgt_vocab_size

# 分头后的q、k、v词向量长度，依照原文我们都设为64
# 原文：queries and kes of dimention d_k,and values of dimension d_v .所以q和k的长度都用d_k来表示
d_k = d_v = 64
# 多头注意力中head的个数，原文：we employ h = 8 parallel attention layers, or heads
n_heads = 8
# FFN的隐藏层神经元个数
d_ff = 2048
# 用来表示一个词的向量长度
d_model = 512
# Encoder Layer 和 Decoder Layer的个数
n_layers = 6

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff).cuda()
model.train()

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 获取输入
enc_inputs, dec_inputs, dec_outputs = make_data(sentence)      # (2, 5), (2, 6), (2, 6), 2个句子, 每个句子分词后的id

# 构建DataLoader
loader = DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)

for epoch in range(1000):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()    # (batch_size, 5), (batch_size, 6), (batch_size, 6)
        outputs = model(enc_inputs, dec_inputs)        # (batch_size*tgt_len, tgt_vocab_size)
        loss = criterion(outputs, dec_outputs.view(-1))           # (batch_size*tgt_len, )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}], Loss: {loss.item()}')

torch.save(model, f'MyTransformer_temp.pth')