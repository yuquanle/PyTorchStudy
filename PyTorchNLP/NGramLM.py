import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        # 简单的查找表，存储固定字典和大小的嵌入。通常用于存储单词嵌入并使用索引来检索它们。
        # 模块的输入是一个索引列表，输出是相应的单词嵌入。
        # 初始化一个矩阵(vocab_size * embedding_dim)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # 取那个矩阵里面inputs所在的行
        embdeds = self.embeddings(inputs).view((1, -1))

        out = F.relu(self.linear1(embdeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs, self.embeddings


# model
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)


for epoch in range(6000):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))


        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs,embedd = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
    if epoch % 100 == 0:
        print("total loss: ", total_loss[0])  # The loss decreased every iteration over the training data!

for i in range(0, len(vocab)):
    lookup_tensor = torch.LongTensor([i])
    print(ix_to_word[i])
    print(embedd(autograd.Variable(lookup_tensor)))

