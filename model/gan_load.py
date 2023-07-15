import torch.nn as nn

n_hidden = 100


class Generator(nn.Module):
    def __init__(self, n_input=11,  is_g=True):
        super().__init__()
        self.n_input = n_input
        self.g = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        self.normal = nn.Linear(n_hidden, 11)
        self.is_g = is_g

    def forward(self, x):
        g = self.g(x)
        if self.is_g:
            y = self.normal(g)

        return y


class Discriminator(nn.Module):
    def __init__(self, n_input=11):
        super().__init__()
        self.d_1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, 1),
            nn.ReLU(True)
        )
        self.d_2 = nn.Sequential(
            nn.Linear(11, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, 1),
            nn.ReLU(True)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        s_1 = self.d_1(x)
        s_2 = self.d_2(y)
        s = s_1 + s_2
        return self.sig(s)
