import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_size*2, 1056)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(1056, 528)
        self.relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(528, 528)
        self.relu_3 = nn.ReLU()
        self.fc_4 = nn.Linear(528, 128)
        self.relu_4 = nn.ReLU()
        self.fc_5 = nn.Linear(128, output_size)
        self.relu_5 = nn.ReLU()

    def forward(self, x):
        # У нас двухнаправленная RNN, скрытые состояния в обе стороны
        h0 = torch.zeros(self.num_layers*2, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, self.hidden_size).to(x.device)
        # shape out (32 * 128)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc_1(out)
        out = self.relu_1(out)
        out = self.fc_2(out)
        out = self.relu_2(out)
        out = self.fc_3(out)
        out =  self.relu_3(out)
        out = self.fc_4(out)
        out =  self.relu_4(out)
        out = self.fc_5(out)
        out =  self.relu_5(out)
        # 2 outputs отдельно индексы от start и отдельно индексы от end
        return out[:, 0], out[:, 1]
