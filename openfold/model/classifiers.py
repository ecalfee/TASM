# Copyright 2026 Adaptive Biotechnologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, beta0, beta1, freeze_params, **kwargs):
        """
        Args:
            beta0 (float): intercept in logistic regression
            beta1 (float): slope in logistic regression
            freeze_params (bool): whether to freeze parameters
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

        # initialize parameters
        self.linear.bias.data.fill_(beta0)
        self.linear.weight.data.fill_(beta1)

        if freeze_params:
            for param in self.linear.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output of logistic regression on logodds scale.

        Args:
            x (torch.Tensor): tcr-pmhc iptm scores
        Returns:
            y (torch.Tensor, size=[batch_size, 1]): binding probs
        """
        # input must be have the same device and dtype as model
        x = torch.as_tensor(
            x,
            device=self.linear.bias.device,
            dtype=self.linear.bias.dtype
        )
        # convert from Tensor(a) to Tensor([a])
        if x.shape == ():
            x = x.unsqueeze(dim=0)
        # convert from Tensor([a]) to Tensor([[a]])
        if len(x.shape) == 1:
            x = x[:, None]

        #y_pred = torch.sigmoid(self.linear(x))
        y_pred_logodds = self.linear(x)

        return y_pred_logodds