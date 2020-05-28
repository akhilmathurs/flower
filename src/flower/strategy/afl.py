# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Agnostic Federated Averaging (AFL) strategy.

"""


from typing import Callable, Dict, List, Optional, Tuple
from logging import ERROR, INFO

import numpy as np

from flower.client_manager import ClientManager
from flower.client_proxy import ClientProxy
from flower.typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Weights
from flower.logger import configure, log

from .aggregate import aggregate, weighted_loss_avg
from .fedavg import FedAvg
from .parameter import parameters_to_weights, weights_to_parameters
from .strategy import Strategy


class AFL(FedAvg):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        afl_learning_rate: float = 0.01,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        accept_failures: bool = True,
    ) -> None:
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.latest_lambdas = np.ones(min_fit_clients) * 1.0 / min_fit_clients
        self.pre_weights: Weights
        self.learning_rate = afl_learning_rate

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate model weights using an evaluation function (if provided)."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        return self.eval_fn(weights)

    def on_configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.pre_weights = weights
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = (parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def on_configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = (parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def on_aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        # Convert results

        self.post_weights = self.pre_weights

        def project(y):
            """ algorithm comes from:
            https://arxiv.org/pdf/1309.1541.pdf
            """
            u = sorted(y, reverse=True)
            x = []
            rho = 0
            for i in range(len(y)):
                if (u[i] + (1.0 / (i + 1)) * (1 - np.sum(np.asarray(u)[:i]))) > 0:
                    rho = i + 1
            lambda_ = (1.0 / rho) * (1 - np.sum(np.asarray(u)[:rho]))
            for i in range(len(y)):
                x.append(max(y[i] + lambda_, 0))
            return x

        def aggregate(wsolns):
            total_weight = 0.0
            base = [0] * len(wsolns[0][1])

            for (w, soln) in wsolns:
                total_weight += w
                for i, v in enumerate(soln):
                    base[i] += w * v.astype(np.float64)

            averaged_soln = [v / total_weight for v in base]
            return averaged_soln


        log(INFO, "O a, here ")

        losses = []
        solns = []
        for idx, val in enumerate(results):
            _, (parameters, _, _, _, loss, gradients) = val
            losses.append(loss)
            solns.append((self.latest_lambdas[idx], parameters_to_weights(gradients)))



        avg_gradient = aggregate(solns)

        for v, g in zip(self.pre_weights, avg_gradient):
            v -= self.learning_rate * g

        for idx in range(len(self.latest_lambdas)):
            self.latest_lambdas[idx] += self.learning_rate * losses[idx]

        self.latest_lambdas = project(self.latest_lambdas)

        # log("O a, here ")
        # log(self.post_weights)

        for k in range(len(self.post_weights)):
            self.post_weights[k] = (
                (self.post_weights[k] * rnd + self.pre_weights[k]) * 1.0 / (rnd + 1)
            )

        return self.post_weights

    def on_aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        return weighted_loss_avg([evaluate_res for _, evaluate_res in results])

    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float]
    ) -> bool:
        """Always continue training."""
        return True
