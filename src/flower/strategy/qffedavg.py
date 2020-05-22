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
"""FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING [Li et al., 2020] strategy.
Paper: https://openreview.net/pdf?id=ByexElSYDr
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from flower.client_manager import ClientManager
from flower.client_proxy import ClientProxy
from flower.typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Weights

from .aggregate import aggregate, weighted_loss_avg, aggregate_qffl
from .parameter import parameters_to_weights, weights_to_parameters
from .strategy import Strategy

class QffedAvg(Strategy):
    """Configurable QffedAvg strategy implementation."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        q: float = 0.2,
        qffl_learning_rate: float = 0.1,
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
        self.learning_rate = qffl_learning_rate
        self.q = q

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
        if self.eval_fn is None:
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


        def norm_grad(grad_list):
            # input: nested gradients
            # output: square of the L-2 norm
            client_grads = grad_list[0]
            for i in range(1, len(grad_list)):
                client_grads = np.append(client_grads, grad_list[i]) # output a flattened array
            return np.sum(np.square(client_grads))

        Deltas = []
        hs = []

        weights_before = self.pre_weights
        loss, _ = self.evaluate(weights_before)

        for client, (parameters, num_examples, _) in results:
            new_weights = parameters_to_weights(parameters)
            # plug in the weight updates into the gradient
            grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]        
            Deltas.append([np.float_power(loss+1e-10, self.q) * grad for grad in grads])        
            # estimation of the local Lipchitz constant
            hs.append(self.q * np.float_power(loss+1e-10, (self.q-1)) * norm_grad(grads) + (1.0/self.learning_rate) * np.float_power(loss+1e-10, self.q))

        return aggregate_qffl(weights_before, Deltas, hs)

    def _get_initial_weights(self) -> Weights:
        """Get initial weights from one of the available clients."""
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        return parameters_to_weights(parameters_res.parameters)


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
        return weighted_loss_avg([evaluate_res for client, evaluate_res in results])

    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float]
    ) -> bool:
        """Always continue training."""
        return True