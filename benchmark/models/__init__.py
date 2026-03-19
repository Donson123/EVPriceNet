from benchmark.registry import register
from benchmark.models.flat_tariff import FlatTariff
from benchmark.models.q_learning import QLearning
from benchmark.models.dqn import DQN
from benchmark.models.pddpg import PDDPG
from benchmark.models.ddpg import DDPG

def register_all(env_kwargs, epochs, congestion_hours):

    register("flat_tariff", lambda: FlatTariff(env_kwargs=env_kwargs, epochs=epochs))
    register("q_learning",  lambda: QLearning(env_kwargs=env_kwargs, epochs=epochs))
    register("dqn",         lambda: DQN(env_kwargs=env_kwargs, epochs=epochs))
    register("ddpg",        lambda: DDPG(env_kwargs=env_kwargs, epochs=epochs))
    register("pddpg",       lambda: PDDPG(env_kwargs=env_kwargs, epochs=epochs))
    