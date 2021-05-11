import itertools as it
import multiprocessing as mp

from .simulator import LunarLanderSimulator
from .policygradientagent import PolicyGradientAgent
from .framework import Framework
from .display import LunarLanderWindow, UserAgent
import lunarlander.worker as worker

import typer
from typing import Optional

app = typer.Typer()

@app.command()
def main (
        visualize: bool = typer.Option(True, help="Disable visualization while training."),
        episodes: Optional[int] = typer.Option(None, help="Stop training after number of episodes."),
        procs: int = typer.Option(mp.cpu_count()-1, help="Number of parallel training processes."),
        play: bool = typer.Option(False, "--play", help="Take manual control of lunar lander (with W, A, D keys).")
):

    simulator = LunarLanderSimulator()
    agent = PolicyGradientAgent(simulator) if not play else UserAgent(simulator)
    framework = worker.initialize_worker(simulator, agent)
    
    if not play and procs > 0:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(procs, worker.initialize_worker, (simulator, agent, agent.share_state()))

        episodes = range(episodes) if episodes else it.count()
        pool.imap_unordered(worker.run_episode, episodes, 100)
        pool.close()

    try:
        if play or visualize:
            LunarLanderWindow(framework)
        else:
            pool.join()
            
        if not play:
            pool.terminate()
            pool.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app()
