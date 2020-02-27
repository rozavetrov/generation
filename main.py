import numpy as np
import random
import time
import threading
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox

# GLOBALS
AGENT_ID = 0
WORLD_SIZE = (100, 300)
SHARE_OF_AGENTS = 0.001
SIZE_OF_GENOTYPE = 5
DELAY_OF_ANIMATION = 25  # ms
VALUE_OF_FITNESS_FUNC = 1000
COUNT_OF_POPULATION = int(WORLD_SIZE[0] * WORLD_SIZE[1] * SHARE_OF_AGENTS)
WORLD_VIEW = "HEALTH"


class Agent:
    id_iter = itertools.count()

    def __init__(self, coords=None):
        if coords is None:
            coords = np.array((0, 0))

        self.id = next(Agent.id_iter)
        self.coords = coords
        self.health = 100
        self.genotype = np.random.randint(5, size=SIZE_OF_GENOTYPE)
        self.number_of_generation = 1

    def __str__(self):
        return f"Id:{self.id}, Cs:{self.coords} \n" \
               f"Gen:{self.genotype} \n" \
               f"N_GEN: {self.number_of_generation} \n" \
               f"HEALTH: {self.health}"


class World:
    def __init__(self, size):
        self.size = size
        self.current_generation = defaultdict()

        self.matrix_of_ids = np.zeros(size, dtype=int)
        self.matrix_of_generations = np.zeros(size, dtype=float)
        self.matrix_of_health = np.zeros(size, dtype=int)

        self.rules = {
            0: lambda agent: self.move(agent, "up"),
            1: lambda agent: self.move(agent, "down"),
            2: lambda agent: self.move(agent, "left"),
            3: lambda agent: self.move(agent, "right"),
            4: lambda agent: self.move(agent, "none"),
        }

    def create_start_generation(self, view):
        for _ in range(COUNT_OF_POPULATION):
            agent = Agent()
            self.current_generation[agent.id] = agent
            self.plant_agent_to_random_place(agent)

        return self.get_matrix_with_view(view)

    def make_step(self, i, view):
        agents_to_kill = []

        for agent in self.current_generation.values():
            self.rules[agent.genotype[i]](agent)
            if agent.health <= 0:
                agents_to_kill.append(agent)

        for agent in agents_to_kill:
            self.kill_agent(agent)
            new_agent = self.reproduction()
            self.current_generation[new_agent.id] = new_agent
            self.plant_agent_to_random_place(new_agent)

        return self.get_matrix_with_view(view)

    def get_matrix_with_view(self, view):
        if view == "IDS":
            return self.matrix_of_ids
        elif view == "GENERATION":
            return self.matrix_of_generations
        elif view == "HEALTH":
            return self.matrix_of_health

    def reproduction(self):
        global AGENT_ID

        possible_parents_indexes = np.array(np.where(self.matrix_of_ids != 0)).T
        parents_indexes = random.choices(possible_parents_indexes, k=2)

        agent1 = self.get_agent(coords=parents_indexes[0])
        agent2 = self.get_agent(coords=parents_indexes[1])

        count_of_gens = int(SIZE_OF_GENOTYPE / 2)
        child_gens_1 = random.choices(agent1.genotype, k=count_of_gens)
        child_gens_2 = random.choices(agent2.genotype, k=SIZE_OF_GENOTYPE - count_of_gens)
        new_gens = np.array(child_gens_1 + child_gens_2)

        child_agent = Agent()
        child_agent.number_of_generation = max(agent1.number_of_generation, agent2.number_of_generation) + 1

        child_agent.genotype = new_gens

        return child_agent

    def get_agent(self, coords):
        a_id = self.matrix_of_ids[coords[0], coords[1]]
        return self.current_generation[a_id]

    def kill_agent(self, agent):
        self.current_generation.pop(agent.id)
        self.unset_agent(agent)

        return agent

    def plant_agent_to_random_place(self, agent):
        possible_indexes = np.array(np.where(self.matrix_of_ids == 0)).T
        seat_indexes = random.choice(possible_indexes)
        self.set_agent(agent, seat_indexes)

        return seat_indexes

    def unset_agent(self, agent):
        self.matrix_of_ids[agent.coords[0], agent.coords[1]] = 0
        self.matrix_of_generations[agent.coords[0], agent.coords[1]] = 0
        self.matrix_of_health[agent.coords[0], agent.coords[1]] = 0

    def set_agent(self, agent, new_coords):
        if new_coords[0] > self.size[0] - 1:
            new_coords[0] = 0

        if new_coords[0] < 0:
            new_coords[0] = self.size[0] - 1

        if new_coords[1] > self.size[1] - 1:
            new_coords[1] = 0

        if new_coords[1] < 0:
            new_coords[1] = self.size[1] - 1

        if self.matrix_of_ids[new_coords[0], new_coords[1]] == 0:
            self.matrix_of_ids[agent.coords[0], agent.coords[1]] = 0
            self.matrix_of_generations[agent.coords[0], agent.coords[1]] = 0
            self.matrix_of_health[agent.coords[0], agent.coords[1]] = 0

            agent.coords = new_coords

            self.matrix_of_ids[agent.coords[0], agent.coords[1]] = agent.id
            self.matrix_of_generations[agent.coords[0], agent.coords[1]] = agent.number_of_generation
            self.matrix_of_health[agent.coords[0], agent.coords[1]] = agent.health

    def move(self, agent, direction):
        new_coords = [agent.coords[0], agent.coords[1]]
        agent.health -= 20

        if direction == "up":
            new_coords[0] -= 1
        if direction == "down":
            new_coords[0] += 1
        if direction == "left":
            new_coords[1] -= 1
        if direction == "right":
            new_coords[1] += 1
        if direction == "none":
            agent.health += 24

        self.set_agent(agent, new_coords)

    def __str__(self):
        return str(self.matrix_of_ids)


class Program:
    def __init__(self, size):
        self.world = World(size)
        self.current_state = None
        self.count_of_iterations = 0
        self.time_of_exec = 0

    def fitness_func(self):
        return sum(
            agent.health for agent in self.world.current_generation.values()) / COUNT_OF_POPULATION

    def gen_alg(self):
        start_time = time.time()

        self.current_state = self.world.create_start_generation(WORLD_VIEW)
        time.sleep(DELAY_OF_ANIMATION / 1000)

        while self.fitness_func() < VALUE_OF_FITNESS_FUNC:
            self.count_of_iterations += 1
            for i in range(SIZE_OF_GENOTYPE):
                self.current_state = self.world.make_step(i, WORLD_VIEW)
                time.sleep(DELAY_OF_ANIMATION / 1000)

        self.time_of_exec = time.time() - start_time

    def heatmap(self, data):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        box_ax = ax.get_position()
        im = ax.imshow(data, vmax=100)

        text_box = plt.axes([box_ax.x0, box_ax.y0, 0.2, 0.1])
        text_box.set_visible(False)

        return fig, im, text_box

    def start(self):
        thread = threading.Thread(target=self.gen_alg)
        thread.daemon = True
        thread.start()

        time.sleep(1)
        fig, im, text_box = self.heatmap(self.current_state)

        def animate_func(i):
            text = text_box.text(0, 0, f"Iterations: {self.count_of_iterations}", size="medium")
            im.set_array(self.current_state)
            return [im, text]

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            interval=DELAY_OF_ANIMATION,  # in ms
            blit=True,
            repeat=False
        )

        plt.show()

    def __str__(self):
        return f"FITNESS FUNC: {self.fitness_func()} \n" \
               f"ITERATIONS: {self.count_of_iterations} \n" \
               f"POPULATION: {COUNT_OF_POPULATION} \n" \
               f"TIME: {self.time_of_exec}"

    # def animate(self):
    #     first_world_state = self.world.create_start_generation(WORLD_VIEW)
    #
    #     fig, ax = plt.subplots()
    #     ax.set_title("GENERATION ALGORITHM")
    #
    #     im = plt.imshow(first_world_state)
    #
    #     def animate_func(args):
    #         im.set_clim(0, 100)
    #         im.set_array(args[0])
    #         return [im] + args[1]
    #
    #     def make_step():
    #         while self.fitness_func() < VALUE_OF_FITNESS_FUNC:
    #             self.count_of_iterations += 1
    #             for i in range(SIZE_OF_GENOTYPE):
    #                 anim_texts = []
    #                 world_state = self.world.make_step(i, WORLD_VIEW)
    #                 for agent in self.world.current_generation.values():
    #                     i, j = agent.coords
    #                     anim_texts.append(im.axes.text(j, i, world_state[i, j],
    #                                                    ha="center", va="center", color="w", size="small"))
    #
    #                 yield world_state, anim_texts
    #
    #     # make_step()
    #     anim = animation.FuncAnimation(
    #         fig,
    #         animate_func,
    #         make_step,
    #         interval=DELAY_OF_ANIMATION,  # in ms
    #         blit=True,
    #         repeat=False
    #     )
    #
    #     plt.show()


program = Program(size=WORLD_SIZE)
program.start()
print(program)
