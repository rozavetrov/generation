import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.animation as animation

# GLOBALS
AGENT_ID = 0
WORLD_SIZE = (200, 200)
SHARE_OF_AGENTS = 0.01
SIZE_OF_GENOTYPE = 5
DELAY_OF_ANIMATION = 10  # ms
VALUE_OF_FITNESS_FUNC = 300
COUNT_OF_POPULATION = int(WORLD_SIZE[0] * WORLD_SIZE[1] * SHARE_OF_AGENTS)


class Agent:
    def __init__(self, a_id, coords=None):
        if coords is None:
            coords = np.array((0, 0))

        self.id = a_id
        self.coords = coords
        self.health = int(100)
        self.alive = True
        self.genotype = np.random.randint(5, size=SIZE_OF_GENOTYPE)
        self.number_of_generation = 1

    def __str__(self):
        return f"Id:{self.id}, Cs:{self.coords} \n" \
               f"Gen:{self.genotype} \n" \
               f"N_GEN: {self.number_of_generation} \n" \
               f"HEALTH: {self.health}"


class World:
    def __init__(self, size, share_of_agents):
        self.size = size
        self.share_of_agents = share_of_agents
        self.matrix = np.zeros(size, dtype=int)
        self.current_generation = defaultdict()
        self.rules = {
            0: lambda agent: self.move(agent, "up"),
            1: lambda agent: self.move(agent, "down"),
            2: lambda agent: self.move(agent, "left"),
            3: lambda agent: self.move(agent, "right"),
            4: lambda agent: self.move(agent, "none"),
        }

    def create_start_generation(self):
        global AGENT_ID

        for _ in range(COUNT_OF_POPULATION):
            agent = Agent(_)
            self.current_generation[AGENT_ID] = agent
            self.plant_agent_to_random_place(agent)

            AGENT_ID += 1

        return self.matrix

    def make_step(self, i):
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

        return self.matrix

    def reproduction(self):
        global AGENT_ID

        possible_parents_indexes = np.array(np.where(self.matrix != 0)).T
        parents_indexes = random.choices(possible_parents_indexes, k=2)

        agent1 = self.get_agent(coords=parents_indexes[0])
        agent2 = self.get_agent(coords=parents_indexes[1])

        count_of_gens = random.randint(1, 4)
        child_gens_1 = random.choices(agent1.genotype, k=count_of_gens)
        child_gens_2 = random.choices(agent2.genotype, k=5 - count_of_gens)
        new_gens = list(child_gens_1 + child_gens_2)

        child_agent = Agent(AGENT_ID)
        child_agent.number_of_generation = max(agent1.number_of_generation, agent2.number_of_generation) + 1

        AGENT_ID += 1
        child_agent.genotype = new_gens

        return child_agent

    def get_agent(self, coords=None):
        a_id = self.matrix[coords[0], coords[1]]
        return self.current_generation[a_id]

    def kill_agent(self, agent):
        self.current_generation.pop(agent.id)
        self.matrix[agent.coords[0], agent.coords[1]] = 0

        return agent

    def plant_agent_to_random_place(self, agent):
        seat_indexes = (0, 0)

        possible_indexes = np.array(np.where(self.matrix == 0)).T
        seat_indexes = random.choice(possible_indexes)
        self.set_agent(agent, seat_indexes)

        return seat_indexes

    def set_agent(self, agent, new_coords):
        if new_coords[0] > self.size[0] - 1:
            new_coords[0] = 0

        if new_coords[0] < 0:
            new_coords[0] = self.size[0] - 1

        if new_coords[1] > self.size[1] - 1:
            new_coords[1] = 0

        if new_coords[1] < 0:
            new_coords[1] = self.size[1] - 1

        if self.matrix[new_coords[0], new_coords[1]] == 0:
            self.matrix[agent.coords[0], agent.coords[1]] = 0
            agent.coords = new_coords
            self.matrix[agent.coords[0], agent.coords[1]] = agent.id

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
        return str(self.matrix)


class Program:
    def __init__(self, size, share_of_agents):
        self.world = World(size, share_of_agents)
        self.world_states = []

    # def start(self):
    #     count_of_iterations = 0
    #     first_world_state = self.world.create_start_generation()
    #
    #     while self.fitness_func() < VALUE_OF_FITNESS_FUNC:
    #         count_of_iterations += 1
    #         for i in range(SIZE_OF_GENOTYPE):
    #             world_state = self.world.make_step(i)
    #
    #     print(f"FITNESS FUNC: {self.fitness_func()} \n"
    #           f"ITERATIONS: {count_of_iterations} \n"
    #           f"POPULATION: {COUNT_OF_POPULATION}")

    def fitness_func(self):
        return sum(
            agent.health for agent in self.world.current_generation.values()) / COUNT_OF_POPULATION

    def __str__(self):
        return str(self.world)

    def animate(self):
        first_world_state = self.world.create_start_generation()

        fig = plt.figure(figsize=(10, 10))
        im = plt.imshow(first_world_state)

        def animate_func(world_state):
            im.set_array(world_state)
            return [im]

        def make_step():
            while self.fitness_func() < VALUE_OF_FITNESS_FUNC:
                print(self.fitness_func())
                for i in range(SIZE_OF_GENOTYPE):
                    world_state = self.world.make_step(i)
                    yield world_state

        # make_step()
        anim = animation.FuncAnimation(
            fig,
            animate_func,
            make_step,
            interval=DELAY_OF_ANIMATION,  # in ms
            blit=True,
            repeat=False
        )

        plt.show()


program = Program(share_of_agents=SHARE_OF_AGENTS, size=WORLD_SIZE)
program.animate()
