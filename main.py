import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# GLOBALS
AGENT_ID = 1
WORLD_SIZE = (15, 15)
SHARE_OF_AGENTS = 0.01
SIZE_OF_GENOTYPE = 15
DELAY_OF_ANIMATION = 500  # ms


class World:
    def __init__(self, size):
        self.size = size
        self.matrix = np.zeros(size)

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

        if direction == "up":
            new_coords[0] -= 1
        if direction == "down":
            new_coords[0] += 1
        if direction == "left":
            new_coords[1] -= 1
        if direction == "right":
            new_coords[1] += 1
        if direction == "none":
            pass

        self.set_agent(agent, new_coords)

    def __str__(self):
        return str(self.matrix)


class Agent:
    def __init__(self, a_id, coords=None):
        if coords is None:
            coords = np.array((0, 0))

        self.id = a_id
        self.coords = coords
        self.health = int(100)
        self.genotype = np.random.randint(5, size=SIZE_OF_GENOTYPE)

    def __str__(self):
        return f"Id:{self.id}, Cs:{self.coords}, \n" \
               f"Gen:{self.genotype}"


class Programm:
    def __init__(self, prob_of_agents=0.5, size=(8, 8)):
        self.world = World(size)
        self.generation = list()
        self.probability = prob_of_agents
        self.rules = {
            0: lambda agent: self.world.move(agent, "up"),
            1: lambda agent: self.world.move(agent, "down"),
            2: lambda agent: self.world.move(agent, "left"),
            3: lambda agent: self.world.move(agent, "right"),
            4: lambda agent: self.world.move(agent, "none"),
        }
        self.world_states = []

    def create_generation(self):
        global AGENT_ID

        mask = np.random.choice([0, 1], size=self.world.matrix.shape, p=((1 - self.probability), self.probability))
        count_of_agents = np.count_nonzero(mask)

        indexes = np.where(mask == 1)
        indexes = list(zip(indexes[0], indexes[1]))

        # self.generation = [Agent(i) for i in range(1, count_of_agents + 1)]
        for i in range(1, count_of_agents + 1):
            self.generation.append(Agent(AGENT_ID))
            AGENT_ID += 1

        for i, agent in enumerate(self.generation):
            new_coords = list(indexes[i])
            self.world.set_agent(agent, new_coords)

        self.world_states.append(np.copy(self.world.matrix))

    def iter_generation(self):
        agents_to_kill = []
        for i in range(SIZE_OF_GENOTYPE):
            for agent in self.generation:
                self.rules[agent.genotype[i]](agent)
                agent.health -= 20

                if agent.health == 0:
                    agents_to_kill.append(agent.id)

            for agent in self.generation:
                """TO DO"""
            self.world_states.append(np.copy(self.world.matrix))

    def kill_agent(self, agent):
        self.world.matrix[agent.coords[0], agent.coords[1]] = 0

        del agent

    def fitness_func(self):
        return sum([agent.health for agent in self.generation]) / len(self.generation)

    def start(self):
        self.create_generation()
        f = self.fitness_func()

        while f != 100:
            self.iter_generation()
            f = self.fitness_func()

    def reproduction(self, agent1, agent2):
        new_agent = Agent()
        new_agent = np.array(
            random.sample(agent1.genotype.tolist(), k=3) + random.sample(agent2.genotype.tolist(), k=3))

    def animate(self):
        fig = plt.figure(figsize=(8, 8))
        im = plt.imshow(self.world_states[0])

        def animate_func(i):
            im.set_array(self.world_states[i])
            return [im]

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            interval=DELAY_OF_ANIMATION,  # in ms
            frames=len(self.world_states)
        )

        plt.show()
        # anim.save("test_anim.gif", writer="imagemagick")

    def __str__(self):
        return str(self.world)


programm = Programm(prob_of_agents=SHARE_OF_AGENTS, size=WORLD_SIZE)
programm.create_generation()
programm.iter_generation()
programm.animate()
