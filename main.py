import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# GLOBALS
AGENT_ID = 1
WORLD_SIZE = (200, 200)
SHARE_OF_AGENTS = 0.01
SIZE_OF_GENOTYPE = 30
DELAY_OF_ANIMATION = 100  # ms


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


class World:
    def __init__(self, size, share_of_agents):
        self.size = size
        self.share_of_agents = share_of_agents
        self.matrix = np.zeros(size)
        self.current_generation = []
        self.world_states = []
        self.rules = {
            0: lambda agent: self.move(agent, "up"),
            1: lambda agent: self.move(agent, "down"),
            2: lambda agent: self.move(agent, "left"),
            3: lambda agent: self.move(agent, "right"),
            4: lambda agent: self.move(agent, "none"),
        }

    def create_start_generation(self):
        global AGENT_ID

        count_of_agents = int(self.size[0] * self.size[1] * self.share_of_agents)
        for _ in range(count_of_agents):
            agent = Agent(AGENT_ID)
            self.current_generation.append(agent)
            self.plant_agent_to_random_place(agent)

            AGENT_ID += 1

        return np.copy(self.matrix)

    def make_action(self, i):
        for agent in self.current_generation:
            self.rules[agent.genotype[i]](agent)

        return np.copy(self.matrix)

    def plant_agent_to_random_place(self, agent):
        possible_indexes = np.where(self.matrix == 0)
        possible_indexes = list(zip(possible_indexes[0], possible_indexes[1]))

        seat_indexes = random.choice(possible_indexes)
        self.set_agent(agent, seat_indexes)

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
            agent.health += 20

        self.set_agent(agent, new_coords)

    def __str__(self):
        return str(self.matrix)

    # def reproduction(self, agent1, agent2):
    #     new_agent = Agent()
    #     new_agent = np.array(
    #         random.sample(agent1.genotype.tolist(), k=3) + random.sample(agent2.genotype.tolist(), k=3))


class Programm:
    def __init__(self, size, share_of_agents):
        self.world = World(size, share_of_agents)
        self.world_states = []

    def start(self):
        first_world_state = self.world.create_start_generation()
        self.world_states.append(first_world_state)

        for i in range(SIZE_OF_GENOTYPE):
            world_state = self.world.make_action(i)
            self.world_states.append(world_state)

        self.animate()

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


programm = Programm(share_of_agents=SHARE_OF_AGENTS, size=WORLD_SIZE)
programm.start()
