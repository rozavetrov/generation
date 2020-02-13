import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class World:
    def __init__(self, size):
        self.size = size
        self.matrix = np.full(size, 0)

    def set_agent(self, agent, new_coords):
        is_in_matrix = (np.array(new_coords) < self.size).all()
        if is_in_matrix:
            self.matrix[agent.coords[0], agent.coords[1]] = 0
            agent.coords = new_coords
            self.matrix[agent.coords[0], agent.coords[1]] = agent.id

    def __str__(self):
        return str(self.matrix)


class Agent:
    def __init__(self, a_id, coords=None):
        if coords is None:
            coords = np.array((0, 0))

        self.id = a_id
        self.coords = coords
        self.health = int()
        self.genotype = np.random.randint(4, size=4)

    def __str__(self):
        return str(self.genotype)


class Programm:
    def __init__(self, prob_of_agents=0.5, size=(8, 8)):
        self.world = World(size)
        self.generation = list()
        self.probability = prob_of_agents
        self.rules = {
            0: "up",
            1: "down",
            2: "left",
            3: "right"
        }
        self.world_states = []

    def create_generation(self):
        mask = np.random.choice([0, 1], size=self.world.matrix.shape, p=((1 - self.probability), self.probability))
        count_of_agents = np.count_nonzero(mask)

        indexes = np.where(mask == 1)
        indexes = list(zip(indexes[0], indexes[1]))

        self.generation = [Agent(i) for i in range(count_of_agents)]

        for i, agent in enumerate(self.generation):
            new_coords = list(indexes[i])
            self.world.set_agent(agent, new_coords)

        print(self.world)

    def iter_generation(self):
        for i in range(4):
            for agent in self.generation:
                rule = self.rules[agent.genotype[i]]

                new_coords = [agent.coords[0], agent.coords[1]]
                if rule == "up":
                    new_coords[0] += 1
                if rule == "down":
                    new_coords[0] -= 1
                if rule == "left":
                    new_coords[1] -= 1
                if rule == "right":
                    new_coords[1] += 1

                self.world.set_agent(agent, new_coords)
                self.world_states.append(self.world.matrix)
                # print(self.world)

    def animate(self):
        fig = plt.figure(figsize=(8, 8))
        a = self.world_states[0]
        im = plt.imshow(a)

        def animate_func(i):
            im.set_array(self.world_states[i])
            return [im]

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            interval=1000,  # in ms
            frames=len(self.world_states)
        )

        anim.save("gen_alg.gif", writer="imagemagick")

    def __str__(self):
        return str(self.world)


programm = Programm(0.1)
programm.create_generation()
programm.iter_generation()
programm.animate()
