import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class World:
    def __init__(self, size):
        self.size = size
        self.matrix = np.zeros(size)

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
        return f"Id:{self.id}, Cs:{self.coords}, \n" \
               f"Gen:{self.genotype}"


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

        self.generation = [Agent(i) for i in range(1, count_of_agents+1)]

        for i, agent in enumerate(self.generation):
            new_coords = list(indexes[i])
            self.world.set_agent(agent, new_coords)

        self.world_states.append(np.copy(self.world.matrix))

    def iter_generation(self):
        for i in range(4):
            for agent in self.generation:
                rule = self.rules[agent.genotype[i]]

                new_coords = [agent.coords[0], agent.coords[1]]
                if rule == "up":
                    new_coords[0] -= 1
                if rule == "down":
                    new_coords[0] += 1
                if rule == "left":
                    new_coords[1] -= 1
                if rule == "right":
                    new_coords[1] += 1

                self.world.set_agent(agent, new_coords)
                self.world_states.append(np.copy(self.world.matrix))

    def animate(self):
        fig = plt.figure(figsize=(8, 8))
        im = plt.imshow(self.world_states[0])

        def animate_func(i):
            print(i)
            im.set_array(self.world_states[i])
            return [im]

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            interval=100,  # in ms
            frames=len(self.world_states)
        )

        plt.show()
        # anim.save("test_anim.gif", writer="imagemagick")

    def __str__(self):
        return str(self.world)


programm = Programm(prob_of_agents=0.05, size=(8, 8))
programm.create_generation()
programm.iter_generation()
programm.iter_generation()
programm.iter_generation()
programm.iter_generation()
print(len(programm.world_states))
programm.animate()
