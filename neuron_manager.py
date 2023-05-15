import pygame
import grid



class Neuron:
    def __init__(self):
        self.loc = 0, 0
        self.axon_hillock = 0
        self.nt_sum = 0

    def should_fire(self) -> bool:
        if self.nt_sum >= self.axon_hillock:
            return True




class NeuronManager:

    def __init__(self, grid_in: grid.Grid):
        self.grid = grid_in
