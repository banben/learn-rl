import h5py
import numpy as np
import pickle

class Memory():
    def __init__(self, max_size=1000000):
        self.buffer_file = h5py.File('memory.h5','w')
        self.buffer_state = self.buffer_file.create_dataset("state", (max_size, 110, 84, 4), dtype='f8')
        self.buffer_action = self.buffer_file.create_dataset("action", (max_size, 8), dtype='i8')
        self.buffer_reward = self.buffer_file.create_dataset("reward", (max_size, ), dtype='f2')
        self.buffer_next_state = self.buffer_file.create_dataset("next_state", (max_size, 110, 84, 4), dtype='f8')
        self.buffer_done = self.buffer_file.create_dataset("done", (max_size, ), dtype='b1')
        self.len = 0
    
    def add(self, experience):
        self.buffer_state[self.len] = experience[0]
        self.buffer_action[self.len] = experience[1]
        self.buffer_reward[self.len] = experience[2]
        self.buffer_next_state[self.len] = experience[3]
        self.buffer_done[self.len] = experience[4]
        self.len = self.len + 1
    
    def sample(self, batch_size):
        buffer_size = self.len
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        batch = [(self.buffer_state[i], self.buffer_action[i], self.buffer_reward[i], self.buffer_next_state[i], self.buffer_done[i]) for i in index]

        return batch

memory = Memory()

exp = pickle.load(open('mem.pkl', 'rb'))
# print(exp[4])
#110 8 8 110 110 
# (110, 84, 4) (8) 1 (110, 84, 4) 1

for i in range(64):
    memory.add(exp)
print(np.shape(exp[4]))
print(np.shape(memory.sample(64)))
