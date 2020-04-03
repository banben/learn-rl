from vizdoom import *        # Doom Environment

def create_environment():
    game = DoomGame()
    
    # Load the correct configuration
    game.load_config("basic.cfg")
    
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")
    
    game.init()
    
    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    
    return game, possible_actions

game, possible_actions = create_environment()