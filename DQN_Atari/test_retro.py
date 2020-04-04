import retro

def main():
    env = retro.make(game='SpaceInvaders-Atari2600')
    obs = env.reset()
    print("The size of our frame is: ", env.observation_space)
    print("The action size is : ", env.action_space.n)
    env.close()


if __name__ == "__main__":
    main()