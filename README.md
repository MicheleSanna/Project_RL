# Project_RL
A prerequisite for running this project is to install the PokerRL library for the pokre environment.
The list of all the other libraries of my environment is contained in the .env file.

The main.py file is a bit messy, because there are a lot of unused objects declared, this will be fixed later. In the future I will add a command line interface to select the type of training/run, for now you can change the type of training/run by substing the type of player in the "hero" and "opponent" argument of the training loop invocations in main.py. The classes of players available are contained in the "players" folder. There are some istances of player already declared in main.py (for example opponent_dqn, hero_dqn, hero_nfsp etc)
