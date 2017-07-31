This is a bot designed for playing generals.io using a convolutional neural network (http://bot.generals.io/profiles/%5BBot%5D%20%5Buw%5D%20zxqfl).

The program is undocumented and probably difficult to understand. When writing it I didn't plan for it to be read by other people. However, here is some information that will help you understand it if you're interested:

- The most important files are `featurize.py`, which converts a game state to a feature vector, and `tf_model.py`, which defines and trains a model to map feature vectors to move probabilities.
- The paths in `1v1_sorted_paths.txt` refer to files from http://dev.generals.io/replays.

The workflow looks like this:

1. Get some replays and feed them to `simulator.js`.
2. Pipe the output of the simulator into `featurize.py`, which will write features to files. (This takes a while.)
3. Train a model using `tf_model.py` with the features produced in the previous step.
4. Once the model is trained, use it on the server with `run_bot.py`.

You will want a GPU for step 3 and possibly step 4 too.
