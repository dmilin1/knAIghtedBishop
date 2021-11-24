
# knAIghtedBishop

#### A chess AI that uses [lichess.org](https://lichess.org) as a front end.

knAIghtedBishop is the successor to [BishopAI](https://github.com/dmilin1/BishopAI). It is a complete rewrite with a few goals:
- Cleaner modernized code
- More accurate prediction
- Faster evaluation

The primary ways knAIghtedBishop differentiates itself are:
- Rebuilt to use TensorFlow 2
- Support for batched predictions
- Support for faster low precision inference using TFLite
- Tree paths remembered between moves
- Cached board predictions to prevent unnecessary inferences

knAIghtedBishop operates by building a tree representing the possible states for the game. The tree begins as a single node representing the starting position. The AI then performs simulations of future game paths and builds out the tree accordingly. The number of simulations is predefined and the path the simulation chooses is dictated by an Upper Confidence Bounds algorithm with parameters provided by the policy and value outputs of a convolutional neural network.


### Requirements
- Python 3.8
- Tensorflow (GPU optional but highly recommended)
- The following pip installable python packages: python-chess, numpy, keras, tensorflow, berserk, dotenv
- a [Lichess.com](https://lichess.org/  "Lichess.com") account and API token


### How To Use
Make sure all the requirements have been met. The setup steps for the requirements differ heavily between systems so this guide will only cover information assuming the requirements have already been properly met.

------------
1. Clone the [GitHub project](https://github.com/dmilin1/BishopAI  "GitHub project") to a file location of your choosing.
2. cd into the downloaded directory.
3. Create a `.env` file in the root of the project and set your Lichess API token `LICHESS_TOKEN = 'XXXX'`

To play Chess:
1. Start the Lichess server with `py run.py play`
2. Log into Lichess.com, go to your bot page, and play the bot.

To train a net from scratch:
1. In the file `brain.py` change this line to point to the `.pgn` file you want to train on `pgn = open("D:/Chess Datasets/big.pgn")`
2. Start training with `py run.py train -n`
3. The model auto saves at the end of every epoch. Training can be resumed by excluding the `-n`.