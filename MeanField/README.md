# Annealing Mean Field Approach

To run the code the paths in `<>` brackets in the files `ConditionalExpectation.py`, `train.py` and `LoadGraphDataset.py` have to be replaced with the corresponding correct paths.

Then in order to train a model, one chooses the hyperparameters in `main.py` and executes `main.py` by running `python main.py`. This trains the model, logs to weights and bias and saves the model in checkpoints (and the best model).

To perform conditional expectation on a trained model, one initialized the `ConditionalExpectation` class, loads the dataset via `ConditialExpectation.init_dataset(dataset_name=None)` (`dataset_name` is in case the desired dataset has a different name then the ones used in training, e.g. perform conditional expectation with one model on different dencities).
Then, `ConditionalExpectation.run()` performs Conditional Expectation. This also logs to weights and bias and saves the results.