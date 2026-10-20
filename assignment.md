The goal of this assignment is to mimic being an AI engineer who has inherited a bad codebase from a colleague and has been tasked with improving both the code and model performance.

Note: There is a data leak (and you will need to fix it) AND a really, really, really significant bug (just FYI). Do not trust the work you read. If you are naive about this, you will fail. Must hit at least 93% test accuracy.

You must do the following:
1. You must find the bugs in the code and fix them. There are two bugs at least. Update any code and any things that do not match what we are trying to do. Update code to be well documented with docstrings. Make sure all the code is setup for the UCF50 dataset. 

2. We need to create a custom dataloader for the dataset UCF50 that is in the `data/` folder. We also need to update/create/fix any managemnt helpers for this as needed.

2. Introduce weights and bias logging to the base (and submit evidence of performance on weights and biases for loss/accuracy for train/test/val)

3. Submit an updated README for the base that describes the errors found / cleaned. 

4. Make at least 2 model-level improvements to the base using techniques discussed in class (you can use LLM / AI tools to help you figure out what improvements to make, but you must be able to discuss them within your readme).

5. Lint all code using pylint.

6. Update the README.md to be a one-stop guide for running the project and getting a reproducible setup and making it easy to understand what the project is doing. Document the general structure of the codebase and give a run down of what all the pieces do and how to use them. Explain where things can be tweaked or modified to changes the training etc. 

7. Add unit tests for key functionality using pytest. You only need to create a handful for each piece of **key** functionality. DO NOT create unit tests for everything. 

8. Create a commit or commits in the git repo we have setup.


**REMEMBER** You will not be able to hit 93% without (1) improving the code and (2) fixing the bugs.
