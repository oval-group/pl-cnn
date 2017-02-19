import os
import cPickle as pickle

my_dir = "./experiments/paper/cifar10/bp/"
# my_dir = "./experiments/paper/temp/"
for filename in os.listdir(my_dir):
	if filename.endswith(".p"):
		print(filename)

		obj = pickle.load(open(my_dir + filename, "rb"))
		
		print("C value: \t\t %g" % obj.dataset.C)
		print("Solver: \t\t %s" % obj.dataset.solver)
		print("Learning rate: \t\t %g\n" % obj.dataset.learning_rate)

		if "svm" in obj.dataset.solver:
			continue

		print("Training obj: \t\t %g" % obj.train_err[-1])
		print("Training acc: \t\t %.2f%%\n" % obj.train_acc[-1])

		print("Validation obj: \t %g" % obj.val_err[-1])
		print("Validation acc: \t %.2f%%\n" % obj.val_acc[-1])

		print("Testing obj: \t\t %g" % obj.test_err)
		print("Testing acc: \t\t %.2f%%\n" % obj.test_acc)

		print("Train time: \t\t %g" % obj.train_time[-1])
		print("Number of epochs: \t %i" % len(obj.val_acc))

		# print("Comment:\n%s\n" % obj.comment)

		print("-"*80)