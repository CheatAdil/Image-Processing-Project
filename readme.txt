Greetings!

The main file is the main.py, in this file there are two lines

if __name__ == "__main__":
    demonstration(sshow_images=True, train_model_again=0)

If you want to change some configs - this is the place to do so

there are two variables you can change

1 sshow_images
it HAS to have two 's' in the name, otherwise it will break
if it is set to True then the program will show you every image it has to work with.
it will show the original, the compressed and the uncompressed versions of each and every image
if you want to run our program multiple times in a row to see if it breaks - viewing every image may be a big slowdown for you
In that case just set it to False and it will not show you the images.
Keep in mind that setting this to True is the only way to see the result of programm's work for
it doesnt store images on your pickle

2 train_model_again

if its set to 0 then it will use the results of our training of the train_model_again
if it is set to 1 then it will train again but with the same parameters as when we trained it
if it is set to 2 then it will do hyperparameter validation and THEN train

keep in mind that right now the values given to program with train_model_again=1 and train_model_again=2
are the same as the ones we trained it with
you can replace them with your own values

for hyperparameter picking you need to enter your values to main.py at lines 122 and 123
for just training with set on parameters you need to enter them on lines 128 and 129


The code will also dump one compressed image as a pickle dump of a pytorch tensor. 
It runs this code for every image so the one you will have in files
should be the last image of the 19 available