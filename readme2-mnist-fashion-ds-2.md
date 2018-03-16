keras-mnist-fashion-ds-2.py
program uses the MNIST fashion datasetto sort through thousands of images of clothing types (jackets, shirts, pants, dresses etc.)
and accessories (handbags, shoes etc.) to classify the images into their proper categories; 
the program also saves the model and then re-uses it. 
I also use a key-value pair process to map the clothing code to the description and loop through samples of the testing data
to pull and view some of the images and the predictions for those images. 
1. import the libraries and modules - tools required to run the program.
2. load the mnist fashion dataset into x_train, y_train, x_test and y_test sets - divides the data into training (used by program to learn) and test (program tests its predictions against test data and gets better at learning for optimal results based on defined parameters).
3. preprocess the data and class tables - turns the datasets into more standardized, program readable sets.
4. Define the model architecture - tell the program which parameters to use and methods to use for learning; define the input shape.
5. Compile the model - configure the learning process for the program.
6.  Fit the model to data - run the model on the data.
7. *Save the model for future use - (I saved the model on first run and re-used it on subsequent runs, commenting out the save step).
8.  key-value pair -  map the label code to a description that people can make sense of (e.g., 4 = Coat, 8 = Bag).
9.  added a range - pulled out sample images from the test data to view the image and the program's prediction of what the image is. 
