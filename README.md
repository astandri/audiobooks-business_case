# audiobooks-business_case
My Capstone Project for Udemy Machine Learning with Tensorflow Course

## Business Case:
Given audiobook database consist of customer data with each customer in database has made a purchase at least once.
We will create a machine learning algorithm that will predict the customer is likely to buy again or not. So, if the company focused on customer which is likely to buy again, they could increase the sales.

## Dataset:
the dataset consist of following columns:

  ID, Book_length(mins)_overall, Book_length(mins)_avg, price_overall, price_avg, 
  Review, Review 10/10, Minutes listened, Completion, Support Requests, Last visited minus purchase date, Targets/Labels
  
  The Targets/Labels show the customer conversion, 0 if not, 1 if the customer buy again (2 classes)

## What's covered in this project
This project covers some activities, they are:
  1. Data Preprocessing
  
      a. Balance the dataset (for better learning process, the priors has been set to approximately 50:50 for 2 classes)
      b. Divide the dataset to 3 parts: Training, Validation, Testing
      c. save the data to tensor friendly format (.npz)
	  
  2. Batching
  
      a. A class created to handle batching so the model can train using dataset batches
	  b. This class also can be used as a template for other machine learning projects
	  
  3. Machine learning algorithm (model)
  
The testing result concluded on 83% accuracy, impressive result
