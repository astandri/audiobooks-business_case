import numpy as np
import tensorflow as tf

class Model():
	def __init__(self, train_data, validation_data):
		self.type = 'model'
		self.train_data = train_data
		self.validation_data = validation_data
	
	def train(self):		
		#MODEL CREATION (depth 2, width 50)
		input_size = 10
		output_size = 2
		hidden_layer_size = 50
		
		tf.reset_default_graph()
		
		##Setting up the placeholders
		self.inputs = tf.placeholder(tf.float32, [None, input_size])
		self.targets = tf.placeholder(tf.int32, [None, output_size])
		
		#setting up 1st weights and biases in hidden layers
		w_1 = tf.get_variable('w_1', [input_size,hidden_layer_size])
		b_1 = tf.get_variable('b_1', [hidden_layer_size])
		
		o_1 = tf.nn.relu(tf.matmul(self.inputs,w_1) + b_1)
		
		#setting up 2nd weights and biases in hidden layers
		w_2 = tf.get_variable('w_2', [hidden_layer_size,hidden_layer_size])
		b_2 = tf.get_variable('b_2', [hidden_layer_size])
		
		o_2 = tf.nn.relu(tf.matmul(o_1,w_2) + b_2)
		
		#setting up output layer
		w_3 = tf.get_variable('w_3', [hidden_layer_size,output_size])
		b_3 = tf.get_variable('b_3', [output_size])
		
		outputs = tf.matmul(o_2,w_3) + b_3
		
		#loss function and optimizer
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=self.targets)
		mean_loss = tf.reduce_mean(loss)
		
		optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)
		
		out_equals_target = tf.equal(tf.argmax(outputs,1), tf.argmax(self.targets,1))
		self.accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))	
		
		#initiate tensorflow session
		self.sess = tf.InteractiveSession()
		self.initializer = tf.global_variables_initializer()
		self.sess.run(self.initializer)
				
		max_epoch = 50
		
		prev_validation_loss = 9999999.
		
		#Epoch loops
		for epoch_counter in range(max_epoch):
			curr_epoch_loss = 0
			
			#Training
			for input_batch, target_batch in self.train_data:
				_, batch_loss = self.sess.run([optimize,mean_loss], feed_dict={self.inputs: input_batch, self.targets: target_batch})
			
				curr_epoch_loss += batch_loss
			
			curr_epoch_loss /= self.train_data.batch_count
		
			#Validation
			validation_loss = 0.
			validation_accuracy = 0.
			
			for input_batch, target_batch in self.validation_data:
				validation_loss, validation_accuracy = self.sess.run([mean_loss,self.accuracy], 
				feed_dict={self.inputs: input_batch, self.targets: target_batch})

			print('Epoch'+str(epoch_counter+1)+
				'. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
				'. Validation loss: '+'{0:.3f}'.format(validation_loss)+
				'. Validation Accuracy: '+'{0:.2f}'.format(validation_accuracy*100.)+'%')
				
			if validation_loss > prev_validation_loss:
				break
			
			prev_validation_loss = validation_loss
		
		print('End of Training')
		print('==========================')
		
	def test(self, test_data):
		for input_batch, target_batch in test_data:
			test_accuracy = self.sess.run([self.accuracy], feed_dict={self.inputs: input_batch, self.targets: target_batch})
			
		test_accuracy_percentage = test_accuracy[0]*100.

		print('Test accuracy:'+'{0:.2f}'.format(test_accuracy_percentage)+'%')