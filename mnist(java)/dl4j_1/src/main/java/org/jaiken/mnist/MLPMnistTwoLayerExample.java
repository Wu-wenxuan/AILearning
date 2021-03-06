package org.jaiken.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("deprecation")
public class MLPMnistTwoLayerExample {

	private static Logger log = LoggerFactory.getLogger(MLPMnistTwoLayerExample.class);

	public static void main(String[] args) throws Exception {
		// number of rows and columns in the input pictures
		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10; // number of output classes
		int batchSize = 64; // batch size for each epoch
		int rngSeed = 123; // random number seed for reproducibility
		int numEpochs = 15; // number of epochs to perform
		double rate = 0.0015; // learning rate

		// Get the DataSetIterators:
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

		log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) // include a random seed for													// reproducibility
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER).updater(new Nesterovs(rate, 0.98))
				.l2(rate * 0.005) // regularize learning model
				.list()
				.layer(0, new DenseLayer.Builder() // create the first input layer.
						.nIn(numRows * numColumns).nOut(500).build())
				.layer(1, new DenseLayer.Builder() // create the second input layer
						.nIn(500).nOut(100).build())
				.layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
						.activation(Activation.SOFTMAX).nIn(100).nOut(outputNum).build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(5)); // print the score with every iteration

		log.info("Train model....");
		for (int i = 0; i < numEpochs; i++) {
			log.info("Epoch " + i);
			model.fit(mnistTrain);
		}

		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum); // create an evaluation object with 10 possible classes
		while (mnistTest.hasNext()) {
			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatures()); // get the networks prediction
			eval.eval(next.getLabels(), output); // check the prediction against the true class
		}

		log.info(eval.stats());
		log.info("****************Example finished********************");

	}

}
