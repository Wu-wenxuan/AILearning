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
public class MLPMnistSingleLayerExample {

	private static Logger log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.class);

	public static void main(String[] args) throws Exception {
		final int numRows = 28; // 矩阵的行数。
		final int numColumns = 28; // 矩阵的列数。
		int outputNum = 10; // 潜在结果（比如0到9的整数标签）的数量。
		int batchSize = 128; // 每一步抓取的样例数量。
		int rngSeed = 123; // 这个随机数生成器用一个随机种子来确保训练时使用的初始权重维持一致。
		int numEpochs = 15; // 一个epoch指将给定数据集全部处理一遍的周期。

		// 抓取Mnist 数据集中的数据
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

		log.info("Build model....");
		/**
		 * .seed(rngSeed) 该参数将一组随机生成的权重确定为初始权重。如果一个示例运行很多次，而每次开始时都生成一组新的随机权重，
		 * 那么神经网络的表现（准确率和F1值）有可能会出现很大的差异，因为不同的初始权重可能会将算法导向误差曲面上不同的局部极小值，
		 * 在其他条件不变的情况下，保持相同的随机权重可以使调整其他超参数所产生的效果表现得更加清晰。
		 * 
		 * 
		 * .optimizationAlgo (OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		 * 随机梯度下降（Stochastic Gradient
		 * Descent，SGD）是一种用于优化代价函数的常见方法。要了解SGD和其他帮助实现误差最小化的优化算法，可参考Andrew
		 * Ng的机器学习课程以及本网站术语表中对SGD的定义。
		 * 
		 * 
		 * .iterations(1) 对一个神经网络而言，一次迭代（iteration）指的是一个学习步骤，亦即模型权重的一次更新。
		 * 神经网络读取数据并对其进行预测，然后根据预测的错误程度来修正自己的参数。因此迭代次数越多，网络的学习步骤和学习量也越多，让误差更接近极小值。
		 * 
		 * 
		 * .learningRate(0.006) 本行用于设定学习速率（learning
		 * rate），即每次迭代时对于权重的调整幅度，亦称步幅。学习速率越高，神经网络“翻越”整个误差曲面的速度就越快，但也更容易错过误差极小点。
		 * 学习速率较低时，网络更有可能找到极小值，但速度会变得非常慢，因为每次权重调整的幅度都比较小。
		 * 
		 * 
		 * .updater(Updater.NESTEROVS).momentum(0.9)
		 * 动量（momentum）是另一项决定优化算法向最优值收敛的速度的因素。动量影响权重调整的方向，所以在代码中，我们将其视为一种权重的更新器（updater）。
		 * 
		 * 
		 * .regularization(true).l2(1e-4)
		 * 正则化（regularization）是用来防止过拟合的一种方法。过拟合是指模型对训练数据的拟合非常好，
		 * 然而一旦在实际应用中遇到从未出现过的数据，运行效果就变得很不理想。 我们用L2正则化来防止个别权重对总体结果产生过大的影响。
		 * 
		 * 
		 * .list() 函数可指定网络中层的数量；它会将您的配置复制n次，建立分层的网络结构。 再次提醒：如果对以上任何内容感到困惑，建议您参考Andrew
		 * Ng的机器学习课程。
		 * 
		 * 
		 */
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed)
				.updater(new Nesterovs(0.006, 0.9)).l2(1e-4).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numRows * numColumns).nOut(1000).activation(Activation.RELU)
								.weightInit(WeightInit.XAVIER).build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nIn(1000).nOut(outputNum)
						.activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		// print the score with every 1 iteration
		model.setListeners(new ScoreIterationListener(1));

		log.info("Train model....");
		for (int i = 0; i < numEpochs; i++) {
			model.fit(mnistTrain);
		}

		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum);
		// create an evaluation object with 10 possible classes
		while (mnistTest.hasNext()) {
			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatures());
			// get the networks prediction
			eval.eval(next.getLabels(), output);
			// check the prediction against the true class
		}

		log.info(eval.stats());
		log.info("****************Example finished********************");

	}

}
