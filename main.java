package Projekt.Zagadnienie3;

import java.util.Arrays;

import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.SupervisedHebbianNetwork;
import org.neuroph.nnet.UnsupervisedHebbianNetwork;
import org.neuroph.nnet.learning.OjaLearning;
import org.neuroph.nnet.learning.SupervisedHebbianLearning;
import org.neuroph.nnet.learning.UnsupervisedHebbianLearning;

import Projekt.Data.Data;

public class Zadanie3 {

	private SupervisedHebbianLearning supervisedHebian;
	private UnsupervisedHebbianLearning unsupervisedHebian;
	private OjaLearning oja;

	private SupervisedHebbianNetwork supervisedHebianNetwork;
	private UnsupervisedHebbianNetwork unsupervisedHebianNetwork;

	long start = 0;
	long stop = 0;
	double executionTime = 0.0;

	public Zadanie3(Data data) {
		System.out.println("SUPERVISED HEBBIAN");
		System.out.println();
		initSupervisedHebbian(data);
		validateSupervisedHebbian(data);

		System.out.println("UNSUPERVISED HEBBIAN");
		System.out.println();
		initUnsupervisedHebbian(data);
		validateUnsupervisedHebbian(data);
		
		System.out.println("OJA");
		System.out.println();
		initOja(data);
		validateOja(data);
		

	}

	private void initOja(Data data) {
		unsupervisedHebianNetwork = new UnsupervisedHebbianNetwork(35, 1);

		oja = new OjaLearning();
		oja.setLearningRate(0.5);
		oja.setMaxIterations(100);
		//oja.isPausedLearning();
		/*oja.addListener(new LearningEventListener() {

			@Override
			public void handleLearningEvent(LearningEvent event) {
				UnsupervisedHebbianLearning oja = (UnsupervisedHebbianLearning) event.getSource();
				System.out.println("Epoka: " + oja.getCurrentIteration());
				System.out.println(oja.);
			}
		});*/
		
		unsupervisedHebianNetwork.setLearningRule(oja);
		
		/*int epoch = 1;
		do
		{
		  oja.doOneLearningIteration(data.getTrainingSetForUnsupervised());
		  System.out.println("Epoch " + oja.getCurrentIteration());
		  //oja.doOneLearningIteration(data.getTrainingSet());
		  epoch++;
		  } while(oja.getCurrentIteration() > 100);
		*/
		
	
		start = System.currentTimeMillis();
		unsupervisedHebianNetwork.learn(data.getTrainingSetForUnsupervised());
		stop = System.currentTimeMillis();

		executionTime = stop - start;

	}

	private void validateOja(Data data) {
		System.out.println();
		for (DataSetRow dataRow : data.getValidatingSet().getRows()) {

			unsupervisedHebianNetwork.setInput(dataRow.getInput());
			unsupervisedHebianNetwork.calculate();

			double[] networkOutput = unsupervisedHebianNetwork.getOutput();
			System.out.println("Input: ");
			data.printMatrix(dataRow.getInput());
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
		System.out.println("Execution time: " + executionTime + " ms");
		System.out.println();
		
	}

	private void validateUnsupervisedHebbian(Data data) {
		System.out.println();
		for (DataSetRow dataRow : data.getTrainingSetForUnsupervised().getRows()) {

			unsupervisedHebianNetwork.setInput(dataRow.getInput());
			unsupervisedHebianNetwork.calculate();

			double[] networkOutput = unsupervisedHebianNetwork.getOutput();
			System.out.println("Input: ");
			data.printMatrix(dataRow.getInput());
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
		System.out.println("Execution time: " + executionTime + " ms");
		System.out.println();

	}

	private void initUnsupervisedHebbian(Data data) {
		unsupervisedHebianNetwork = new UnsupervisedHebbianNetwork(35, 1);

		unsupervisedHebian = new UnsupervisedHebbianLearning();
	/*	unsupervisedHebian.addListener(new LearningEventListener() {

			@Override
			public void handleLearningEvent(LearningEvent event) {
				UnsupervisedHebbianLearning unsupervisedHebbian = (UnsupervisedHebbianLearning) event.getSource();
				System.out.println("Epoka: " + unsupervisedHebbian.getCurrentIteration());
			}
		});*/

		//unsupervisedHebianNetwork.setLearningRule(unsupervisedHebian);

		start = System.currentTimeMillis();
		//for(int i=0;i<100;i++)
			unsupervisedHebianNetwork.learn(data.getTrainingSetForUnsupervised());
		stop = System.currentTimeMillis();

		executionTime = stop - start;

	}

	private void validateSupervisedHebbian(Data data) {
		System.out.println();
		for (DataSetRow dataRow : data.getValidatingSet().getRows()) {

			supervisedHebianNetwork.setInput(dataRow.getInput());
			supervisedHebianNetwork.calculate();

			double[] networkOutput = supervisedHebianNetwork.getOutput();
			System.out.println("Input: ");
			data.printMatrix(dataRow.getInput());
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
		System.out.println("Execution time: " + executionTime + " ms");
		System.out.println();

	}

	private void initSupervisedHebbian(Data data) {
		supervisedHebianNetwork = new SupervisedHebbianNetwork(35, 1);

		supervisedHebian = new SupervisedHebbianLearning();
		supervisedHebian.addListener(new LearningEventListener() {

			@Override
			public void handleLearningEvent(LearningEvent event) {
				SupervisedHebbianLearning supervisedHebbian = (SupervisedHebbianLearning) event.getSource();
				System.out.println("Epoka: " + supervisedHebbian.getCurrentIteration() + " MSE: " + supervisedHebbian.getTotalNetworkError());

			}
		});

		supervisedHebianNetwork.setLearningRule(supervisedHebian);

		start = System.currentTimeMillis();
		supervisedHebianNetwork.learn(data.getTrainingSet());
		stop = System.currentTimeMillis();

		executionTime = stop - start;
	}

}
