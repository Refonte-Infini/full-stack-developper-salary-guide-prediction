import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;

public class CombinedModels {
    // 1. Inflation Adjustment Model
    public static double inflationAdjustedSalary(double nominalSalary, double inflationRate) {
        return nominalSalary * (1 + inflationRate / 100);
    }

    // 2. Compound Annual Growth Rate (CAGR)
    public static double calculateCAGR(double fv, double pv, int years) {
        return Math.pow(fv / pv, 1.0 / years) - 1;
    }

    // 3. Neural Network Regression (DL4J Example)
    public static void neuralNetworkRegression() {
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(1).nOut(5).build())
                .layer(new OutputLayer.Builder().nIn(5).nOut(1).build())
                .build());
        model.init();

        INDArray input = Nd4j.create(new double[][] {{1}, {2}, {3}, {4}});
        INDArray output = Nd4j.create(new double[][] {{70}, {85}, {100}, {130}});
        model.fit(new DataSet(input, output));

        INDArray prediction = model.output(Nd4j.create(new double[][] {{5}}));
        System.out.println("Neural Network Prediction for 5 Years: $" + prediction);
    }

    public static void main(String[] args) {
        System.out.println("Inflation Adjusted Salary: $" + inflationAdjustedSalary(90000, 2.5));
        System.out.printf("CAGR: %.2f%%\n", calculateCAGR(120000, 90000, 3) * 100);
        neuralNetworkRegression();
        // Bayesian Linear Regression and GPT implementation requires external specialized libraries
        System.out.println("Bayesian Regression and GPT are not natively supported in Java.");
    }
}
