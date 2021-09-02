using System;

namespace BigBrian {
    public class BackPropTrainer {

        private (double[] input, double[] output)[] testData;
        private NeuralNet billy;

        public const double CostThreshold = 0.02;

        public int Calculations { get; private set; }
        public int Modifications { get; private set; }

        private CSVFile deltaLog, valueLog;

        private double BestTotalCost = double.MaxValue;

        public BackPropTrainer(int[] structure, (double[] input, double[] output)[] testData) {
            this.testData = testData;
            billy = new NeuralNet(structure, 7654383 * DateTime.Now.Millisecond);

            string header = "calc";

            for (int i = 0; i < billy.layers.Length; i++) {

                for (int x = 0; x < billy.layers[i].Size; x++) {
                    header += $",b{i+1}_{x}";
                }
                
                for (int x = 0; x < billy.layers[i].Size; x++) {
                    for (int y = 0; y < billy.layers[i].weights.Length; y++) {
                        header += $",w{i+1}_{x*y}";
                    }
                }
            }
            header += ",cost";

            deltaLog = new CSVFile("delta", header);
            valueLog = new CSVFile("value", header);
        }

        public double TestDataSet(bool verbose) {
            double dataSetCost = 0.0;
            foreach (var dataPoint in testData) {
                var output = billy.Passthrough(dataPoint.input);
                Calculations++;
                billy.CalculateCost(output, dataPoint.output);

                if (verbose) {
                    string result = $"Calculation {Calculations}: [";
                    foreach (var x in dataPoint.input) {
                        result += $"{x}, ";
                    }
                    result = result.Substring(0, result.Length - 2);
                    result += "] => [";
                    foreach (var x in dataPoint.output) {
                        result += $"{x}, ";
                    }
                    result = result.Substring(0, result.Length - 2);
                    result += "] [";
                    foreach (var x in output) {
                        result += $"{x}, ";
                    }
                    result = result.Substring(0, result.Length - 2);
                    result += "]";
                    Console.WriteLine(result);
                }
                billy.CalculateDeltas(output, dataPoint.output);
                dataSetCost += billy.Cost;

                if (Calculations % 100 == 0) {

                    string deltaData = $"{Calculations}";
                    string valueData = $"{Calculations}";
                    for (int i = 0; i < billy.layers.Length; i++) {

                        for (int x = 0; x < billy.layers[i].Size; x++) {
                            deltaData += $",{billy.layers[i].biasDeltas[x]}";
                            valueData += $",{billy.layers[i].biases[x]}";
                        }
                        
                        for (int x = 0; x < billy.layers[i].Size; x++) {
                            for (int y = 0; y < billy.layers[i].weights.Length; y++) {
                                deltaData += $",{billy.layers[i].weightDeltas[y][x]}";
                                valueData += $",{billy.layers[i].weights[y][x]}";
                            }
                        }
                    }
                    deltaLog.Log(deltaData + $",{billy.Cost}");
                    valueLog.Log(valueData + $",{billy.Cost}");

                }

                if (Calculations % 16 == 0) {
                    int a = 0;
                }
            }
            if (dataSetCost < BestTotalCost)
                BestTotalCost = dataSetCost;
            return dataSetCost;
        }

        public void Train(int iterations, bool verbose) {
            double c = 0;
            for (int i = 0; i < iterations; i++) {

                c = TestDataSet(verbose);

                if (c <= CostThreshold)
                    break;

                // Apply deltas
                billy.ApplyDeltas(true);
                Modifications++;
                if (verbose)
                    Console.WriteLine($"Modifications {Modifications}");

            }
            Program.Done = true;
            Console.WriteLine("=====End Result=====\n");
            TestDataSet(true);
            Console.WriteLine($"Modifications {Modifications}");
            Console.WriteLine($"Best Cost: {BestTotalCost}");
            Console.WriteLine($"Current Cost: {c}");
        }

    }
}