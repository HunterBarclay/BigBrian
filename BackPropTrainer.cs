using System;

namespace BigBrian {
    public class BackPropTrainer {

        private (double[] input, double[] output)[] testData;
        private NeuralNet billy;

        public int Calculations { get; private set; }
        public int Modifications { get; private set; }

        private CSVFile log;

        public BackPropTrainer(int[] structure, (double[] input, double[] output)[] testData) {
            this.testData = testData;
            billy = new NeuralNet(structure, 7654383 * DateTime.Now.Millisecond);

            log = new CSVFile("in1,in2,out1,targ1,cost");
        }

        public void TestDataSet(bool verbose) {
            avgCost = 0.0;
            foreach (var dataPoint in testData) {
                var output = billy.Passthrough(dataPoint.input);
                Calculations++;
                billy.CalculateCost(output, dataPoint.output);
                log.Log($"{dataPoint.input[0]},{dataPoint.input[1]},{output[0]},{dataPoint.output[0]},{billy.Cost}");
                if (verbose)
                    Console.WriteLine($"Calculation {Calculations}: [Cost] {billy.Cost}");
                billy.CalculateDeltas(output, dataPoint.output);
                avgCost += billy.Cost;

                if (Calculations % 16 == 0) {
                    int a = 0;
                }
            }
            avgCost /= testData.Length;
        }

        double avgCost = 0.0;
        double deltaAvgCost = 0.0;

        public void Train(int iterations, bool verbose) {

            double lastAvgCost = avgCost;
            double lastDeltaAvgCost = deltaAvgCost;

            for (int i = 0; i < iterations; i++) {

                TestDataSet(verbose);
                deltaAvgCost = Math.Abs(avgCost - lastAvgCost);

                if (i > 1) {
                    if (deltaAvgCost > lastDeltaAvgCost) {
                        NeuralNet.TrimMod *= -0.999;
                    } else {
                        NeuralNet.TrimMod *= 0.999;
                    }
                }
                
                lastDeltaAvgCost = deltaAvgCost;
                lastAvgCost = avgCost;

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
        }

    }
}