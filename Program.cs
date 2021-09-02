using System;
using System.Collections.Generic;
using System.Threading;
using BigBrian.v2;

namespace BigBrian
{
    class Program
    {
        public static void Main(string[] args) {

            TestCase[] data = new TestCase[] {
                new TestCase { inputs = new double[] { 0.0, 0.0 }, outputs = new double[] { 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 0.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 0.0, 1.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 1.0, 1.0 }, outputs = new double[] { 0.0 } }
            };

            var omg = new OmegaTrainer(new int[] { 2, 2, 1 }, data);
            for (int i = 0; i < 50000; ++i) {
                omg.Test(i % 10000 == 0);
            }
            Console.WriteLine($"\n Final Cost -> {omg.LastCost}");
        }

        #region Old

        static NeuralNet n;
        public static bool Done = false;

        static int[] structure = new int[] { 2, 2, 3 };

        static (double[], double[])[] ORGateData = new (double[], double[])[] {
            (new double[]{ 0.0, 0.0 }, new double[]{ 0.0 }),
            (new double[]{ 0.0, 1.0 }, new double[]{ 1.0 }),
            (new double[]{ 1.0, 0.0 }, new double[]{ 1.0 }),
            (new double[]{ 1.0, 1.0 }, new double[]{ 1.0 })
        };
        static (double[], double[])[] NotGateData = new (double[], double[])[] {
            (new double[]{ 0.0 }, new double[]{ 1.0 }),
            (new double[]{ 1.0 }, new double[]{ 0.0 })
        };
        static (double[], double[])[] ANDGateData = new (double[], double[])[] {
            (new double[]{ 0.0, 0.0 }, new double[]{ 0.0, 0.0, 0.0 }),
            (new double[]{ 0.0, 1.0 }, new double[]{ 0.0, 0.0, 0.0 }),
            (new double[]{ 1.0, 0.0 }, new double[]{ 0.0, 0.0, 0.0 }),
            (new double[]{ 1.0, 1.0 }, new double[]{ 1.0, 1.0, 1.0 })
        };

        static void Main2(string[] args)
        {
            var rand = new Random();

            List<(double[], double[])> newData = new List<(double[], double[])>();
            for (int i = 0; i < 500; i++) {
                double x = (rand.NextDouble() * 50) - 25;
                double y = (rand.NextDouble() * 50) - 25;
                double output = (0.5 * x) + 2 > y ? 1.0 : 0.0;
                newData.Add((new double[]{x, y}, new double[]{output}));
            }

            int interations = 100000;
            var trainer = new BackPropTrainer(structure, ANDGateData);
            var progressTracker = new Thread(() => {
                try {
                    while (!Done) {
                        Thread.Sleep(1000);
                        double percent = (100 * ((double)trainer.Modifications / (double)interations));
                        if (!Done) {
                            // Console.Clear();
                            Console.WriteLine($"Progress => {percent.ToString().Substring(0, 5)}%");
                        }
                    }
                } catch (Exception e) { } // Whoop... fuck
            });
            progressTracker.Start();
            trainer.Train(interations, false);
            // Console.ReadKey();
        }

        public static List<double> MergeSortNetworks(List<double> a) => MergeSortNetworks(a.GetRange(0, a.Count / 2), a.GetRange(a.Count / 2, a.Count - (a.Count / 2)));

        public static List<double> MergeSortNetworks(List<double> a, List<double> b) {
            if (a.Count > 1)
                a = MergeSortNetworks(a.GetRange(0, a.Count / 2), a.GetRange(a.Count / 2, a.Count - (a.Count / 2)));
            if (b.Count > 1)
                b = MergeSortNetworks(b.GetRange(0, b.Count / 2), b.GetRange(b.Count / 2, b.Count - (b.Count / 2)));

            List<double> sortedList = new List<double>();
            int counterA = 0;
            int counterB = 0;

            while (sortedList.Count < a.Count + b.Count) {

                if (counterA >= a.Count) {
                    sortedList.Add(b[counterB]);
                    counterB++;
                } else if (counterB >= b.Count) {
                    sortedList.Add(a[counterA]);
                    counterA++;
                } else {
                    if (a[counterA] > b[counterB]) {
                        sortedList.Add(b[counterB]);
                        counterB++;
                    } else {
                        sortedList.Add(a[counterA]);
                        counterA++;
                    }
                }
            }

            return sortedList;
        }

        #endregion
    }
}
