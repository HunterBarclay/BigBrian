using System;
using BigBrian.v2;
using Newtonsoft.Json;
using System.IO;

namespace BigBrianTester {
    class Program {
        public static TestCase[] XOR_DATA = new TestCase[] {
                new TestCase { inputs = new double[] { 0.0, 0.0 }, outputs = new double[] { 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 0.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 0.0, 1.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 1.0, 1.0 }, outputs = new double[] { 0.0 } }
            };

        public static TestCase[] AND_DATA = new TestCase[] {
                new TestCase { inputs = new double[] { 0.0, 0.0 }, outputs = new double[] { 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 0.0 }, outputs = new double[] { 0.0 } },
                new TestCase { inputs = new double[] { 0.0, 1.0 }, outputs = new double[] { 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 1.0 }, outputs = new double[] { 1.0 } }
            };

        public static TestCase[] OR_DATA = new TestCase[] {
                new TestCase { inputs = new double[] { 0.0, 0.0 }, outputs = new double[] { 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 0.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 0.0, 1.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 1.0, 1.0 }, outputs = new double[] { 1.0 } }
            };

        public static TestCase[] NAND_DATA = new TestCase[] {
                new TestCase { inputs = new double[] { 0.0, 0.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 1.0, 0.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 0.0, 1.0 }, outputs = new double[] { 1.0 } },
                new TestCase { inputs = new double[] { 1.0, 1.0 }, outputs = new double[] { 0.0 } }
            };

        public static void Main2(string[] args) {
            Network n = new Network(JsonConvert.DeserializeObject<NetworkData>(File.ReadAllText("./NetworkData.json")));
            var fs = File.Create("Data.csv");
            var sw = new StreamWriter(fs);
            sw.WriteLine("input1,input2,output1");
            for (double x = 0; x < 1; x += 0.02) {
                for (double y = 0; y < 1; y += 0.02) {
                    sw.Write($"{x},{y}");
                    var output = n.Calculate(new double[] { x, y });
                    sw.Write($",{output[0]}\n");
                    sw.Flush();
                }
            }
            sw.Close();
            fs.Close();
        }

        public static void Main(string[] args) {

            if (args.Length > 0) {
                Network n = new Network(JsonConvert.DeserializeObject<NetworkData>(File.ReadAllText(args[0])));
                while (true) {
                    double[] input = new double[n.Structure[0]];
                    for (int i = 0; i < n.Structure[0]; ++i) {
                        Console.Write($"[{i}]: ");
                        input[i] = double.Parse(Console.ReadLine());
                    }
                    Console.WriteLine("\nResults\n");
                    var output = n.Calculate(input);
                    for (int i = 0; i < output.Length; ++i) {
                        // int result = output[i] > 0.5 ? 1 : 0;
                        Console.WriteLine($"[{i}] -> {output[i]}");
                    }
                    Console.WriteLine("\n");
                }
            } else {

                var omg = new OmegaTrainer(new int[] { 2, 2, 1 }, XOR_DATA);

                int iterations = 800000;
                for (int i = 0; i < iterations; ++i) {
                    int period = 100000;
                    omg.Test(i % period == 0, false);
                    if (i % period == 0) {
                        Console.WriteLine($"Progress [{(int)((double)i * 100.0 / (double)iterations)}%]");
                    }
                }
                Console.WriteLine($"\n Final Cost -> {omg.LastCost}");

                omg.Test(false, true);

                var data = omg.TargetNetwork.GetNetworkData();
                string json = JsonConvert.SerializeObject(data, Formatting.Indented);
                File.WriteAllText("./Networks/NetworkData.json", json);
            }
        }
    }
}
