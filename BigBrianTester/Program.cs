using System;
using BigBrian.v2;
using Newtonsoft.Json;
using System.IO;
using System.Linq;
using System.Collections.Generic;

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

        public static TestCase[] HALF_ADDER_DATA = new TestCase[] {
                new TestCase { inputs = new double[] { 0.0, 0.0 }, outputs = new double[] { 0.0, 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 0.0 }, outputs = new double[] { 1.0, 0.0 } },
                new TestCase { inputs = new double[] { 0.0, 1.0 }, outputs = new double[] { 1.0, 0.0 } },
                new TestCase { inputs = new double[] { 1.0, 1.0 }, outputs = new double[] { 0.0, 1.0 } }
            };

        public static TestCase[] FULL_ADDER_DATA = new TestCase[] {
            new TestCase { inputs = new double[] { 0.0, 0.0, 0.0 }, outputs = new double[] { 0.0, 0.0 } },
            new TestCase { inputs = new double[] { 1.0, 0.0, 0.0 }, outputs = new double[] { 1.0, 0.0 } },
            new TestCase { inputs = new double[] { 0.0, 1.0, 0.0 }, outputs = new double[] { 1.0, 0.0 } },
            new TestCase { inputs = new double[] { 1.0, 1.0, 0.0 }, outputs = new double[] { 0.0, 1.0 } },
            new TestCase { inputs = new double[] { 0.0, 0.0, 1.0 }, outputs = new double[] { 1.0, 0.0 } },
            new TestCase { inputs = new double[] { 1.0, 0.0, 1.0 }, outputs = new double[] { 0.0, 1.0 } },
            new TestCase { inputs = new double[] { 0.0, 1.0, 1.0 }, outputs = new double[] { 0.0, 1.0 } },
            new TestCase { inputs = new double[] { 1.0, 1.0, 1.0 }, outputs = new double[] { 1.0, 1.0 } },
            };

        public static void Main(string[] args) {
            switch (args.Length > 0 ? args[0].ToLower() : "") {
                case "eval":
                    Evaluate(args.Skip(1).ToArray());
                    break;
                default:
                    Train();
                    break;
            }
        }

        [Obsolete]
        public static void Evaluate(string[] args) {
            Network n = new Network(JsonConvert.DeserializeObject<NetworkData>(File.ReadAllText("./Networks/NetworkData.json")));
            var fs = File.Create("Data.csv");
            var sw = new StreamWriter(fs);

            for (int i = 0; i < n.Structure[0]; ++i) {
                if (i > 0)
                    sw.Write(",");
                sw.Write($"input{i}");
            }
            for (int i = 0; i < n.Structure[n.Structure.Length - 1]; ++i) {
                sw.Write($",output{i}");
            }
            sw.Write("\n");

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

        public static void Train() {

            var EIGHT_BIT_ADDER_DATA = GenerateAdderData();

            var omg = new OmegaTrainer(new int[] { 8, 12, 8, 5, 5, 5, 4 }, EIGHT_BIT_ADDER_DATA);

            int iterations = 8000;
            int i;
            for (i = 0; i < iterations; ++i) {
                int period = 1;
                var done = omg.Test(0.01, i % period == 0, i % period == 50);
                if (i % period == 0) {
                    Console.WriteLine($"Progress [{(int)((double)i * 100.0 / (double)iterations)}%]");
                }
                if (done)
                    break;
            }
            if (i < iterations) {
                Console.WriteLine($"\n Finished after {i + 1} Iterations");
            }
            Console.WriteLine($"\n Final Cost -> {omg.LastCost}");

            omg.Test(0, false, true);

            var data = omg.TargetNetwork.GetNetworkData();
            string json = JsonConvert.SerializeObject(data, Formatting.Indented);
            File.WriteAllText("./Networks/NetworkData.json", json);
        }

        public static TestCase[] GenerateAdderData() {
            List<TestCase> cases = new List<TestCase>();
            for (byte a = 0; a < 16; a++) {
                for (byte b = 0; b < 16; b++) {
                    byte c = (byte)(a + b);
                    var input = new List<double>(ByteToData(a));
                    input = new List<double>(input.Skip(4));
                    input.AddRange(ByteToData(b).Skip(4));
                    var output = ByteToData(c).Skip(4);
                    cases.Add(new TestCase { inputs = input.ToArray(), outputs = output.ToArray() });
                }
            }
            return cases.ToArray();
        }

        private static double[] ByteToData(byte a) {
            double[] data = new double[8];
            for (int i = 7; i >= 0; --i) {
                data[7 - i] = (a >> i) & 0x0001;
            }
            return data;
        }
    }
}
