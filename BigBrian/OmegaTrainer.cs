using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace BigBrian.v2 {

    public static class Rando {
        public static Random rand;

        static Rando() {
            rand = new Random(DateTime.Now.Millisecond);
        }

        public static double NextDouble() => rand.NextDouble();
    }

    public class OmegaTrainer {

        public const double DELTA_ADJUSTMENT = 1;
        public TestCase[] Data;
        public Network TargetNetwork;

        public double LastCost = double.NaN;

        public List<(double[][][] weightDelta, double[] biasDelta)> gradients = new List<(double[][][] weightDelta, double[] biasDelta)>();

        public OmegaTrainer(int[] structure, TestCase[] data) {
            TargetNetwork = new Network(structure);
            Data = data;
        }

        public bool Test(double costThreshold, bool printImprovement, bool printResults) {
            double cost = 0;
            
            var lastTime = DateTime.Now;

            List<TimeSpan> intervals = new List<TimeSpan>();

            int count = 1;
            foreach (var d in Data) {
                var start = DateTime.Now;
                if ((DateTime.Now - lastTime).Seconds > 10) {
                    TimeSpan timePerTest = new TimeSpan();
                    for (int i = intervals.Count - 1; i >= 0 && i >= intervals.Count - 10; i--) {
                        timePerTest += intervals[i];
                    }
                    timePerTest /= 10;
                    lastTime = DateTime.Now;
                    Console.WriteLine($"Starting Test {count} of {Data.Length}");
                    Console.WriteLine($"Time until Gradient Application: {(timePerTest * (Data.Length - count)).ToString()}");
                }
                string o = $"    IN: [ {d.inputs[0]}";
                foreach (var a in d.inputs.Skip(1)) { o += $", {a}"; }
                o += $" ]\n    OUT: [ {d.outputs[0]}";
                foreach (var a in d.outputs.Skip(1)) { o += $", {a}"; }
                var actual = Calculate(d.inputs, d.outputs);
                o += $" ]\n    ACTUAL: [ {actual[0]}";
                foreach (var a in actual.Skip(1)) { o += $", {a}"; }
                o += $" ]";
                for (int i = 0; i < d.outputs.Length; i++) {
                    cost += Math.Pow(d.outputs[i] - actual[i], 2);
                }
                if (printResults)
                    Console.WriteLine(o);
                count++;
                intervals.Add(DateTime.Now - start);
            }
            if (printImprovement) {
                Console.WriteLine($" Improvement -> {LastCost - cost}\n Cost -> {cost}");
                LastCost = cost;
            }

            ApplyGradient();

            if (cost <= costThreshold) {
                LastCost = cost;
                return true;
            }
            return false;
        }

        public double[] Calculate(double[] inputs, double[] outputs) {
            var results = TargetNetwork.Calculate(inputs);

            var gradient = CalculateGradient(outputs, results);
            //for (int x = 0; x < gradient.weightDelta.Length; ++x) {
            //    for (int y = 0; y < gradient.weightDelta[x].Length; ++y) {
            //        for (int z = 0; z < gradient.weightDelta[x][y].Length; ++z) {
            //            Console.WriteLine($"Weight [{x}][{y}][{z}] -> {gradient.weightDelta[x][y][z]}");
            //        }
            //    }
            //    Console.WriteLine($"Bias [{x}] -> {gradient.biasDelta[x]}");
            //}
            gradients.Add(gradient);
            // ApplyGradient(gradient);

            return results;
        }

        public void ApplyGradient() {

            (double[][][] weightDelta, double[] biasDelta) gradient = gradients[0];

            for (int i = 1; i < gradients.Count; ++i) {
                for (int x = 0; x < gradient.weightDelta.Length; ++x) {
                    for (int y = 0; y < gradient.weightDelta[x].Length; ++y) {
                        for (int z = 0; z < gradient.weightDelta[x][y].Length; ++z) {
                            gradient.weightDelta[x][y][z] += gradients[i].weightDelta[x][y][z];
                        }
                    }
                    gradient.biasDelta[x] += gradients[i].biasDelta[x];
                }
            }

            for (int x = 0; x < gradient.weightDelta.Length; ++x) {
                for (int y = 0; y < gradient.weightDelta[x].Length; ++y) {
                    for (int z = 0; z < gradient.weightDelta[x][y].Length; ++z) {
                        TargetNetwork.Layers[x].Weights[y][z] -= (gradient.weightDelta[x][y][z] / gradients.Count) * DELTA_ADJUSTMENT;
                    }
                }
                TargetNetwork.Layers[x].Bias -= (gradient.biasDelta[x] / gradients.Count) * DELTA_ADJUSTMENT;
            }

            gradients.Clear();
        }

        private (double[][][] weightDelta, double[] biasDelta) CalculateGradient(double[] target, double[] actual) {
            double[][][] weightGradient = new double[TargetNetwork.Layers.Length][][];
            double[] biasGradient = new double[TargetNetwork.Layers.Length];

            List<Task> taskDump = new List<Task>();

            for (int _l = 0; _l < TargetNetwork.Layers.Length; ++_l) {
                weightGradient[_l] = new double[TargetNetwork.Structure[_l]][];

                int l = _l;

                for (int _i = 0; _i < TargetNetwork.Structure[_l]; ++_i) {
                    int i = _i; // Cuz when it adds it will break reference with i, but not if the thread still uses the _i reference
                    
                    taskDump.Add(Task.Factory.StartNew(() => {
                        weightGradient[l][i] = new double[TargetNetwork.Structure[l + 1]];
                        for (int o = 0; o < TargetNetwork.Structure[l + 1]; ++o) {
                            double weightSum = 0;
                            if ((TargetNetwork.Layers.Length - 1) == l) {
                                weightSum = 2 * (actual[o] - target[o]) * Derivative(TargetNetwork.Layers.Length - 1, o, l, (i, o), false);
                            } else {
                                for (int resultNode = 0; resultNode < target.Length; ++resultNode) {
                                    weightSum += 2 * (actual[resultNode] - target[resultNode]) * Derivative(TargetNetwork.Layers.Length - 1, resultNode, l, (i, o), false);
                                }
                            }

                            weightGradient[l][i][o] = weightSum;
                        }
                    }));
                }

                taskDump.Add(Task.Factory.StartNew(() => {
                    double biasSum = 0;
                    for (int resultNode = 0; resultNode < target.Length; ++resultNode) {
                        biasSum += 2 * (actual[resultNode] - target[resultNode]) * Derivative(TargetNetwork.Layers.Length - 1, resultNode, l, default, true);
                    }
                    biasGradient[l] = biasSum;
                }));
            }

            // while (true) {
            //     int completed = 0;
            //     taskDump.ForEach(x => completed += x.IsCompleted ? 1 : 0); // Halt until all tasks complete
            //     Console.WriteLine($"{completed} / {taskDump.Count}");
            //     if (completed >= taskDump.Count) {
            //         break;
            //     } else {
            //         Thread.Sleep(1000);
            //     }
            // }
            taskDump.ForEach(x => x.Wait());

            return (weightGradient, biasGradient);
        }

        public double Derivative(int currentLayer, int currentNode, int targetLayer, (int prev, int outp) targetWeight, bool getBias) {
            double deriv = TargetNetwork.Layers[currentLayer].ActivationDerivatives[currentNode];

            if (getBias) {
                if (currentLayer != targetLayer) {
                    double sum = 0;
                    for (int i = 0; i < TargetNetwork.Structure[currentLayer]; ++i) {
                        sum += TargetNetwork.Layers[currentLayer].Weights[i][currentNode]
                            * Derivative(currentLayer - 1, i, targetLayer, targetWeight, getBias);
                    }
                    deriv *= sum;
                }
            } else {
                if (currentLayer == targetLayer) {
                    deriv *= TargetNetwork.Layers[currentLayer].PreviousActivation[targetWeight.prev];
                } else if (currentLayer - targetLayer < 2) {
                    deriv *= TargetNetwork.Layers[currentLayer].Weights[targetWeight.outp][currentNode]
                        * Derivative(currentLayer - 1, targetWeight.outp, targetLayer, targetWeight, getBias);
                } else {
                    double sum = 0;
                    for (int i = 0; i < TargetNetwork.Structure[currentLayer]; ++i) {
                        sum += TargetNetwork.Layers[currentLayer].Weights[i][currentNode]
                            * Derivative(currentLayer - 1, i, targetLayer, targetWeight, getBias);
                    }
                    deriv *= sum;
                }
            }

            return deriv;
        }

        
    }

    public struct NetworkData {
        public int[] Structure;
        public double[][][] Weights;
        public double[] Biases;
    }

    public class Network {
        public int[] Structure;
        public Layer[] Layers;

        public Network(int[] structure) {
            Structure = structure;
            Init();
        }

        public Network(NetworkData data) {
            Structure = data.Structure;
            Layers = new Layer[Structure.Length - 1];
            for (int i = 0; i < Structure.Length - 1; i++) {
                Layers[i] = new Layer(data.Weights[i], data.Biases[i], Sigmoid, SigmoidDer);
            }
        }

        public NetworkData GetNetworkData() {
            double[][][] weights = new double[Layers.Length][][];
            double[] biases = new double[Layers.Length];
            for (int i = 0; i < Layers.Length; ++i) {
                weights[i] = Layers[i].Weights;
                biases[i] = Layers[i].Bias;
            }
            return new NetworkData { Structure = Structure, Weights = weights, Biases = biases };
        }

        private void Init() {
            Layers = new Layer[Structure.Length - 1];
            for (int l = 0; l < Structure.Length - 1; l++) {
                Layers[l] = new Layer(new int[] { Structure[l], Structure[l + 1] }, Sigmoid, SigmoidDer);
            }
        }

        public double[] Calculate(double[] input) {
            for (int l = 0; l < Layers.Length; ++l) {
                input = Layers[l].Calculate(input);
            }
            return input;
        }

        public static double Sigmoid(double a) => 1 / (1 + Math.Pow(Math.E, -a));
        public static double SigmoidDer(double a) => Sigmoid(a) * (1 - Sigmoid(a));
    }

    public class Layer {
        private int[] _structure;
        public double Bias;
        public double[][] Weights; // [input node][output node]
        public double[] PreviousActivation;
        public double[] BeforeActivation;
        private double[] _afterActivation;
        private Func<double, double> _activation;
        private Func<double, double> _activationDeriv;

        public double[] ActivationDerivatives;

        private int outputNodeCount => _structure[1];
        private int inputNodeCount => _structure[0];

        public Layer(int[] structure, Func<double, double> activation, Func<double, double> activationDeriv) {
            _structure = structure;
            _activation = activation;
            _activationDeriv = activationDeriv;
            Randomize();
        }

        public Layer(double[][] weights, double bias, Func<double, double> activation, Func<double, double> activationDeriv) {
            _structure = new int[] { weights.Length, weights[0].Length };
            _activation = activation;
            _activationDeriv = activationDeriv;
            Weights = weights;
            Bias = bias;
        }

        private void Randomize() {

            Weights = new double[inputNodeCount][];
            for (int inputNode = 0; inputNode < inputNodeCount; ++inputNode) {
                Weights[inputNode] = new double[outputNodeCount];
                for (int outputNode = 0; outputNode < outputNodeCount; ++outputNode) {
                    Weights[inputNode][outputNode] = (Rando.NextDouble() * 5) - 2.5;
                }
            }
            Bias = (Rando.NextDouble() * 2) - 1;
        }

        public double[] Calculate(double[] input) {
            PreviousActivation = input;
            BeforeActivation = new double[outputNodeCount];
            _afterActivation = new double[outputNodeCount];
            ActivationDerivatives = new double[outputNodeCount];

            for (int o = 0; o < outputNodeCount; ++o) {
                double productSum = Bias;
                for (int i = 0; i < inputNodeCount; ++i) {
                    productSum += Weights[i][o] * input[i];
                }
                BeforeActivation[o] = productSum;
                ActivationDerivatives[o] = _activationDeriv(BeforeActivation[o]);
                _afterActivation[o] = _activation(BeforeActivation[o]);
            }

            return _afterActivation;
        }

        public LayerTrainingMeta CalculateTraining(double[] input) {
            var layerMeta = new LayerTrainingMeta();
            layerMeta.PreviousActivation = input;
            layerMeta.BeforeActivation = new double[outputNodeCount];
            layerMeta.AfterActivation = new double[outputNodeCount];
            layerMeta.ActivationDerivatives = new double[outputNodeCount];

            for (int o = 0; o < outputNodeCount; ++o) {
                double productSum = Bias;
                for (int i = 0; i < inputNodeCount; ++i) {
                    productSum += Weights[i][o] * input[i];
                }
                layerMeta.BeforeActivation[o] = productSum;
                layerMeta.ActivationDerivatives[o] = _activationDeriv(layerMeta.BeforeActivation[o]);
                layerMeta.AfterActivation[o] = _activation(layerMeta.BeforeActivation[o]);
            }

            return layerMeta;
        }
        
    }

    public struct LayerTrainingMeta {
        public double[] PreviousActivation;
        public double[] BeforeActivation;
        public double[] AfterActivation;
        public double[] ActivationDerivatives;
    }

    public struct TestCase {
        public double[] inputs;
        public double[] outputs;
    }
}
